# =============================================================================
# FILE: quant_arena/metricas/filtros.py
# Filtro de Kalman unidimensional para suavizado causal de señales financieras
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, NamedTuple, Optional, Tuple


class _EstadoKalman(NamedTuple):
    """Estado interno del filtro en un instante t."""
    x: float   # Media del estado estimado
    P: float   # Varianza del error de estimación


class KalmanSignalFilter:
    """
    Filtro de Kalman escalar (1-D) para suavizado óptimo causal de métricas.

    Modelo de espacio de estados (random walk + ruido de medición):
        Proceso:  x_t  = x_{t-1} + w_t,   w_t ~ N(0, Q)
        Medición: z_t  = x_t    + v_t,   v_t ~ N(0, R)

    Interpretación de parámetros:
        Q pequeño → el estado "verdadero" cambia poco; señal muy suave, alto lag.
        Q grande  → el estado cambia rápidamente; filtro más reactivo.
        R grande  → poca confianza en cada observación individual.
        Ratio Q/R calibra el balance bias-varianza del suavizado.

    En estado estacionario la ganancia converge a:
        K* = u / (u + R),   u = Q/2 + sqrt(Q²/4 + QR)
    que es equivalente a un EWMA con alfa = K*. Este resultado se usa en
    `filtrar_rapido()` como aproximación vectorizada para series largas.
    """

    def __init__(
        self,
        Q: float = 1e-4,
        R: float = 1e-2,
        P0: float = 1.0,
    ) -> None:
        """
        Args:
            Q:  Varianza del ruido de proceso (velocidad de cambio del estado).
                Valores típicos para Sharpe ratio mensual: 1e-5 – 1e-3.
            R:  Varianza del ruido de medición (nivel de ruido observacional).
                Valores típicos: 1e-2 – 1e-1.
            P0: Varianza inicial del error de estimación (prior difuso).
                Un valor alto (ej. 1.0) indica máxima incertidumbre inicial.
        """
        if Q <= 0 or R <= 0 or P0 <= 0:
            raise ValueError("Q, R y P0 deben ser estrictamente positivos.")
        self._Q: float = Q
        self._R: float = R
        self._P0: float = P0

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------

    @property
    def ganancia_estacionaria(self) -> float:
        """
        Ganancia de Kalman en régimen estacionario (solución de la ecuación de Riccati).

        Derivación: la ecuación de Riccati discreta escalar en estado estacionario
        P_ss satisface P_ss = R*(P_ss + Q)/(P_ss + Q + R).
        Sustituyendo u = P_ss + Q se obtiene u² - Q*u - Q*R = 0, cuya raíz positiva
        es u = Q/2 + sqrt(Q²/4 + Q*R), y por tanto K* = u/(u + R).
        """
        u: float = self._Q / 2.0 + np.sqrt(self._Q ** 2 / 4.0 + self._Q * self._R)
        return u / (u + self._R)

    @property
    def periodos_convergencia(self) -> int:
        """
        Número aproximado de períodos para que el filtro alcance el 99% de K*.
        Se estima como ceil(log(0.01) / log(1 - K*)).
        """
        k_ss = self.ganancia_estacionaria
        if k_ss >= 1.0:
            return 1
        return int(np.ceil(np.log(0.01) / np.log(max(1.0 - k_ss, 1e-12))))

    # ------------------------------------------------------------------
    # Filtro exacto (secuencial — causal por construcción)
    # ------------------------------------------------------------------

    def filtrar(
        self,
        serie: pd.Series,
        x0: Optional[float] = None,
    ) -> pd.Series:
        """
        Aplica el filtro de Kalman causal elemento a elemento.

        NaN tratados como observaciones ausentes: se ejecuta el paso de
        predicción pero NO el de corrección, propagando el último estado
        válido sin modificación de la covarianza de error actualizada.

        El bucle sobre el eje temporal es inherente al algoritmo secuencial
        de Kalman; no puede vectorizarse sin asumir estado estacionario.
        Para una aproximación vectorizada ver `filtrar_rapido()`.

        Args:
            serie: Serie temporal de la señal (pd.Series con cualquier índice).
            x0:    Estado inicial. Si None, se usa la primera observación válida.

        Returns:
            pd.Series con el mismo índice que `serie`. NaN donde el estado
            aún no fue inicializado (antes del primer valor válido).
        """
        valores: np.ndarray = serie.to_numpy(dtype=float, na_value=np.nan)
        n: int = len(valores)
        salida: np.ndarray = np.full(n, np.nan, dtype=float)

        # Localizar primer valor válido para inicializar el estado
        indices_validos = np.where(~np.isnan(valores))[0]
        if len(indices_validos) == 0:
            return pd.Series(salida, index=serie.index, name=serie.name)

        inicio: int = int(indices_validos[0])
        x: float = x0 if x0 is not None else float(valores[inicio])
        P: float = self._P0

        for i in range(inicio, n):
            # --- Predicción ---
            x_pred: float = x
            P_pred: float = P + self._Q

            z: float = valores[i]

            if np.isnan(z):
                # Observación ausente: propaga estado sin corrección
                x = x_pred
                P = P_pred
            else:
                # --- Corrección ---
                K: float = P_pred / (P_pred + self._R)   # Ganancia de Kalman
                x = x_pred + K * (z - x_pred)            # Media actualizada
                P = (1.0 - K) * P_pred                   # Varianza actualizada

            salida[i] = x

        return pd.Series(salida, index=serie.index, name=serie.name)

    def filtrar_con_varianza(
        self,
        serie: pd.Series,
        x0: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Igual que `filtrar()` pero retorna también la varianza posterior P_t.

        Returns:
            (serie_filtrada, serie_varianza) — ambas con el índice de `serie`.
        """
        valores: np.ndarray = serie.to_numpy(dtype=float, na_value=np.nan)
        n: int = len(valores)
        salida_x: np.ndarray = np.full(n, np.nan, dtype=float)
        salida_P: np.ndarray = np.full(n, np.nan, dtype=float)

        indices_validos = np.where(~np.isnan(valores))[0]
        if len(indices_validos) == 0:
            return (
                pd.Series(salida_x, index=serie.index, name=serie.name),
                pd.Series(salida_P, index=serie.index, name=f"{serie.name}_var"),
            )

        inicio: int = int(indices_validos[0])
        x: float = x0 if x0 is not None else float(valores[inicio])
        P: float = self._P0

        for i in range(inicio, n):
            x_pred: float = x
            P_pred: float = P + self._Q
            z: float = valores[i]

            if np.isnan(z):
                x, P = x_pred, P_pred
            else:
                K: float = P_pred / (P_pred + self._R)
                x = x_pred + K * (z - x_pred)
                P = (1.0 - K) * P_pred

            salida_x[i] = x
            salida_P[i] = P

        return (
            pd.Series(salida_x, index=serie.index, name=serie.name),
            pd.Series(salida_P, index=serie.index, name=f"{serie.name}_var"),
        )

    # ------------------------------------------------------------------
    # Aproximación vectorizada (estado estacionario)
    # ------------------------------------------------------------------

    def filtrar_rapido(self, serie: pd.Series) -> pd.Series:
        """
        Aproximación vectorizada basada en la ganancia estacionaria K*.

        Una vez que el filtro converge (tras ~`periodos_convergencia` pasos),
        la recursión x_t = x_{t-1} + K*(z_t - x_{t-1}) es idéntica a un EWMA
        con alfa = K*. Esta implementación usa `pd.Series.ewm()` directamente.

        Ventaja: O(N) sin loop Python; apto para series de > 10 000 puntos.
        Limitación: ignora la fase transitoria inicial; usar `filtrar()` cuando
        la precisión en los primeros `periodos_convergencia` períodos importe.

        NaN son interpolados linealmente antes de aplicar el EWMA y restaurados.
        """
        k_ss: float = self.ganancia_estacionaria
        # Rellenar NaN solo internamente para que ewm funcione; restaurar al final
        serie_sin_nan = serie.interpolate(method='linear', limit_direction='forward')
        filtrada = serie_sin_nan.ewm(alpha=k_ss, adjust=False).mean()
        # Restaurar NaN originales (posiciones donde serie era NaN desde el inicio)
        filtrada[serie.isna()] = np.nan
        return filtrada.rename(serie.name)

    # ------------------------------------------------------------------
    # Operaciones batch sobre DataFrame y diccionarios de métricas
    # ------------------------------------------------------------------

    def filtrar_dataframe(
        self,
        df: pd.DataFrame,
        rapido: bool = False,
    ) -> pd.DataFrame:
        """
        Aplica el filtro independientemente a cada columna del DataFrame.

        Args:
            df:     DataFrame [tiempo x señales].
            rapido: Si True usa `filtrar_rapido()`; si False usa `filtrar()`.

        Returns:
            DataFrame con las mismas dimensiones e índice que `df`.
        """
        metodo = self.filtrar_rapido if rapido else self.filtrar
        return df.apply(metodo, axis=0)

    def filtrar_historial_metricas(
        self,
        historial: Dict[str, List[Tuple[pd.Timestamp, float]]],
        campo: str = 'sharpe',
    ) -> Dict[str, pd.Series]:
        """
        Filtra un historial periódico de métricas generado por el Motor.

        Args:
            historial: {nombre_estrategia: [(timestamp, valor_metrica), ...]}
                       Formato de salida típico del Motor de Backtesting.
            campo:     Nombre del campo (solo documentativo; los valores ya
                       deben estar extraídos del MetricasResultado por el llamador).

        Returns:
            {nombre_estrategia: pd.Series filtrada con DatetimeIndex}
        """
        resultado: Dict[str, pd.Series] = {}
        for nombre, puntos in historial.items():
            if not puntos:
                continue
            fechas, valores = zip(*puntos)
            serie_cruda = pd.Series(
                list(valores),
                index=pd.DatetimeIndex(list(fechas)),
                name=f"{nombre}_{campo}",
                dtype=float,
            )
            resultado[nombre] = self.filtrar(serie_cruda)
        return resultado
