# =============================================================================
# FILE: quant_arena/zoo/estrategias/ejemplo_momentum.py
# Estrategia de Momentum de Sección Transversal — implementación de referencia
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Literal

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo, normalizar_pesos, señales_a_binario


@RegistroZoo.registrar('momentum_126d')
class MomentumStrategy(AbstractStrategy):
    """
    Momentum de sección transversal con ventana paramétrica (default: 126 días hábiles).

    Lógica vectorizada:
        1. Para cada activo: señal = precio(t) / precio(t - ventana) - 1
           (retorno acumulado bruto sobre la ventana, sin skip period).
        2. Seleccionar los `top_n` activos con mayor señal.
        3. Asignar pesos según `esquema_pesos`:
           - 'equal':        1/N igual para cada activo seleccionado.
           - 'proporcional': peso ∝ señal positiva (rank-weighting opcional).
           - 'rank':         peso ∝ rango ordinal (suaviza el efecto outlier).

    Garantías:
        - generar_señales() solo accede a datos <= fecha_corte (causalidad estricta).
        - Todas las operaciones sobre el DataFrame de precios son vectorizadas
          (sin bucles Python sobre filas o activos).
        - La suma de pesos retornada es exactamente 1.0 o 0.0 (universo insuficiente).
    """

    _ESQUEMAS_VALIDOS = frozenset({'equal', 'proporcional', 'rank'})

    def __init__(
        self,
        universo: List[str],
        ventana: int = 126,
        top_n: int = 10,
        esquema_pesos: Literal['equal', 'proporcional', 'rank'] = 'equal',
        max_peso_activo: float = 1.0,
    ) -> None:
        """
        Args:
            universo:         Lista de tickers que componen el universo invertible.
            ventana:          Días hábiles de lookback para calcular el retorno
                              de momentum. Default: 126 (~6 meses).
            top_n:            Número de activos a seleccionar en el portfolio.
            esquema_pesos:    Método de asignación de pesos dentro del top N.
                              'equal'       → 1/top_n por activo.
                              'proporcional'→ ∝ magnitud de señal positiva.
                              'rank'        → ∝ rango ordinal (1/rank normalizado).
            max_peso_activo:  Límite superior de concentración por activo (0-1].
                              Activa el truncamiento iterativo en normalizar_pesos().
        """
        super().__init__(nombre='momentum_126d', universo=universo)

        if ventana < 2:
            raise ValueError("ventana debe ser >= 2 días.")
        if top_n < 1:
            raise ValueError("top_n debe ser >= 1.")
        if esquema_pesos not in self._ESQUEMAS_VALIDOS:
            raise ValueError(
                f"esquema_pesos inválido: '{esquema_pesos}'. "
                f"Opciones: {self._ESQUEMAS_VALIDOS}"
            )
        if not 0.0 < max_peso_activo <= 1.0:
            raise ValueError("max_peso_activo debe estar en (0, 1].")

        self._ventana: int = ventana
        self._top_n: int = top_n
        self._esquema: str = esquema_pesos
        self._max_peso: float = max_peso_activo

    @property
    def descripcion(self) -> str:
        return (
            f"Momentum XS {self._ventana}d | top {self._top_n} | "
            f"pesos={self._esquema} | max_pos={self._max_peso:.0%}"
        )

    # ------------------------------------------------------------------
    # Núcleo de generación de señales (vectorizado, causal)
    # ------------------------------------------------------------------

    def generar_señales(
        self,
        datos: pd.DataFrame,
        fecha_corte: pd.Timestamp,
    ) -> pd.Series:
        """
        Genera pesos por activo usando solo precios hasta `fecha_corte`.

        Implementación 100% vectorizada:
            - Un slice temporal para aplicar el corte causal.
            - Una división de vectores para calcular retornos de momentum.
            - rank() para selección top N sin sort explícito.
            - normalizar_pesos() para asignación final.

        Args:
            datos:       DataFrame de precios ajustados [fecha x ticker].
                         Debe contener todas las columnas de self._universo.
            fecha_corte: Límite duro — datos posteriores son invisibles.

        Returns:
            pd.Series [ticker → peso] con suma == 1.0 si hay >= top_n activos
            con suficiente historia, 0.0 en caso contrario.
        """
        # RESTRICCIÓN CAUSAL: nunca ver datos posteriores a fecha_corte
        precios_hist: pd.DataFrame = datos.loc[
            datos.index <= fecha_corte,
            [col for col in self._universo if col in datos.columns],
        ]

        activos_disponibles: List[str] = precios_hist.columns.tolist()
        indice_completo = pd.Index(self._universo)

        # Verificar historia mínima (necesitamos ventana + 1 observaciones)
        if len(precios_hist) <= self._ventana:
            return pd.Series(0.0, index=indice_completo, dtype=float)

        # --- Señal de momentum vectorizada ---
        # precio_actual: vector de precios en la última fecha disponible
        # precio_pasado: vector de precios hace `ventana` días
        # Ambas operaciones son O(N_activos) sin bucle Python
        precio_actual: pd.Series = precios_hist.iloc[-1]
        precio_pasado: pd.Series = precios_hist.iloc[-(self._ventana + 1)]

        # Evitar división por cero: activos con precio_pasado == 0 quedan con NaN
        señal_raw: pd.Series = precio_actual / precio_pasado - 1.0
        señal_raw = señal_raw.replace([np.inf, -np.inf], np.nan).dropna()

        if len(señal_raw) == 0:
            return pd.Series(0.0, index=indice_completo, dtype=float)

        # Número efectivo de activos seleccionables
        top_n_efectivo: int = min(self._top_n, len(señal_raw))

        # --- Selección top N (vectorizada via rank) ---
        seleccion: pd.Series = señales_a_binario(señal_raw, top_n=top_n_efectivo)
        señal_seleccionada: pd.Series = señal_raw[seleccion > 0]

        if señal_seleccionada.empty:
            return pd.Series(0.0, index=indice_completo, dtype=float)

        # --- Esquema de pesos ---
        pesos_seleccion: pd.Series = self._calcular_pesos(señal_seleccionada)

        # Reindexar al universo completo (activos no seleccionados → 0.0)
        pesos_completo: pd.Series = pesos_seleccion.reindex(indice_completo, fill_value=0.0)

        return pesos_completo

    def _calcular_pesos(self, señales_top: pd.Series) -> pd.Series:
        """
        Convierte las señales del top N seleccionado en pesos normalizados.
        Método delegado desde generar_señales(); opera solo sobre activos elegidos.
        """
        if self._esquema == 'equal':
            n: int = len(señales_top)
            return pd.Series(1.0 / n, index=señales_top.index, dtype=float)

        if self._esquema == 'proporcional':
            # Peso ∝ señal positiva; señales negativas en el top se descartan
            pesos_crudos = señales_top.clip(lower=0.0)
            return normalizar_pesos(pesos_crudos, long_only=True, max_peso=self._max_peso)

        if self._esquema == 'rank':
            # Peso ∝ 1/rango (rango 1 = mayor señal → mayor peso)
            # Suaviza el impacto de outliers positivos respecto a 'proporcional'
            rangos: pd.Series = señales_top.rank(ascending=False)   # 1 = mejor
            pesos_rank: pd.Series = 1.0 / rangos
            return normalizar_pesos(pesos_rank, long_only=True, max_peso=self._max_peso)

        # Rama defensiva — no debería alcanzarse por la validación en __init__
        raise ValueError(f"Esquema de pesos no soportado: '{self._esquema}'")

    # ------------------------------------------------------------------
    # Cálculo de retornos históricos
    # ------------------------------------------------------------------

    def calcular_retornos(
        self,
        datos: pd.DataFrame,
        pesos_historicos: pd.DataFrame,
    ) -> pd.Series:
        """
        Aplica los pesos históricos precalculados por el Motor y retorna
        la serie de retornos diarios de la estrategia.

        El Motor precalcula pesos_historicos ejecutando generar_señales() para
        cada fecha en la ventana de backtesting; este método solo aplica
        los pesos para obtener el P&L.

        Vectorizado: retornos_activos y dot product son operaciones de Pandas/NumPy.

        Args:
            datos:             DataFrame de precios ajustados [fecha x ticker].
            pesos_historicos:  DataFrame [fecha x ticker] con pesos diarios.

        Returns:
            pd.Series de retornos diarios (sin NaN del primer día por pct_change).
        """
        activos: List[str] = [c for c in self._universo if c in datos.columns]

        # Retornos diarios simples de los activos: vectorizado via pct_change
        retornos_activos: pd.DataFrame = datos[activos].pct_change()

        # Alinear pesos y retornos en el eje temporal
        pesos_al, ret_al = pesos_historicos[activos].align(
            retornos_activos, join='inner', axis=0
        )

        # Retorno del portafolio: suma ponderada diaria — dot product vectorizado
        retornos_portfolio: pd.Series = (pesos_al * ret_al).sum(axis=1)

        return retornos_portfolio.dropna()
