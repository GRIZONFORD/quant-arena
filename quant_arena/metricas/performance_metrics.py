# =============================================================================
# FILE: quant_arena/metricas/performance_metrics.py
# Calculadora vectorizada de KPIs institucionales (NumPy / Pandas)
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional

from quant_arena.core.abstracciones import AbstractMetricas, MetricasResultado


class PerformanceMetrics(AbstractMetricas):
    """
    Calculadora stateless de métricas de performance institucional.

    Convenciones:
      - Retornos de entrada: series de retornos simples diarios (no log-retornos).
      - Factor de anualización: 252 días hábiles.
      - La tasa libre de riesgo se convierte a tasa diaria internamente.
    """

    DIAS_HABILES_ANIO: int = 252

    def __init__(self, tasa_libre_riesgo_anual: float = 0.0) -> None:
        """
        Args:
            tasa_libre_riesgo_anual: Tasa anual (ej. 0.05 = 5%). Se divide entre
                                     252 para obtener la tasa diaria equivalente.
        """
        self._rfr_anual: float = tasa_libre_riesgo_anual
        self._rfr_diaria: float = tasa_libre_riesgo_anual / self.DIAS_HABILES_ANIO

    # ------------------------------------------------------------------
    # Validación interna
    # ------------------------------------------------------------------

    @staticmethod
    def _validar(serie: pd.Series, nombre: str = 'retornos', min_obs: int = 20) -> None:
        if len(serie) < min_obs:
            raise ValueError(
                f"'{nombre}' requiere >= {min_obs} observaciones; se recibieron {len(serie)}."
            )
        if serie.isna().any():
            raise ValueError(f"'{nombre}' contiene NaN. Limpiar antes de calcular métricas.")

    # ------------------------------------------------------------------
    # Métricas base (implementan AbstractMetricas)
    # ------------------------------------------------------------------

    def sharpe_ratio(self, retornos: pd.Series) -> float:
        """
        Sharpe Ratio anualizado.
        SR = E[r_e] / std(r_e) * sqrt(252), donde r_e = retorno - rfr_diaria.
        """
        self._validar(retornos, 'retornos')
        excesos: pd.Series = retornos - self._rfr_diaria
        std: float = float(excesos.std(ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.sqrt(self.DIAS_HABILES_ANIO) * excesos.mean() / std)

    def sortino_ratio(self, retornos: pd.Series) -> float:
        """
        Sortino Ratio anualizado.
        Downside deviation = sqrt(E[min(r_e, 0)^2]) — penaliza solo retornos negativos.
        MAR implícita = tasa libre de riesgo diaria.
        """
        self._validar(retornos, 'retornos')
        excesos: pd.Series = retornos - self._rfr_diaria
        retornos_negativos: np.ndarray = excesos.values[excesos.values < 0.0]

        if len(retornos_negativos) == 0:
            return np.inf

        downside_dev: float = float(np.sqrt(np.mean(retornos_negativos ** 2)))
        if downside_dev == 0.0:
            return np.inf

        return float(np.sqrt(self.DIAS_HABILES_ANIO) * excesos.mean() / downside_dev)

    def max_drawdown(self, retornos: pd.Series) -> float:
        """
        Maximum Drawdown: caída máxima desde el pico acumulado histórico.
        Retorna valor negativo (ej. -0.35 = drawdown del 35%).
        Vectorizado mediante cummax de Pandas.
        """
        self._validar(retornos, 'retornos', min_obs=2)
        riqueza: pd.Series = (1.0 + retornos).cumprod()
        pico: pd.Series = riqueza.cummax()
        drawdowns: pd.Series = (riqueza - pico) / pico
        return float(drawdowns.min())

    def calmar_ratio(self, retornos: pd.Series) -> float:
        """
        Calmar Ratio = Retorno anual compuesto / |Maximum Drawdown|.
        Útil como proxy de eficiencia de riesgo en estrategias de baja frecuencia.
        """
        self._validar(retornos, 'retornos')
        mdd: float = abs(self.max_drawdown(retornos))
        if mdd == 0.0:
            return np.nan
        n: int = len(retornos)
        cagr: float = float((1.0 + retornos).prod() ** (self.DIAS_HABILES_ANIO / n) - 1.0)
        return cagr / mdd

    def information_ratio(
        self, retornos: pd.Series, benchmark: pd.Series
    ) -> float:
        """
        Information Ratio anualizado.
        IR = E[r_activo] / TE * sqrt(252)
        donde r_activo = retorno_estrategia - retorno_benchmark
        y TE = tracking error (std de r_activo).
        """
        self._validar(retornos, 'retornos')
        self._validar(benchmark, 'benchmark')
        ret_al, bench_al = retornos.align(benchmark, join='inner')
        activo: pd.Series = ret_al - bench_al
        te: float = float(activo.std(ddof=1))
        if te == 0.0:
            return 0.0
        return float(np.sqrt(self.DIAS_HABILES_ANIO) * activo.mean() / te)

    def alpha_tstat(
        self, retornos: pd.Series, benchmark: pd.Series
    ) -> float:
        """
        T-estadístico del alfa de Jensen vía OLS vectorizado.
        Modelo: r_p = alpha + beta * r_b + epsilon
        H0: alpha = 0  ->  t = alpha_hat / SE(alpha_hat)
        Usa pseudo-inversa (pinv) para estabilidad numérica con benchmarks constantes.
        """
        self._validar(retornos, 'retornos')
        self._validar(benchmark, 'benchmark')
        r_p, r_b = retornos.align(benchmark, join='inner')
        n: int = len(r_p)

        X: np.ndarray = np.column_stack([np.ones(n), r_b.values])
        y: np.ndarray = r_p.values

        XtX_inv: np.ndarray = np.linalg.pinv(X.T @ X)
        beta_hat: np.ndarray = XtX_inv @ X.T @ y
        residuos: np.ndarray = y - X @ beta_hat

        s2: float = float(np.sum(residuos ** 2) / max(n - 2, 1))
        var_alpha: float = float(s2 * XtX_inv[0, 0])

        if var_alpha <= 0.0:
            return np.nan
        return float(beta_hat[0] / np.sqrt(var_alpha))

    def turnover(self, pesos: pd.DataFrame) -> float:
        """
        Turnover promedio diario del portafolio.
        Definición: media de la suma de cambios absolutos en pesos por período.
        Un valor de 0.50 implica rotación del 50% del portafolio por período.
        """
        if pesos.shape[0] < 2:
            return 0.0
        return float(pesos.diff().abs().sum(axis=1).mean())

    def calcular_todas(
        self,
        retornos: pd.Series,
        benchmark: pd.Series,
        pesos: Optional[pd.DataFrame] = None,
    ) -> MetricasResultado:
        """
        Punto de entrada principal: calcula el conjunto completo de KPIs.

        Args:
            retornos:  Serie de retornos diarios de la estrategia.
            benchmark: Serie de retornos diarios del S&P 500.
            pesos:     DataFrame opcional [fecha x ticker] para calcular turnover.

        Returns:
            MetricasResultado con todos los indicadores institucionales.
        """
        return MetricasResultado(
            sharpe=self.sharpe_ratio(retornos),
            sortino=self.sortino_ratio(retornos),
            max_drawdown=self.max_drawdown(retornos),
            calmar=self.calmar_ratio(retornos),
            information_ratio=self.information_ratio(retornos, benchmark),
            alpha_tstat=self.alpha_tstat(retornos, benchmark),
            turnover=self.turnover(pesos) if pesos is not None else np.nan,
        )

    # ------------------------------------------------------------------
    # Utilidades de análisis rodante (causal)
    # ------------------------------------------------------------------

    _METRICA_MAPA: Dict[str, str] = {
        'sharpe':            'sharpe_ratio',
        'sortino':           'sortino_ratio',
        'max_drawdown':      'max_drawdown',
        'calmar':            'calmar_ratio',
        'information_ratio': 'information_ratio',
        'alpha_tstat':       'alpha_tstat',
    }
    _REQUIERE_BENCHMARK = frozenset({'information_ratio', 'alpha_tstat'})

    def calcular_rolling(
        self,
        retornos: pd.Series,
        benchmark: pd.Series,
        ventana: int = 63,
        metrica: str = 'sharpe',
    ) -> pd.Series:
        """
        Calcula una métrica sobre una ventana rodante estrictamente causal.
        Útil para generar el feature temporal que alimenta al Juez TTT período a período.

        Args:
            retornos:  Serie de retornos diarios.
            benchmark: Serie de retornos diarios del benchmark.
            ventana:   Tamaño de la ventana en días hábiles (default: 63 ≈ 1 trimestre).
            metrica:   Nombre de la métrica (ver _METRICA_MAPA).

        Returns:
            pd.Series de la métrica calculada rolling (NaN para los primeros ventana-1 días).
        """
        if metrica not in self._METRICA_MAPA:
            raise ValueError(
                f"Métrica '{metrica}' no disponible. Opciones: {list(self._METRICA_MAPA)}"
            )
        nombre_metodo: str = self._METRICA_MAPA[metrica]
        metodo: Callable = getattr(self, nombre_metodo)
        necesita_bench: bool = metrica in self._REQUIERE_BENCHMARK

        resultados: pd.Series = pd.Series(np.nan, index=retornos.index, dtype=float)

        for i in range(ventana, len(retornos) + 1):
            slice_ret = retornos.iloc[i - ventana : i]
            try:
                if necesita_bench:
                    slice_bench = benchmark.iloc[i - ventana : i]
                    resultados.iloc[i - 1] = metodo(slice_ret, slice_bench)
                else:
                    resultados.iloc[i - 1] = metodo(slice_ret)
            except ValueError:
                pass

        return resultados
