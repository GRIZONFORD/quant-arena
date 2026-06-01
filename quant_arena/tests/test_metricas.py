# =============================================================================
# FILE: quant_arena/tests/test_metricas.py
# Suite pytest para PerformanceMetrics — casos normales y casos límite.
#
# Cobertura:
#   - ValueError para series cortas (< 20 obs) y con NaN
#   - Retornos = rfr_diaria => Sharpe = 0.0 (std de excesos = 0)
#   - Sin retornos negativos => Sortino = inf
#   - MDD siempre <= 0; retornos monotone => MDD = 0
#   - calcular_todas() retorna MetricasResultado con todos los campos
#   - calcular_rolling() respeta NaN iniciales y tamaño de ventana
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_arena.metricas.performance_metrics import PerformanceMetrics
from quant_arena.core.abstracciones import MetricasResultado


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pm() -> PerformanceMetrics:
    return PerformanceMetrics(tasa_libre_riesgo_anual=0.02)


@pytest.fixture
def retornos_normales() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.01, 252))


@pytest.fixture
def benchmark_normales() -> pd.Series:
    rng = np.random.default_rng(123)
    return pd.Series(rng.normal(0.0003, 0.01, 252))


# ---------------------------------------------------------------------------
# Validación: ValueError en casos límite
# ---------------------------------------------------------------------------

class TestValidacion:
    def test_sharpe_menos_de_20_obs_lanza_error(self, pm: PerformanceMetrics) -> None:
        with pytest.raises(ValueError, match="20"):
            pm.sharpe_ratio(pd.Series(np.zeros(19)))

    def test_sharpe_serie_con_nan_lanza_error(self, pm: PerformanceMetrics) -> None:
        serie = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50))
        serie.iloc[25] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            pm.sharpe_ratio(serie)

    def test_sortino_menos_de_20_obs_lanza_error(self, pm: PerformanceMetrics) -> None:
        with pytest.raises(ValueError):
            pm.sortino_ratio(pd.Series([0.01] * 5))

    def test_calmar_menos_de_20_obs_lanza_error(self, pm: PerformanceMetrics) -> None:
        with pytest.raises(ValueError):
            pm.calmar_ratio(pd.Series([0.01] * 10))

    def test_calcular_todas_serie_corta_lanza_error(
        self,
        pm: PerformanceMetrics,
        benchmark_normales: pd.Series,
    ) -> None:
        with pytest.raises(ValueError):
            pm.calcular_todas(pd.Series([0.01, -0.01, 0.02]), benchmark_normales)

    def test_information_ratio_nan_lanza_error(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
    ) -> None:
        bench_con_nan = retornos_normales.copy()
        bench_con_nan.iloc[10] = np.nan
        with pytest.raises(ValueError):
            pm.information_ratio(retornos_normales, bench_con_nan)


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_retornos_iguales_a_rfr_diaria_da_cero(self, pm: PerformanceMetrics) -> None:
        rfr_d = 0.02 / 252
        retornos = pd.Series([rfr_d] * 50)
        assert pm.sharpe_ratio(retornos) == 0.0

    def test_caso_normal_es_finito(
        self, pm: PerformanceMetrics, retornos_normales: pd.Series
    ) -> None:
        assert np.isfinite(pm.sharpe_ratio(retornos_normales))

    def test_retornos_positivos_tienen_sharpe_positivo(
        self, pm: PerformanceMetrics
    ) -> None:
        # Serie con varianza y media > rfr_diaria → Sharpe > 0
        rng = np.random.default_rng(9)
        retornos = pd.Series(rng.normal(0.005, 0.002, 50))
        assert pm.sharpe_ratio(retornos) > 0.0

    def test_retornos_negativos_tienen_sharpe_negativo(
        self, pm: PerformanceMetrics
    ) -> None:
        # Serie con varianza y media << rfr_diaria → excesos negativos → Sharpe < 0
        rng = np.random.default_rng(11)
        retornos = pd.Series(rng.normal(-0.005, 0.002, 50))
        assert pm.sharpe_ratio(retornos) < 0.0


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_sin_retornos_negativos_retorna_inf(self, pm: PerformanceMetrics) -> None:
        retornos = pd.Series([0.005] * 50)
        assert pm.sortino_ratio(retornos) == np.inf

    def test_caso_normal_es_finito(
        self, pm: PerformanceMetrics, retornos_normales: pd.Series
    ) -> None:
        assert np.isfinite(pm.sortino_ratio(retornos_normales))

    def test_solo_retornos_negativos_es_negativo(
        self, pm: PerformanceMetrics
    ) -> None:
        retornos = pd.Series([-0.003] * 50)
        assert pm.sortino_ratio(retornos) < 0.0


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_siempre_no_positivo(
        self, pm: PerformanceMetrics, retornos_normales: pd.Series
    ) -> None:
        assert pm.max_drawdown(retornos_normales) <= 0.0

    def test_retornos_monotone_positivos_da_cero(
        self, pm: PerformanceMetrics
    ) -> None:
        retornos = pd.Series([0.001] * 50)
        assert pm.max_drawdown(retornos) == pytest.approx(0.0, abs=1e-12)

    def test_acepta_solo_dos_observaciones(self, pm: PerformanceMetrics) -> None:
        resultado = pm.max_drawdown(pd.Series([-0.05, 0.03]))
        assert resultado <= 0.0


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_mdd_cero_retorna_nan(self, pm: PerformanceMetrics) -> None:
        # Serie monotone creciente: MDD = 0 → Calmar = NaN
        retornos = pd.Series([0.001] * 50)
        assert np.isnan(pm.calmar_ratio(retornos))

    def test_caso_normal_es_float(
        self, pm: PerformanceMetrics, retornos_normales: pd.Series
    ) -> None:
        resultado = pm.calmar_ratio(retornos_normales)
        assert isinstance(resultado, float)


# ---------------------------------------------------------------------------
# Information Ratio y Alpha T-stat
# ---------------------------------------------------------------------------

class TestMetricasConBenchmark:
    def test_information_ratio_serie_identica_da_cero(
        self, pm: PerformanceMetrics, retornos_normales: pd.Series
    ) -> None:
        ir = pm.information_ratio(retornos_normales, retornos_normales)
        assert ir == 0.0

    def test_information_ratio_caso_normal_es_finito(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        assert np.isfinite(pm.information_ratio(retornos_normales, benchmark_normales))

    def test_alpha_tstat_caso_normal_es_finito(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        assert np.isfinite(pm.alpha_tstat(retornos_normales, benchmark_normales))


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_pesos_constantes_da_cero(self, pm: PerformanceMetrics) -> None:
        pesos = pd.DataFrame(
            np.full((50, 3), 1 / 3), columns=["A", "B", "C"]
        )
        assert pm.turnover(pesos) == pytest.approx(0.0, abs=1e-12)

    def test_un_solo_periodo_da_cero(self, pm: PerformanceMetrics) -> None:
        pesos = pd.DataFrame([[0.4, 0.6]], columns=["A", "B"])
        assert pm.turnover(pesos) == 0.0

    def test_rotacion_total_es_casi_dos(self, pm: PerformanceMetrics) -> None:
        # Alternancia completa cada período: cambio absoluto = 2 por período.
        # diff() produce NaN en la primera fila; sum(axis=1, skipna=True) = 0 ahí.
        # Resultado: mean = 2*(n-1)/n. Con n=50 → 1.96.
        n = 50
        pesos = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]] * (n // 2), columns=["A", "B"])
        esperado = 2.0 * (n - 1) / n
        assert pm.turnover(pesos) == pytest.approx(esperado, abs=1e-10)


# ---------------------------------------------------------------------------
# calcular_todas — punto de entrada unificado
# ---------------------------------------------------------------------------

class TestCalcularTodas:
    def test_retorna_metricas_resultado(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        resultado = pm.calcular_todas(retornos_normales, benchmark_normales)
        assert isinstance(resultado, MetricasResultado)

    def test_todos_los_campos_finitos_caso_normal(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        r = pm.calcular_todas(retornos_normales, benchmark_normales)
        assert np.isfinite(r.sharpe)
        assert np.isfinite(r.sortino)
        assert r.max_drawdown <= 0.0
        assert np.isfinite(r.information_ratio)
        assert np.isfinite(r.alpha_tstat)
        assert np.isnan(r.turnover)  # pesos=None por defecto

    def test_con_pesos_calcula_turnover(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        rng = np.random.default_rng(7)
        pesos = pd.DataFrame(
            rng.dirichlet([1] * 5, size=252), columns=[f"T{i}" for i in range(5)]
        )
        r = pm.calcular_todas(retornos_normales, benchmark_normales, pesos=pesos)
        assert np.isfinite(r.turnover)
        assert r.turnover >= 0.0


# ---------------------------------------------------------------------------
# calcular_rolling — ventana causal
# ---------------------------------------------------------------------------

class TestCalcularRolling:
    def test_longitud_igual_a_retornos(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        ventana = 63
        resultado = pm.calcular_rolling(
            retornos_normales, benchmark_normales, ventana=ventana
        )
        assert len(resultado) == len(retornos_normales)

    def test_primeros_ventana_menos_uno_son_nan(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        ventana = 63
        resultado = pm.calcular_rolling(
            retornos_normales, benchmark_normales, ventana=ventana
        )
        assert resultado.iloc[: ventana - 1].isna().all()

    def test_despues_de_ventana_hay_valores_finitos(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        ventana = 63
        resultado = pm.calcular_rolling(
            retornos_normales, benchmark_normales, ventana=ventana
        )
        assert resultado.iloc[ventana - 1 :].notna().any()

    def test_metrica_invalida_lanza_error(
        self,
        pm: PerformanceMetrics,
        retornos_normales: pd.Series,
        benchmark_normales: pd.Series,
    ) -> None:
        with pytest.raises(ValueError, match="inexistente"):
            pm.calcular_rolling(
                retornos_normales, benchmark_normales, metrica="inexistente"
            )
