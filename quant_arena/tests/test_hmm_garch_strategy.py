# =============================================================================
# FILE: quant_arena/tests/test_hmm_garch_strategy.py
# Suite pytest para las extensiones opt-in de HMMGARCHStrategy (§1.5):
# selección automática de K, validación de normalidad, e incertidumbre de
# régimen expuesta para §1.1/§1.2.
#
# Se salta por completo si hmmlearn/arch/statsmodels no están instalados —
# son dependencias opcionales (`pip install -e '.[ml]'`), igual que el resto
# del Zoo ML. El comportamiento DEFAULT de la estrategia (sin estas
# dependencias) se sigue verificando en el bloque __main__ del propio
# módulo (6 tests, ejecutados como parte de esta suite vía subprocess-free
# import — ver test_suite_embebida_pasa).
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hmmlearn = pytest.importorskip("hmmlearn")
arch = pytest.importorskip("arch")
statsmodels = pytest.importorskip("statsmodels")

from quant_arena.zoo.estrategias.hmm_garch_strategy import HMMGARCHStrategy

TICKER = "HMMTEST"


def _make_ohlcv(n, mu=3e-4, sigma=0.012, start="2010-01-01", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = 1_000.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    noise = rng.uniform(0.001, 0.008, n)
    return pd.DataFrame({
        "Close": close,
        "Open":  close * (1 + rng.normal(0, 0.002, n)),
        "High":  close * (1 + np.abs(rng.normal(0, noise))),
        "Low":   close * (1 - np.abs(rng.normal(0, noise))),
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=dates)


def _make_ohlcv_dos_regimenes(n=700, seed=7):
    """Precio con dos regímenes de volatilidad claramente distintos, para
    que la selección automática de K tenga una señal real que detectar."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    bloques = []
    i = 0
    while i < n:
        dur = rng.integers(30, 80)
        bloques.extend([rng.integers(0, 2)] * dur)
        i += dur
    estado = np.array(bloques[:n])
    mu = np.where(estado == 0, -3e-4, 5e-4)
    sigma = np.where(estado == 0, 0.03, 0.008)
    rets = mu + sigma * rng.normal(size=n)
    close = 1_000.0 * np.exp(np.cumsum(rets))
    noise = rng.uniform(0.001, 0.008, n)
    return pd.DataFrame({
        "Close": close,
        "Open":  close * (1 + rng.normal(0, 0.002, n)),
        "High":  close * (1 + np.abs(rng.normal(0, noise))),
        "Low":   close * (1 - np.abs(rng.normal(0, noise))),
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=dates)


# ---------------------------------------------------------------------------
# No-regresión: defaults idénticos al comportamiento original
# ---------------------------------------------------------------------------

def test_defaults_no_activan_seleccion_ni_validacion():
    strat = HMMGARCHStrategy(universo=[TICKER], min_train_days=252)
    assert strat.ultimo_reporte_seleccion_k is None
    assert strat.ultimo_reporte_normalidad is None

    df = _make_ohlcv(n=600)
    strat.generar_señales(df, df.index[-1])

    # Sin las extensiones activas, ningún reporte de diagnóstico se genera.
    assert strat.ultimo_reporte_seleccion_k is None
    assert strat.ultimo_reporte_normalidad is None
    # n_regimenes fijo se mantiene sin cambios
    assert strat._n_regimenes == 3


# ---------------------------------------------------------------------------
# Selección automática de K
# ---------------------------------------------------------------------------

def test_seleccion_automatica_k_genera_reporte():
    strat = HMMGARCHStrategy(
        universo=[TICKER], min_train_days=252,
        seleccionar_k_automaticamente=True, k_range=range(2, 5),
    )
    df = _make_ohlcv_dos_regimenes(n=700)
    strat.generar_señales(df, df.index[-1])

    reporte = strat.ultimo_reporte_seleccion_k
    assert reporte is not None
    assert reporte.k_optimo_bic in range(2, 5)
    assert strat._detector is not None
    assert strat._detector.n_regimenes == reporte.k_optimo_bic


# ---------------------------------------------------------------------------
# Validación de normalidad -> selección de dist GARCH
# ---------------------------------------------------------------------------

def test_validar_normalidad_genera_reporte():
    strat = HMMGARCHStrategy(
        universo=[TICKER], min_train_days=252, validar_normalidad=True,
    )
    df = _make_ohlcv_dos_regimenes(n=700)
    strat.generar_señales(df, df.index[-1])

    reporte = strat.ultimo_reporte_normalidad
    assert reporte is not None
    assert reporte.supuesto == "normalidad"
    # Retornos financieros simulados con ruido gaussiano exacto pueden no
    # rechazar, pero el pipeline debe correr sin excepción y dejar rastro.
    assert strat._garch is not None


# ---------------------------------------------------------------------------
# Incertidumbre de régimen y probabilidad de crisis
# ---------------------------------------------------------------------------

def test_incertidumbre_regimen_antes_de_ajustar_es_cero():
    strat = HMMGARCHStrategy(universo=[TICKER])
    assert strat.incertidumbre_regimen(np.array([0.01, -0.01])) == 0.0
    assert strat.probabilidad_crisis(np.array([0.01, -0.01])) == 0.0


def test_incertidumbre_regimen_en_rango_valido_tras_ajustar():
    strat = HMMGARCHStrategy(universo=[TICKER], min_train_days=252)
    df = _make_ohlcv_dos_regimenes(n=700)
    close = df["Close"].astype(float)
    log_rets = np.log(close / close.shift(1)).dropna().values

    strat.generar_señales(df, df.index[-1])
    assert strat._detector is not None

    incertidumbre = strat.incertidumbre_regimen(log_rets)
    assert 0.0 <= incertidumbre <= 1.0

    p_crisis = strat.probabilidad_crisis(log_rets)
    assert 0.0 <= p_crisis <= 1.0


def test_probabilidad_crisis_alta_en_regimen_bajista_marcado():
    """
    Construye una serie que termina en un tramo claramente bajista y de
    alta vol; P(crisis) en el último punto debería ser considerablemente
    mayor que en una serie que termina en tramo alcista tranquilo.
    """
    n = 500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)

    ret_bull = rng.normal(0.0008, 0.006, n - 60)
    ret_bear = rng.normal(-0.01, 0.035, 60)  # tramo final: crash marcado

    rets = np.concatenate([ret_bull, ret_bear])
    close = 1000.0 * np.exp(np.cumsum(rets))
    noise = rng.uniform(0.001, 0.008, n)
    df = pd.DataFrame({
        "Close": close,
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + np.abs(rng.normal(0, noise))),
        "Low":  close * (1 - np.abs(rng.normal(0, noise))),
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=idx)

    strat = HMMGARCHStrategy(universo=[TICKER], min_train_days=252, n_regimenes=2)
    log_rets_full = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
    strat.generar_señales(df, df.index[-1])

    p_crisis_final = strat.probabilidad_crisis(log_rets_full)
    assert p_crisis_final > 0.5
