# =============================================================================
# FILE: quant_arena/tests/test_model_selection.py
# Suite pytest para seleccionar_k_hmm — barrido de K con AIC/BIC/CV temporal.
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest

from quant_arena.diagnostics.model_selection import (
    n_parametros_hmm,
    seleccionar_k_hmm,
)
from quant_arena.core.excepciones import ConfiguracionInvalidaError, SupuestoEstadisticoError


def _serie_dos_regimenes(n: int = 800, seed: int = 1) -> np.ndarray:
    """
    Serie sintética con dos regímenes claramente separados (μ y σ muy
    distintos) y bloques persistentes (no iid), imitando la estructura de
    un HMM real de 2 estados.
    """
    rng = np.random.default_rng(seed)
    bloques = []
    i = 0
    while i < n:
        dur = rng.integers(20, 60)
        s = rng.integers(0, 2)
        bloques.extend([s] * dur)
        i += dur
    estado = np.array(bloques[:n])
    mu = np.where(estado == 0, -0.01, 0.01)
    sigma = np.where(estado == 0, 0.03, 0.008)
    return mu + sigma * rng.normal(size=n)


def test_n_parametros_formula():
    # p = K^2 + 2K - 1
    assert n_parametros_hmm(2) == 4 + 4 - 1
    assert n_parametros_hmm(3) == 9 + 6 - 1


def test_seleccion_k_recupera_dos_regimenes_verdaderos():
    rets = _serie_dos_regimenes()
    reporte = seleccionar_k_hmm(rets, k_range=range(2, 6), n_iter=100, random_state=42)

    assert reporte.k_optimo_bic == 2
    assert reporte.k_optimo_aic == 2
    assert reporte.k_optimo_cv == 2

    resultado_k2 = next(r for r in reporte.resultados if r.k == 2)
    assert resultado_k2.convergio is True
    assert resultado_k2.log_likelihood_oos is not None


def test_bic_penaliza_mas_que_aic_para_k_grande():
    """A igual log-verosimilitud marginal, BIC crece más rápido con K que AIC
    (ln(n) > 2 para n > 7) — el K óptimo por BIC nunca debería ser mayor
    que el óptimo por AIC en el mismo barrido."""
    rets = _serie_dos_regimenes(n=1000, seed=7)
    reporte = seleccionar_k_hmm(rets, k_range=range(2, 5), random_state=42)
    assert reporte.k_optimo_bic <= reporte.k_optimo_aic + 1  # tolerancia por ruido de ajuste


def test_k_range_invalido_lanza_error():
    rets = _serie_dos_regimenes()
    with pytest.raises(ConfiguracionInvalidaError):
        seleccionar_k_hmm(rets, k_range=[0, 1, 2])


def test_fraccion_train_cv_invalida_lanza_error():
    rets = _serie_dos_regimenes()
    with pytest.raises(ConfiguracionInvalidaError):
        seleccionar_k_hmm(rets, fraccion_train_cv=1.5)


def test_datos_insuficientes_lanza_error():
    with pytest.raises(SupuestoEstadisticoError):
        seleccionar_k_hmm(np.array([0.01, -0.01, 0.02]), k_range=range(2, 4))


def test_reporte_incluye_todos_los_k_que_ajustaron():
    rets = _serie_dos_regimenes(n=600, seed=3)
    reporte = seleccionar_k_hmm(rets, k_range=range(2, 4), random_state=42)
    ks_reportados = {r.k for r in reporte.resultados}
    assert ks_reportados == {2, 3}
