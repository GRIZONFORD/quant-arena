# =============================================================================
# FILE: quant_arena/tests/test_assumption_validator.py
# Suite pytest para AssumptionValidator — cada test usa una serie sintética
# de distribución CONOCIDA (semilla fija) para que el resultado esperado sea
# determinista, no una corazonada estadística.
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_arena.diagnostics.assumption_validator import AssumptionValidator
from quant_arena.core.excepciones import ConfiguracionInvalidaError, SupuestoEstadisticoError

RNG = np.random.default_rng(0)


@pytest.fixture
def av() -> AssumptionValidator:
    return AssumptionValidator(alfa=0.05)


def test_alfa_fuera_de_rango_lanza_error():
    with pytest.raises(ConfiguracionInvalidaError):
        AssumptionValidator(alfa=0.0)
    with pytest.raises(ConfiguracionInvalidaError):
        AssumptionValidator(alfa=1.0)


# ---------------------------------------------------------------------------
# Normalidad
# ---------------------------------------------------------------------------

def test_normalidad_no_rechaza_datos_gaussianos(av):
    normal = RNG.normal(0, 1, 2000)
    r = av.test_normalidad(normal)
    assert r.rechaza_h0 is False
    assert r.p_valor >= 0.05


def test_normalidad_rechaza_cola_pesada(av):
    t3 = RNG.standard_t(df=3, size=2000)
    r = av.test_normalidad(t3)
    assert r.rechaza_h0 is True
    assert "t de Student" in r.accion_recomendada


def test_normalidad_n_insuficiente_lanza_error(av):
    with pytest.raises(SupuestoEstadisticoError):
        av.test_normalidad([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Estacionariedad
# ---------------------------------------------------------------------------

def test_estacionariedad_ruido_blanco_es_estacionaria(av):
    wn = RNG.normal(0, 1, 500)
    r = av.test_estacionariedad(wn)
    assert r.rechaza_h0 is False
    assert "estacionaria" in r.decision and "no estacionaria" not in r.decision


def test_estacionariedad_random_walk_no_es_estacionaria(av):
    rw = np.cumsum(RNG.normal(0, 1, 500))
    r = av.test_estacionariedad(rw)
    assert r.rechaza_h0 is True
    assert "no estacionaria" in r.decision


# ---------------------------------------------------------------------------
# Independencia de residuos
# ---------------------------------------------------------------------------

def test_independencia_ruido_blanco_no_rechaza(av):
    wn = RNG.normal(0, 1, 500)
    r = av.test_independencia_residuos(wn)
    assert r.rechaza_h0 is False


def test_independencia_ar1_fuerte_rechaza(av):
    ar1 = np.zeros(500)
    for i in range(1, 500):
        ar1[i] = 0.9 * ar1[i - 1] + RNG.normal(0, 1)
    r = av.test_independencia_residuos(ar1)
    assert r.rechaza_h0 is True
    assert "AR" in r.accion_recomendada or "régimen" in r.accion_recomendada


# ---------------------------------------------------------------------------
# Homocedasticidad (ARCH-LM)
# ---------------------------------------------------------------------------

def test_homocedasticidad_ruido_blanco_no_rechaza(av):
    wn = RNG.normal(0, 1, 500)
    r = av.test_homocedasticidad(wn)
    assert r.rechaza_h0 is False


def test_homocedasticidad_arch_rechaza(av):
    n = 1000
    x = np.zeros(n)
    for t in range(1, n):
        sigma = np.sqrt(0.05 + 0.85 * x[t - 1] ** 2)
        x[t] = sigma * RNG.normal()
    r = av.test_homocedasticidad(x)
    assert r.rechaza_h0 is True
    assert "GARCH" in r.accion_recomendada


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def test_vif_features_no_colineales_bajo(av):
    x1 = RNG.normal(0, 1, 300)
    x2 = RNG.normal(0, 1, 300)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    resultado = av.calcular_vif(df)
    assert (resultado["vif"] < 10.0).all()
    assert not resultado["excede_umbral"].any()


def test_vif_features_colineales_alto(av):
    x1 = RNG.normal(0, 1, 300)
    x2 = RNG.normal(0, 1, 300)
    x3 = x1 + RNG.normal(0, 0.01, 300)  # casi idéntico a x1
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    resultado = av.calcular_vif(df)
    assert resultado.loc["x1", "excede_umbral"]
    assert resultado.loc["x3", "excede_umbral"]
    assert not resultado.loc["x2", "excede_umbral"]


def test_vif_una_sola_columna_lanza_error(av):
    df = pd.DataFrame({"x1": RNG.normal(0, 1, 100)})
    with pytest.raises(SupuestoEstadisticoError):
        av.calcular_vif(df)


# ---------------------------------------------------------------------------
# Orden de Markov
# ---------------------------------------------------------------------------

def test_orden_markov_cadena_verdadera_orden1_no_rechaza(av):
    K = 2
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    seq = [0]
    for _ in range(3000):
        seq.append(int(RNG.choice(K, p=P[seq[-1]])))
    r = av.test_orden_markov(seq, n_estados=K)
    assert r.rechaza_h0 is False


def test_orden_markov_dependencia_orden2_rechaza(av):
    """
    Construye una secuencia con dependencia de orden 2 explícita: el
    siguiente estado depende de los DOS anteriores (paridad de su suma),
    algo que un modelo Markov de orden 1 no puede capturar.
    """
    K = 2
    seq = [0, 1]
    for _ in range(3000):
        anteriores = seq[-2] + seq[-1]
        determinista = anteriores % 2
        # 90% sigue la regla determinista de orden 2, 10% ruido
        if RNG.random() < 0.9:
            seq.append(determinista)
        else:
            seq.append(1 - determinista)
    r = av.test_orden_markov(seq, n_estados=K)
    assert r.rechaza_h0 is True


def test_orden_markov_n_insuficiente_lanza_error(av):
    with pytest.raises(SupuestoEstadisticoError):
        av.test_orden_markov([0, 1, 0, 1], n_estados=2)


# ---------------------------------------------------------------------------
# validar_serie_completa
# ---------------------------------------------------------------------------

def test_validar_serie_completa_retorna_los_cuatro_supuestos(av):
    wn = RNG.normal(0, 1, 500)
    resultado = av.validar_serie_completa(wn)
    assert set(resultado.keys()) == {
        "normalidad", "estacionariedad", "independencia_residuos", "homocedasticidad"
    }
