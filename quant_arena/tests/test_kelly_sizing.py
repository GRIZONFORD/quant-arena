# =============================================================================
# FILE: quant_arena/tests/test_kelly_sizing.py
# Suite pytest para KellyBayesianSizer, VolTargetSizer y FixedFractionSizer.
#
# Cobertura:
#   - exposicion() reproduce Kelly clásico cuando kappa_skill=0
#   - sigma_skill alta reduce la exposición (penalización bayesiana)
#   - mu_edge <= 0 produce exposición en el piso (0.0)
#   - cap_individual acota la exposición de una sola estrategia
#   - SizingError ante mu_edge no finito o varianza total nula
#   - ConfiguracionInvalidaError ante parámetros fuera de dominio
#   - pesos_exposicion(): estrategias sin datos -> 0.0, no abortan el resto
#   - pesos_exposicion(): renormaliza cuando la suma cruda excede cap_bruto
#   - convicción nula en todas las estrategias -> exposición total baja (no 1/N)
#   - VolTargetSizer y FixedFractionSizer cumplen el contrato AbstractPositionSizer
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest

from quant_arena.backtesting.kelly_sizing import (
    FixedFractionSizer,
    KellyBayesianSizer,
    VolTargetSizer,
)
from quant_arena.core.excepciones import ConfiguracionInvalidaError, SizingError


# ---------------------------------------------------------------------------
# KellyBayesianSizer.exposicion
# ---------------------------------------------------------------------------

def test_kelly_clasico_sin_penalizacion_bayesiana():
    sizer = KellyBayesianSizer(lam=1.0, kappa_skill=0.0, cap_individual=10.0, cap_bruto=10.0)
    mu, sigma = 0.01, 0.05
    esperado = mu / sigma ** 2
    assert sizer.exposicion(mu_edge=mu, sigma_retornos=sigma) == pytest.approx(esperado)


def test_sigma_skill_alta_reduce_exposicion():
    sizer = KellyBayesianSizer(lam=0.5, kappa_skill=1.0, cap_individual=10.0)
    baja_incertidumbre = sizer.exposicion(mu_edge=0.01, sigma_retornos=0.05, sigma_skill=0.1)
    alta_incertidumbre = sizer.exposicion(mu_edge=0.01, sigma_retornos=0.05, sigma_skill=5.0)
    assert alta_incertidumbre < baja_incertidumbre
    assert alta_incertidumbre >= 0.0


def test_mu_edge_no_positivo_da_piso():
    sizer = KellyBayesianSizer()
    assert sizer.exposicion(mu_edge=0.0, sigma_retornos=0.05) == 0.0
    assert sizer.exposicion(mu_edge=-0.01, sigma_retornos=0.05) == 0.0


def test_cap_individual_acota_exposicion():
    sizer = KellyBayesianSizer(lam=1.0, kappa_skill=0.0, cap_individual=0.3)
    # mu/sigma^2 muy grande sin cap se dispararía por encima de 0.3
    assert sizer.exposicion(mu_edge=1.0, sigma_retornos=0.01) == pytest.approx(0.3)


def test_mu_edge_no_finito_lanza_sizing_error():
    sizer = KellyBayesianSizer()
    with pytest.raises(SizingError):
        sizer.exposicion(mu_edge=float("nan"), sigma_retornos=0.05)
    with pytest.raises(SizingError):
        sizer.exposicion(mu_edge=float("inf"), sigma_retornos=0.05)


def test_varianza_total_nula_lanza_sizing_error():
    sizer = KellyBayesianSizer(kappa_skill=0.0)
    with pytest.raises(SizingError):
        sizer.exposicion(mu_edge=0.01, sigma_retornos=0.0, sigma_skill=0.0)


def test_sigma_negativa_lanza_sizing_error():
    sizer = KellyBayesianSizer()
    with pytest.raises(SizingError):
        sizer.exposicion(mu_edge=0.01, sigma_retornos=-0.01)
    with pytest.raises(SizingError):
        sizer.exposicion(mu_edge=0.01, sigma_retornos=0.05, sigma_skill=-1.0)


# ---------------------------------------------------------------------------
# Validación de configuración (fail-fast en __post_init__)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"lam": 0.0},
    {"lam": 1.5},
    {"lam": -0.1},
    {"kappa_skill": -1.0},
    {"cap_individual": 0.0},
    {"cap_bruto": -1.0},
    {"piso": -0.1},
])
def test_configuracion_invalida_lanza_error(kwargs):
    with pytest.raises(ConfiguracionInvalidaError):
        KellyBayesianSizer(**kwargs)


# ---------------------------------------------------------------------------
# KellyBayesianSizer.pesos_exposicion (nivel Zoo)
# ---------------------------------------------------------------------------

def test_pesos_exposicion_estrategia_sin_datos_no_aborta():
    sizer = KellyBayesianSizer(lam=0.5, kappa_skill=0.0, cap_bruto=10.0, cap_individual=10.0)
    resultado = sizer.pesos_exposicion(
        mu_edge={"a": 0.01, "b": 0.02},
        sigma_retornos={"a": 0.05, "b": 0.0},  # 'b' produce SizingError (varianza nula)
        sigma_skill={"a": 0.0, "b": 0.0},
    )
    assert resultado["b"] == 0.0
    assert resultado["a"] > 0.0


def test_pesos_exposicion_renormaliza_bajo_cap_bruto():
    sizer = KellyBayesianSizer(lam=1.0, kappa_skill=0.0, cap_individual=10.0, cap_bruto=1.0)
    resultado = sizer.pesos_exposicion(
        mu_edge={"a": 1.0, "b": 1.0},
        sigma_retornos={"a": 0.01, "b": 0.01},
        sigma_skill={"a": 0.0, "b": 0.0},
    )
    assert sum(resultado.values()) == pytest.approx(1.0)


def test_pesos_exposicion_convicción_nula_reduce_exposicion_total():
    """
    Cuando el Juez no tiene convicción en NINGUNA estrategia (sigma_skill muy
    alta en todas), la exposición total debe caer por debajo de cap_bruto en
    vez de repartir 1/N como hace el fallback ingenuo de TTTJuez (H4).
    """
    sizer = KellyBayesianSizer(lam=0.5, kappa_skill=1.0, cap_individual=1.0, cap_bruto=1.0)
    resultado = sizer.pesos_exposicion(
        mu_edge={"a": 0.001, "b": 0.001, "c": 0.001},
        sigma_retornos={"a": 0.02, "b": 0.02, "c": 0.02},
        sigma_skill={"a": 10.0, "b": 10.0, "c": 10.0},  # sin convicción alguna
    )
    assert sum(resultado.values()) < 1.0 / 3  # muy por debajo de equal-weight


# ---------------------------------------------------------------------------
# VolTargetSizer
# ---------------------------------------------------------------------------

def test_vol_target_sizer_escala_inversamente_a_la_vol():
    sizer = VolTargetSizer(target_vol=0.15, cap=2.0)
    baja_vol = sizer.exposicion(mu_edge=0.0, sigma_retornos=0.005)
    alta_vol = sizer.exposicion(mu_edge=0.0, sigma_retornos=0.05)
    assert baja_vol > alta_vol


def test_vol_target_sizer_respeta_cap():
    sizer = VolTargetSizer(target_vol=0.5, cap=1.0)
    assert sizer.exposicion(mu_edge=0.0, sigma_retornos=1e-6) == pytest.approx(1.0)


def test_vol_target_sizer_sigma_no_positiva_lanza_error():
    sizer = VolTargetSizer()
    with pytest.raises(SizingError):
        sizer.exposicion(mu_edge=0.0, sigma_retornos=0.0)


# ---------------------------------------------------------------------------
# FixedFractionSizer
# ---------------------------------------------------------------------------

def test_fixed_fraction_sizer_ignora_edge_y_varianza():
    sizer = FixedFractionSizer(fraccion=0.4)
    assert sizer.exposicion(mu_edge=-5.0, sigma_retornos=100.0, sigma_skill=100.0) == 0.4


def test_fixed_fraction_sizer_valida_rango():
    with pytest.raises(ConfiguracionInvalidaError):
        FixedFractionSizer(fraccion=1.5)
