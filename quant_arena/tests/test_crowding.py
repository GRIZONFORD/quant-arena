# =============================================================================
# FILE: quant_arena/tests/test_crowding.py
# Suite pytest para CrowdingModel y ArenaCrowding (§1.4).
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_arena.backtesting.crowding import ArenaCrowding, CrowdingModel
from quant_arena.core.excepciones import ConfiguracionInvalidaError


# ---------------------------------------------------------------------------
# CrowdingModel
# ---------------------------------------------------------------------------

def test_kappa_negativo_lanza_error():
    with pytest.raises(ConfiguracionInvalidaError):
        CrowdingModel(kappa=-0.1)


def test_kappa_cero_no_penaliza():
    modelo = CrowdingModel(kappa=0.0)
    assert modelo.factor_decaimiento(participacion=5.0) == 1.0
    assert modelo.alpha_efectivo(alpha_bruto=0.1, capital_dirigido=1e9, adv=1e6) == 0.1


def test_factor_decaimiento_decrece_con_participacion():
    modelo = CrowdingModel(kappa=1.0)
    bajo = modelo.factor_decaimiento(0.01)
    alto = modelo.factor_decaimiento(2.0)
    assert 0.0 < alto < bajo <= 1.0


def test_factor_decaimiento_participacion_invalida_no_penaliza():
    modelo = CrowdingModel(kappa=1.0)
    assert modelo.factor_decaimiento(float("nan")) == 1.0
    assert modelo.factor_decaimiento(-1.0) == 1.0


def test_alpha_efectivo_sin_adv_valido_no_penaliza():
    modelo = CrowdingModel(kappa=2.0)
    assert modelo.alpha_efectivo(0.05, capital_dirigido=1e6, adv=0.0) == 0.05
    assert modelo.alpha_efectivo(0.05, capital_dirigido=1e6, adv=float("nan")) == 0.05


def test_alpha_efectivo_participacion_alta_erosiona_fuertemente():
    modelo = CrowdingModel(kappa=1.0)
    alpha_bruto = 0.10
    # capital_dirigido == adv -> participacion = 1.0 -> factor = exp(-1)
    resultado = modelo.alpha_efectivo(alpha_bruto, capital_dirigido=1_000_000, adv=1_000_000)
    assert resultado == pytest.approx(alpha_bruto * np.exp(-1.0))


# ---------------------------------------------------------------------------
# ArenaCrowding.solapamiento
# ---------------------------------------------------------------------------

def test_solapamiento_diagonal_es_uno():
    pesos = {
        "A": pd.Series({"X": 0.6, "Y": 0.4}),
        "B": pd.Series({"X": 0.3, "Z": 0.7}),
    }
    arena = ArenaCrowding(CrowdingModel(), adv_por_ticker={})
    matriz = arena.solapamiento(pesos)
    assert matriz.loc["A", "A"] == pytest.approx(1.0)
    assert matriz.loc["B", "B"] == pytest.approx(1.0)


def test_solapamiento_posiciones_identicas_da_uno():
    pesos = {
        "A": pd.Series({"X": 0.6, "Y": 0.4}),
        "B": pd.Series({"X": 0.6, "Y": 0.4}),
    }
    arena = ArenaCrowding(CrowdingModel(), adv_por_ticker={})
    matriz = arena.solapamiento(pesos)
    assert matriz.loc["A", "B"] == pytest.approx(1.0)
    assert matriz.loc["B", "A"] == pytest.approx(1.0)  # simetría


def test_solapamiento_universos_disjuntos_da_cero():
    pesos = {
        "A": pd.Series({"X": 1.0}),
        "B": pd.Series({"Y": 1.0}),
    }
    arena = ArenaCrowding(CrowdingModel(), adv_por_ticker={})
    matriz = arena.solapamiento(pesos)
    assert matriz.loc["A", "B"] == pytest.approx(0.0)


def test_solapamiento_parcial_entre_cero_y_uno():
    pesos = {
        "A": pd.Series({"X": 0.5, "Y": 0.5}),
        "B": pd.Series({"X": 0.5, "Z": 0.5}),
    }
    arena = ArenaCrowding(CrowdingModel(), adv_por_ticker={})
    matriz = arena.solapamiento(pesos)
    assert 0.0 < matriz.loc["A", "B"] < 1.0


# ---------------------------------------------------------------------------
# ArenaCrowding.demanda_agregada_por_ticker
# ---------------------------------------------------------------------------

def test_demanda_agregada_suma_contribuciones_de_todas_las_estrategias():
    pesos = {
        "A": pd.Series({"TICKER": 1.0}),
        "B": pd.Series({"TICKER": 1.0}),
    }
    capital = {"A": 0.3, "B": 0.2}
    arena = ArenaCrowding(CrowdingModel(), adv_por_ticker={})
    demanda = arena.demanda_agregada_por_ticker(pesos, capital, aum_total=1_000_000)
    # A: 1.0 * 0.3 * 1e6 = 300_000; B: 1.0 * 0.2 * 1e6 = 200_000
    assert demanda["TICKER"] == pytest.approx(500_000)


def test_demanda_agregada_ignora_estrategias_sin_capital():
    pesos = {"A": pd.Series({"TICKER": 1.0}), "B": pd.Series({"TICKER": 1.0})}
    capital = {"A": 0.0, "B": 0.5}
    arena = ArenaCrowding(CrowdingModel(), adv_por_ticker={})
    demanda = arena.demanda_agregada_por_ticker(pesos, capital, aum_total=1_000_000)
    assert demanda["TICKER"] == pytest.approx(500_000)


# ---------------------------------------------------------------------------
# ArenaCrowding.factores_decaimiento — el mecanismo de acoplamiento completo
# ---------------------------------------------------------------------------

def test_dos_estrategias_convergiendo_en_el_mismo_ticker_se_penalizan_mutuamente():
    """
    Caso central de §1.4: dos estrategias, cada una con baja participación
    INDIVIDUAL, pero que convergen en el mismo ticker generan demanda
    agregada suficiente para erosionar el alpha de AMBAS — eso es crowding
    real, no un límite de capacidad por estrategia aislada.
    """
    pesos = {
        "A": pd.Series({"TICKER": 1.0}),
        "B": pd.Series({"TICKER": 1.0}),
    }
    capital = {"A": 0.5, "B": 0.5}
    adv = {"TICKER": 1_000_000}
    arena = ArenaCrowding(CrowdingModel(kappa=2.0), adv_por_ticker=adv)

    factores = arena.factores_decaimiento(pesos, capital, aum_total=1_000_000)

    # demanda agregada = (0.5*1e6) + (0.5*1e6) = 1e6 = ADV -> participacion=1.0
    esperado = np.exp(-2.0 * 1.0)
    assert factores["A"] == pytest.approx(esperado)
    assert factores["B"] == pytest.approx(esperado)


def test_estrategia_sola_en_su_ticker_penaliza_menos_que_si_hay_convergencia():
    capital = {"A": 0.5, "B": 0.5}
    adv = {"TICKER_A": 1_000_000, "TICKER_B": 1_000_000}
    arena = ArenaCrowding(CrowdingModel(kappa=2.0), adv_por_ticker=adv)

    pesos_separados = {
        "A": pd.Series({"TICKER_A": 1.0}),
        "B": pd.Series({"TICKER_B": 1.0}),
    }
    pesos_convergentes = {
        "A": pd.Series({"TICKER_A": 1.0}),
        "B": pd.Series({"TICKER_A": 1.0}),
    }

    factores_separados = arena.factores_decaimiento(pesos_separados, capital, aum_total=1_000_000)
    factores_convergentes = arena.factores_decaimiento(pesos_convergentes, capital, aum_total=1_000_000)

    assert factores_convergentes["A"] < factores_separados["A"]


def test_estrategia_sin_peso_no_se_penaliza():
    pesos = {"A": pd.Series({"TICKER": 1.0}), "B": pd.Series(dtype=float)}
    capital = {"A": 0.5, "B": 0.0}
    arena = ArenaCrowding(CrowdingModel(kappa=1.0), adv_por_ticker={"TICKER": 1_000_000})
    factores = arena.factores_decaimiento(pesos, capital, aum_total=1_000_000)
    assert factores["B"] == 1.0


def test_sin_adv_conocido_no_penaliza():
    pesos = {"A": pd.Series({"DESCONOCIDO": 1.0})}
    capital = {"A": 1.0}
    arena = ArenaCrowding(CrowdingModel(kappa=5.0), adv_por_ticker={})
    factores = arena.factores_decaimiento(pesos, capital, aum_total=1_000_000)
    assert factores["A"] == 1.0


def test_factores_siempre_en_rango_valido():
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(5)]
    pesos = {
        f"strat_{k}": pd.Series(
            rng.random(5), index=rng.choice(tickers, size=5, replace=False)
        )
        for k in range(4)
    }
    capital = {f"strat_{k}": 0.25 for k in range(4)}
    adv = {t: 500_000.0 for t in tickers}
    arena = ArenaCrowding(CrowdingModel(kappa=3.0), adv_por_ticker=adv)
    factores = arena.factores_decaimiento(pesos, capital, aum_total=2_000_000)
    for factor in factores.values():
        assert 0.0 <= factor <= 1.0
