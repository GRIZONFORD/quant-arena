# =============================================================================
# FILE: quant_arena/tests/test_position_risk.py
# Suite pytest para TakeProfitATRRule, calcular_atr_pct y
# retornos_periodo_con_riesgo.
#
# Cobertura:
#   - TakeProfitATRRule recorta exposición tras cruzar el umbral y lo
#     mantiene (first-touch persistente / cummax).
#   - Sin disparo, exposición se mantiene en 1.0.
#   - atr_pct inválido (NaN, <=0) deja la exposición sin cambios.
#   - ConfiguracionInvalidaError ante parámetros fuera de dominio.
#   - calcular_atr_pct causal: usa solo datos <= fecha; NaN con historial
#     insuficiente.
#   - retornos_periodo_con_riesgo con tp_rule=None reproduce el cálculo
#     vectorizado estándar (dot product de pesos × retornos).
#   - retornos_periodo_con_riesgo con TP activo difiere del vectorizado
#     tras el disparo (menor exposición → menor retorno acumulado tras un
#     rally sostenido).
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_arena.backtesting.position_risk import (
    TakeProfitATRRule,
    calcular_atr_pct,
    retornos_periodo_con_riesgo,
)
from quant_arena.core.excepciones import ConfiguracionInvalidaError


# ---------------------------------------------------------------------------
# TakeProfitATRRule
# ---------------------------------------------------------------------------

def _precio_rally(n=10, entrada=100.0, paso=2.0):
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(entrada + np.arange(n) * paso, index=idx)


def test_take_profit_recorta_tras_disparo_y_persiste():
    regla = TakeProfitATRRule(tp_mult=2.0, recorte=0.5)
    precio = _precio_rally()  # 100, 102, 104, ... sube monotónicamente
    atr_pct = 0.02  # dispara con ganancia >= 4%
    exposicion = pd.Series(1.0, index=precio.index)
    resultado = regla.aplicar(exposicion, {"precio": precio, "atr_pct": atr_pct})

    ganancia = precio / precio.iloc[0] - 1.0
    primer_disparo = (ganancia >= 2.0 * atr_pct).idxmax()

    assert (resultado.loc[:primer_disparo].iloc[:-1] == 1.0).all()
    assert (resultado.loc[primer_disparo:] == 0.5).all()  # 1 - recorte, persistente


def test_take_profit_sin_disparo_mantiene_exposicion():
    regla = TakeProfitATRRule(tp_mult=10.0, recorte=0.5)  # umbral inalcanzable
    precio = _precio_rally()
    exposicion = pd.Series(1.0, index=precio.index)
    resultado = regla.aplicar(exposicion, {"precio": precio, "atr_pct": 0.02})
    assert (resultado == 1.0).all()


@pytest.mark.parametrize("atr_pct", [float("nan"), 0.0, -0.01])
def test_take_profit_atr_pct_invalido_no_cambia_exposicion(atr_pct):
    regla = TakeProfitATRRule()
    precio = _precio_rally()
    exposicion = pd.Series(1.0, index=precio.index)
    resultado = regla.aplicar(exposicion, {"precio": precio, "atr_pct": atr_pct})
    assert (resultado == 1.0).all()


def test_take_profit_configuracion_invalida():
    with pytest.raises(ConfiguracionInvalidaError):
        TakeProfitATRRule(tp_mult=0.0)
    with pytest.raises(ConfiguracionInvalidaError):
        TakeProfitATRRule(recorte=0.0)
    with pytest.raises(ConfiguracionInvalidaError):
        TakeProfitATRRule(recorte=1.5)


# ---------------------------------------------------------------------------
# calcular_atr_pct
# ---------------------------------------------------------------------------

def _ohlc_sintetico(n=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_calcular_atr_pct_causal_y_positivo():
    ohlc = _ohlc_sintetico()
    fecha = ohlc.index[40]
    atr_pct = calcular_atr_pct(ohlc, fecha, atr_window=14)
    assert np.isfinite(atr_pct)
    assert atr_pct > 0.0


def test_calcular_atr_pct_no_usa_datos_futuros():
    ohlc = _ohlc_sintetico()
    fecha = ohlc.index[40]
    atr_con_futuro = calcular_atr_pct(ohlc, fecha, atr_window=14)
    atr_sin_futuro = calcular_atr_pct(ohlc.loc[:fecha], fecha, atr_window=14)
    assert atr_con_futuro == pytest.approx(atr_sin_futuro)


def test_calcular_atr_pct_historial_insuficiente_da_nan():
    ohlc = _ohlc_sintetico(n=60)
    fecha = ohlc.index[5]  # menos de atr_window+1 observaciones
    assert np.isnan(calcular_atr_pct(ohlc, fecha, atr_window=14))


# ---------------------------------------------------------------------------
# retornos_periodo_con_riesgo
# ---------------------------------------------------------------------------

def _datos_precio(n=15, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    precio = 100.0 * (1.0 + rng.normal(0.002, 0.005, n)).cumprod()
    return pd.DataFrame({"ACTIVO": precio}, index=idx)


def test_sin_tp_rule_reproduce_calculo_vectorizado_estandar():
    datos = _datos_precio()
    pesos = pd.Series({"ACTIVO": 0.6})
    fecha_desde, fecha_hasta = datos.index[0], datos.index[-1]

    resultado = retornos_periodo_con_riesgo(pesos, datos, fecha_desde, fecha_hasta)

    esperado = (datos["ACTIVO"].pct_change().iloc[1:] * 0.6)
    pd.testing.assert_series_equal(resultado, esperado, check_names=False)


def test_activos_sin_peso_significativo_retorna_cero():
    datos = _datos_precio()
    pesos = pd.Series({"ACTIVO": 1e-12})
    fecha_desde, fecha_hasta = datos.index[0], datos.index[-1]
    resultado = retornos_periodo_con_riesgo(pesos, datos, fecha_desde, fecha_hasta)
    assert (resultado == 0.0).all()


def test_tp_activo_reduce_retorno_acumulado_tras_rally():
    idx = pd.date_range("2021-01-01", periods=12, freq="B")
    precio = pd.Series(100.0 + np.arange(12) * 3.0, index=idx)  # rally fuerte y sostenido
    datos = pd.DataFrame({"ACTIVO": precio})
    pesos = pd.Series({"ACTIVO": 1.0})
    fecha_desde, fecha_hasta = idx[0], idx[-1]

    sin_tp = retornos_periodo_con_riesgo(pesos, datos, fecha_desde, fecha_hasta)
    con_tp = retornos_periodo_con_riesgo(
        pesos, datos, fecha_desde, fecha_hasta,
        atr_pct={"ACTIVO": 0.02},
        tp_rule=TakeProfitATRRule(tp_mult=1.0, recorte=0.5),
    )

    equity_sin_tp = (1.0 + sin_tp).prod()
    equity_con_tp = (1.0 + con_tp).prod()
    assert equity_con_tp < equity_sin_tp  # TP recorta ganancia del rally


def test_tp_rule_none_es_no_regresion_exacta_con_atr_pct_presente():
    """Si tp_rule es None, atr_pct se ignora por completo (no-regresión)."""
    datos = _datos_precio()
    pesos = pd.Series({"ACTIVO": 0.8})
    fecha_desde, fecha_hasta = datos.index[0], datos.index[-1]

    con_atr_sin_regla = retornos_periodo_con_riesgo(
        pesos, datos, fecha_desde, fecha_hasta,
        atr_pct={"ACTIVO": 0.001}, tp_rule=None,
    )
    estandar = retornos_periodo_con_riesgo(pesos, datos, fecha_desde, fecha_hasta)
    pd.testing.assert_series_equal(con_atr_sin_regla, estandar)


def test_periodo_vacio_retorna_serie_vacia():
    datos = _datos_precio()
    pesos = pd.Series({"ACTIVO": 0.5})
    fecha = datos.index[0]
    resultado = retornos_periodo_con_riesgo(pesos, datos, fecha, fecha)
    assert resultado.empty
