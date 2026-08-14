# =============================================================================
# FILE: quant_arena/tests/test_motor_risk_integration.py
# Suite de integración: BacktestEngine + RiskOverlay + KellyBayesianSizer +
# TakeProfitATRRule.
#
# Cobertura:
#   - No-regresión: sin risk_overlay/position_sizer/tp_rule, pesos_juez
#     sigue sumando exactamente 1.0 (comportamiento TTTJuez.pesos_asignacion()
#     sin alterar) y el engine es determinista.
#   - Con risk_overlay activo, la exposición tras un crash queda por debajo
#     de la exposición sin riesgo (curva de equity con menor drawdown).
#   - Con position_sizer (Kelly) activo, pesos_juez ya NO suma 1.0 —
#     exposición absoluta justificada en vez de reparto normalizado.
#   - Con tp_rule + datos_ohlc activos, los retornos de estrategia difieren
#     de la versión sin TP tras un rally intra-período.
# =============================================================================
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from quant_arena.backtesting.kelly_sizing import KellyBayesianSizer
from quant_arena.backtesting.motor import BacktestEngine
from quant_arena.backtesting.position_risk import TakeProfitATRRule
from quant_arena.backtesting.risk_overlay import RiskOverlay
from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.juez.ttt_juez import TTTJuez
from quant_arena.metricas.performance_metrics import PerformanceMetrics
from quant_arena.zoo.base_estrategia import ZooManager

TICKER = "ACTIVO"


class _EstrategiaFija(AbstractStrategy):
    """Estrategia sintética: mantiene un peso fijo constante en `TICKER`."""

    def __init__(self, nombre: str, universo: List[str], peso: float) -> None:
        super().__init__(nombre=nombre, universo=universo)
        self._peso = peso

    @property
    def descripcion(self) -> str:
        return f"Peso fijo {self._peso} en {TICKER}"

    def generar_señales(self, datos: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.Series:
        return pd.Series({TICKER: self._peso})

    def calcular_retornos(self, datos: pd.DataFrame, pesos_historicos: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("No usado por BacktestEngine en este pipeline.")


def _datos_con_crash(n: int = 400, crash_idx: int = 250, crash_size: float = -0.25, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    ret = rng.normal(0.0004, 0.01, n)
    ret[crash_idx] = crash_size
    precio = 100.0 * (1.0 + pd.Series(ret, index=idx)).cumprod()
    datos = pd.DataFrame({TICKER: precio})
    benchmark = precio.pct_change().dropna()
    return datos, benchmark


def _datos_rally(n: int = 200, seed: int = 5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    ret = rng.normal(0.001, 0.006, n)
    precio = 100.0 * (1.0 + pd.Series(ret, index=idx)).cumprod()
    datos = pd.DataFrame({TICKER: precio})
    benchmark = precio.pct_change().dropna()
    return datos, benchmark


def _zoo_dos_estrategias(datos: pd.DataFrame) -> ZooManager:
    zoo = ZooManager()
    zoo.agregar(_EstrategiaFija("full", universo=[TICKER], peso=1.0))
    zoo.agregar(_EstrategiaFija("half", universo=[TICKER], peso=0.5))
    return zoo


def _ohlc_desde_close(precio: pd.Series, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(precio)
    high = precio.values * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    low = precio.values * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    open_ = precio.values
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": precio.values, "volume": volume},
        index=precio.index,
    )


# ---------------------------------------------------------------------------
# No-regresión: engine sin ninguna extensión opcional
# ---------------------------------------------------------------------------

def test_sin_extensiones_pesos_juez_suma_uno():
    datos, benchmark = _datos_con_crash(n=300)
    zoo = _zoo_dos_estrategias(datos)
    juez = TTTJuez()
    metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)

    engine = BacktestEngine(
        zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
        ventana_metricas=63, metrica_ranking="sharpe",
    )
    resultado = engine.ejecutar_walk_forward(
        fecha_inicio=datos.index[0], fecha_fin=datos.index[-1], frecuencia_rebalanceo=21,
    )

    assert not resultado.pesos_juez.empty
    sumas = resultado.pesos_juez.sum(axis=1)
    assert (np.abs(sumas - 1.0) < 1e-9).all()


def test_engine_es_determinista():
    datos, benchmark = _datos_con_crash(n=300)

    def _correr():
        zoo = _zoo_dos_estrategias(datos)
        juez = TTTJuez()
        metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
        engine = BacktestEngine(zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark)
        return engine.ejecutar_walk_forward(datos.index[0], datos.index[-1], frecuencia_rebalanceo=21)

    r1 = _correr()
    r2 = _correr()
    pd.testing.assert_series_equal(r1.retornos_meta, r2.retornos_meta)


# ---------------------------------------------------------------------------
# RiskOverlay integrado
# ---------------------------------------------------------------------------

def test_risk_overlay_reduce_drawdown_tras_crash():
    datos, benchmark = _datos_con_crash(n=380, crash_idx=250, crash_size=-0.30)

    def _correr(risk_overlay):
        zoo = _zoo_dos_estrategias(datos)
        juez = TTTJuez()
        metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
        engine = BacktestEngine(
            zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
            risk_overlay=risk_overlay,
        )
        return engine.ejecutar_walk_forward(datos.index[0], datos.index[-1], frecuencia_rebalanceo=21)

    sin_riesgo = _correr(None)
    con_riesgo = _correr(RiskOverlay(target_vol=0.10, max_leverage=1.0, dd_limit=0.10))

    def _max_dd(equity: pd.Series) -> float:
        return float((equity / equity.cummax() - 1.0).min())

    dd_sin = _max_dd(sin_riesgo.equity_curve())
    dd_con = _max_dd(con_riesgo.equity_curve())
    assert dd_con > dd_sin  # menos negativo = drawdown más leve


def test_risk_overlay_no_colapsa_exposicion_por_falta_de_lookback():
    """
    Regresión dirigida: `_escalar_por_riesgo` solía pasar solo el fold
    aislado (~21 días) a `RiskOverlay.compute_risk_weight`, que internamente
    calcula la vol rodante on-the-fly si no recibe `market_data[vol_col]`.
    Con `vol_window=21` y un fold de ~21 días, la vol quedaba NaN en casi
    todo el fold -> target-vol -> 0 -> exposición sistemáticamente nula,
    incluso en un mercado tranquilo sin ningún drawdown real.
    """
    datos, benchmark = _datos_rally(n=300, seed=42)  # sin crash, vol estable
    zoo = _zoo_dos_estrategias(datos)
    juez = TTTJuez()
    metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
    engine = BacktestEngine(
        zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
        risk_overlay=RiskOverlay(target_vol=0.15, max_leverage=1.0, dd_limit=0.20),
    )
    resultado = engine.ejecutar_walk_forward(datos.index[0], datos.index[-1], frecuencia_rebalanceo=21)

    # Tras el warm-up, el equity NO debe quedar aplanado en 1.0 (exposición
    # cero permanente sería la firma exacta del bug de lookback).
    equity = resultado.equity_curve()
    assert equity.iloc[-1] != pytest.approx(1.0, abs=1e-6)
    assert resultado.retornos_meta.abs().sum() > 0.0


# ---------------------------------------------------------------------------
# KellyBayesianSizer integrado
# ---------------------------------------------------------------------------

def test_position_sizer_kelly_no_suma_uno():
    datos, benchmark = _datos_con_crash(n=300)
    zoo = _zoo_dos_estrategias(datos)
    juez = TTTJuez()
    metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
    sizer = KellyBayesianSizer(lam=0.5, kappa_skill=1.0, cap_individual=1.0, cap_bruto=1.0)

    engine = BacktestEngine(
        zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
        position_sizer=sizer,
    )
    resultado = engine.ejecutar_walk_forward(
        fecha_inicio=datos.index[0], fecha_fin=datos.index[-1], frecuencia_rebalanceo=21,
    )

    assert not resultado.pesos_juez.empty
    sumas = resultado.pesos_juez.sum(axis=1)
    # Kelly bayesiano no garantiza suma=1: al menos algún rebalanceo debe
    # diferir claramente de 1.0 (exposición absoluta, no reparto normalizado).
    assert (np.abs(sumas - 1.0) > 1e-6).any()
    assert (sumas <= 1.0 + 1e-9).all()  # cap_bruto=1.0 respetado


def test_position_sizer_kelly_reduce_exposicion_en_retornos_meta():
    """
    Regresión dirigida: `_ensamblar_meta` solía RENORMALIZAR `pesos_dict` a
    suma=1 sin importar el origen de los pesos, lo que anulaba en la
    práctica la exposición reducida de Kelly (los retornos del meta-
    portafolio quedaban idénticos a como si se hubiera usado
    `pesos_asignacion()` normal). Con kappa_skill alto, la volatilidad
    realizada del meta-portafolio debe caer muy por debajo del baseline.
    """
    datos, benchmark = _datos_con_crash(n=300)

    def _correr(sizer):
        zoo = _zoo_dos_estrategias(datos)
        juez = TTTJuez()
        metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
        engine = BacktestEngine(
            zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
            position_sizer=sizer,
        )
        return engine.ejecutar_walk_forward(datos.index[0], datos.index[-1], frecuencia_rebalanceo=21)

    sin_sizer = _correr(None)
    con_sizer = _correr(KellyBayesianSizer(lam=0.5, kappa_skill=1.0))

    std_sin = sin_sizer.retornos_meta.std()
    std_con = con_sizer.retornos_meta.std()
    assert std_con < 0.5 * std_sin


# ---------------------------------------------------------------------------
# TakeProfitATRRule integrado
# ---------------------------------------------------------------------------

def test_tp_rule_intra_periodo_cambia_retornos_estrategia():
    datos, benchmark = _datos_rally(n=200)
    ohlc = _ohlc_desde_close(datos[TICKER])

    def _correr(tp_rule, datos_ohlc):
        zoo = _zoo_dos_estrategias(datos)
        juez = TTTJuez()
        metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
        engine = BacktestEngine(
            zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
            tp_rule=tp_rule, datos_ohlc=datos_ohlc,
        )
        return engine.ejecutar_walk_forward(datos.index[0], datos.index[-1], frecuencia_rebalanceo=21)

    sin_tp = _correr(None, None)
    con_tp = _correr(TakeProfitATRRule(tp_mult=1.0, recorte=0.5), {TICKER: ohlc})

    assert not sin_tp.retornos_estrategias["full"].equals(con_tp.retornos_estrategias["full"])


# ---------------------------------------------------------------------------
# Hook opcional §1.5: incertidumbre_regimen -> sigma_skill de Kelly
# ---------------------------------------------------------------------------

class _EstrategiaConIncertidumbreRegimen(_EstrategiaFija):
    """Estrategia sintética que expone `incertidumbre_regimen`, imitando el
    duck-typing hook que HMMGARCHStrategy implementa en §1.5."""

    def __init__(self, nombre: str, universo: List[str], peso: float, incertidumbre: float) -> None:
        super().__init__(nombre=nombre, universo=universo, peso=peso)
        self._incertidumbre = incertidumbre

    def incertidumbre_regimen(self, log_rets) -> float:
        return self._incertidumbre


def test_incertidumbre_regimen_hook_reduce_exposicion_kelly():
    """
    Motor._pesos_via_sizer debe sumar incertidumbre_regimen() (si la
    estrategia la expone) a sigma_skill_TTT antes de pasarla a Kelly. Con
    incertidumbre alta, la exposición de esa estrategia debe caer por
    debajo de la de una estrategia idéntica sin el hook.
    """
    datos, benchmark = _datos_con_crash(n=300)

    def _zoo_con_hook():
        zoo = ZooManager()
        zoo.agregar(_EstrategiaFija("sin_hook", universo=[TICKER], peso=1.0))
        zoo.agregar(_EstrategiaConIncertidumbreRegimen(
            "con_hook_incierto", universo=[TICKER], peso=1.0, incertidumbre=50.0
        ))
        return zoo

    juez = TTTJuez()
    metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
    sizer = KellyBayesianSizer(lam=0.5, kappa_skill=1.0, cap_individual=1.0, cap_bruto=10.0)
    engine = BacktestEngine(
        zoo=_zoo_con_hook(), metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
        position_sizer=sizer,
    )
    resultado = engine.ejecutar_walk_forward(
        fecha_inicio=datos.index[0], fecha_fin=datos.index[-1], frecuencia_rebalanceo=21,
    )

    assert not resultado.pesos_juez.empty
    # La estrategia con alta incertidumbre de régimen debe recibir, en
    # promedio, exposición estrictamente menor que la que no expone el hook.
    media_sin_hook = resultado.pesos_juez["sin_hook"].mean()
    media_con_hook = resultado.pesos_juez["con_hook_incierto"].mean()
    assert media_con_hook < media_sin_hook


def test_incertidumbre_regimen_hook_ausente_no_afecta_default():
    """Estrategias sin el método (mayoría del Zoo) no deben verse afectadas
    — _incertidumbre_regimen_extra debe devolver 0.0 silenciosamente."""
    datos, benchmark = _datos_con_crash(n=300)
    zoo = _zoo_dos_estrategias(datos)
    juez = TTTJuez()
    metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)
    engine = BacktestEngine(
        zoo=zoo, metricas=metricas, juez=juez, datos=datos, benchmark=benchmark,
    )
    extra = engine._incertidumbre_regimen_extra("full", datos.index[100])
    assert extra == 0.0
