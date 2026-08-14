# =============================================================================
# FILE: quant_arena/backtesting/position_risk.py
# Take-Profit por ATR + simulación intra-período con estado (path-dependency).
# =============================================================================
"""
`BacktestEngine._retornos_periodo` (motor.py) mantiene el peso de cada
estrategia CONSTANTE durante todo el período (t_{k-1}, t_k]: es un producto
punto vectorizado sobre el slice de precios. Eso es correcto para un
rebalanceo puro, pero no permite modelar un take-profit intra-período —que
por definición reacciona a un movimiento de precio DENTRO del período, antes
del siguiente rebalanceo.

Este módulo añade esa dependencia de trayectoria sin sacrificar el estilo
100% vectorizado del resto del proyecto (ver feature_engineer.py): el
recorte de exposición se computa como una serie booleana acumulada
(`cummax`) sobre la ganancia no realizada, no con un bucle Python día a día.

Mecanismo (Take-Profit por ATR):
    entrada       = precio del activo en fecha_desde (inicio del período).
    ganancia(t)   = precio(t) / entrada − 1.
    disparado(t)  = ganancia(t) >= tp_mult · atr_pct        # umbral relativo
    activo(t)     = disparado(t).cummax()                   # first-touch persistente
    factor(t)     = 1 − recorte · activo(t)

`atr_pct` (ATR(14) como fracción del precio de entrada) se calcula UNA vez
al inicio del período con `calcular_atr_pct`, reutilizando el ATR ya
implementado en `features/feature_engineer.py` (DRY — no se reimplementa
True Range). Es causal por construcción: solo usa datos <= fecha_desde.

NO-REGRESIÓN: si `tp_rule=None` (o `atr_pct` inválido), el factor es 1.0 en
todo el período y `retornos_periodo_con_riesgo` reproduce exactamente
`BacktestEngine._retornos_periodo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from quant_arena.core.excepciones import ConfiguracionInvalidaError
from quant_arena.core.interfaces_riesgo import AbstractRiskRule
from quant_arena.features.feature_engineer import FeatureConfig, FeatureEngineer


# =============================================================================
# TakeProfitATRRule — AbstractRiskRule
# =============================================================================

@dataclass(frozen=True)
class TakeProfitATRRule(AbstractRiskRule):
    """
    Recorte parcial de exposición al alcanzar `tp_mult · ATR%` de ganancia
    desde la entrada. Complementa (no reemplaza) el trailing stop de
    `RiskOverlay`: éste protege el capital agregado desde el pico histórico
    GLOBAL; `TakeProfitATRRule` asegura ganancias tácticas dentro de una
    única posición, período a período.

    Args:
        tp_mult: Múltiplo de ATR% que dispara el recorte. Default 2.0
                 (2×ATR de ganancia no realizada).
        recorte: Fracción de la exposición retirada al dispararse el TP,
                 en (0, 1]. Default 0.5 (recorte parcial — no cierre total).

    Raises:
        ConfiguracionInvalidaError: si tp_mult <= 0 o recorte fuera de (0, 1].
    """
    tp_mult: float = 2.0
    recorte: float = 0.5

    def __post_init__(self) -> None:
        if self.tp_mult <= 0.0:
            raise ConfiguracionInvalidaError(f"tp_mult={self.tp_mult} debe ser > 0.")
        if not (0.0 < self.recorte <= 1.0):
            raise ConfiguracionInvalidaError(f"recorte={self.recorte} fuera de (0, 1].")

    def aplicar(
        self,
        exposicion: pd.Series,
        contexto: Dict[str, pd.Series],
    ) -> pd.Series:
        """
        Args:
            exposicion: Exposición base (típicamente 1.0 constante — el
                        recorte se expresa como multiplicador relativo).
            contexto:   Debe incluir:
                          'precio':  pd.Series de precios diarios del activo,
                                     con `precio.iloc[0]` = precio de entrada
                                     (primer día del período, inclusive).
                          'atr_pct': float — ATR(14)/precio_entrada, fijado
                                     causalmente al inicio del período (no se
                                     recalcula intra-período).

        Returns:
            `exposicion` multiplicada por el factor de recorte (1.0 si el
            TP no se disparó, o `atr_pct` es inválido/ausente).
        """
        precio = contexto.get("precio")
        atr_pct = contexto.get("atr_pct", float("nan"))

        if precio is None or precio.empty or not np.isfinite(atr_pct) or atr_pct <= 0.0:
            return exposicion

        entrada = float(precio.iloc[0])
        if entrada <= 0.0:
            return exposicion

        ganancia = precio / entrada - 1.0
        disparado = ganancia >= (self.tp_mult * atr_pct)
        activo = disparado.cummax()
        factor = 1.0 - self.recorte * activo.astype(float)

        return exposicion * factor.reindex(exposicion.index).fillna(1.0)


# =============================================================================
# ATR% de entrada — reutiliza FeatureEngineer (DRY)
# =============================================================================

def calcular_atr_pct(
    datos_ohlc: pd.DataFrame,
    fecha: pd.Timestamp,
    atr_window: int = 14,
) -> float:
    """
    ATR(atr_window) como fracción del precio de cierre en `fecha`.

    Reutiliza `FeatureEngineer` (features/feature_engineer.py) para el
    cálculo de True Range — no se reimplementa aquí.

    Causalidad: opera sobre `datos_ohlc.loc[:fecha]` (solo datos <= fecha),
    por lo que el ATR% resultante no contiene información posterior.

    Args:
        datos_ohlc: DataFrame con columnas ['open', 'high', 'low', 'close',
                    'volume'] (minúsculas, contrato de FeatureEngineer) y
                    DatetimeIndex, para UN solo activo.
        fecha:      Fecha de referencia (inclusive) — típicamente el inicio
                    del período de rebalanceo.
        atr_window: Ventana del ATR (default 14, estándar Wilder).

    Returns:
        ATR(atr_window)/close en `fecha`, o NaN si no hay historial
        suficiente o el precio de cierre no es positivo.
    """
    slice_causal = datos_ohlc.loc[:fecha]
    if len(slice_causal) < atr_window + 1:
        return float("nan")

    fe = FeatureEngineer(FeatureConfig(atr_window=atr_window))
    try:
        out = fe.transform(slice_causal)
    except ValueError:
        return float("nan")

    atr = float(out["atr"].iloc[-1])
    close = float(out["close"].iloc[-1])
    if not np.isfinite(atr) or close <= 0.0:
        return float("nan")
    return atr / close


# =============================================================================
# Simulación intra-período con estado (path-dependent)
# =============================================================================

def retornos_periodo_con_riesgo(
    pesos: pd.Series,
    datos: pd.DataFrame,
    fecha_desde: pd.Timestamp,
    fecha_hasta: pd.Timestamp,
    atr_pct: Optional[Mapping[str, float]] = None,
    tp_rule: Optional[TakeProfitATRRule] = None,
) -> pd.Series:
    """
    Variante path-dependent de `BacktestEngine._retornos_periodo`: aplica
    `tp_rule` día a día (vectorizado vía `cummax`, sin bucle Python) antes
    de agregar los retornos del portafolio para el período (fecha_desde,
    fecha_hasta].

    NO-REGRESIÓN: con `tp_rule=None` (o sin entradas válidas en `atr_pct`),
    el resultado es idéntico a `BacktestEngine._retornos_periodo` — mismo
    contrato de entrada/salida, mismo tratamiento de activos sin peso
    significativo y de períodos vacíos.

    Args:
        pesos:        pd.Series [ticker -> peso] generados en fecha_desde.
        datos:        DataFrame de precios [fecha x ticker] (columna por
                      activo, valores de cierre), cubriendo al menos
                      [fecha_desde, fecha_hasta].
        fecha_desde:  Fecha base (inclusive para pct_change, excluida del
                      output) — también la fecha de entrada para el TP.
        fecha_hasta:  Última fecha del período (inclusive en el output).
        atr_pct:      {ticker: ATR%} calculado causalmente en fecha_desde
                      (ver `calcular_atr_pct`). None o ticker ausente
                      desactiva el TP para ese activo.
        tp_rule:      Regla de Take-Profit a aplicar. None desactiva el TP
                      por completo (comportamiento idéntico al vectorizado).

    Returns:
        pd.Series de retornos diarios indexados en (fecha_desde, fecha_hasta].
    """
    activos = [
        a for a in pesos.index
        if a in datos.columns and abs(pesos.get(a, 0.0)) > 1e-10
    ]

    fechas_slice = datos.index[
        (datos.index >= fecha_desde) & (datos.index <= fecha_hasta)
    ]

    if len(fechas_slice) < 2:
        return pd.Series(dtype=float)

    if not activos:
        return pd.Series(0.0, index=fechas_slice[1:], dtype=float)

    precios = datos.loc[fechas_slice, activos]
    retornos_activos = precios.pct_change().iloc[1:]

    multiplicadores = pd.DataFrame(1.0, index=retornos_activos.index, columns=activos)

    if tp_rule is not None and atr_pct:
        exposicion_base = pd.Series(1.0, index=precios.index)
        for activo in activos:
            atr_a = atr_pct.get(activo, float("nan"))
            if not np.isfinite(atr_a) or atr_a <= 0.0:
                continue
            contexto = {"precio": precios[activo], "atr_pct": atr_a}
            factor_completo = tp_rule.aplicar(exposicion_base, contexto)
            multiplicadores[activo] = factor_completo.iloc[1:]

    pesos_validos = pesos[activos]
    retornos_ajustados = retornos_activos * multiplicadores
    return (retornos_ajustados * pesos_validos).sum(axis=1)
