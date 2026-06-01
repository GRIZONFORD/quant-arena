# risk_overlay.py
"""
RiskOverlay — Capa de Gestión de Riesgo con ESTADO GLOBAL (Stateful)
====================================================================
Proyecto Paraguay V2.1 · Drawdown Protection

Escala la exposición "desnuda" de cualquier estrategia ANTES de computar sus
retornos, para proteger el capital frente a la beta del mercado en crisis
sistémicas (el −55% de drawdown de la GFC/COVID en SPY Buy&Hold).

DOS MECANISMOS (vectorizados dentro del fold; estado persistente entre folds):

1. TARGET VOLATILITY (dimensionamiento dinámico)
   ────────────────────────────────────────────────────────────────────────
       risk_weight_tv(t) = clip( target_vol / realized_vol_ann(t−1), 0, cap )
   · target_vol : volatilidad anualizada objetivo (default 15%).
   · cap        : apalancamiento máximo. DEFAULT 1.0 (solo REDUCE exposición —
                  el cap previo de 1.5x amplificaba estrategias malas).

2. TRAILING STOP-LOSS GLOBAL Y PERSISTENTE (circuit-breaker)
   ────────────────────────────────────────────────────────────────────────
   ⚠ FIX V2.1: antes el `cummax()` se reiniciaba dentro de cada fold de 21 días
   (stop "fold-local"), permitiendo sangrados multi-mes (Donchian −65% pese al
   límite −15%). Ahora el overlay GUARDA el equity vol-targeted acumulado y el
   High-Water Mark GLOBAL de cada estrategia, y mide el drawdown contra ese pico
   histórico que atraviesa TODOS los folds:

       hwm_global(t) = max( hwm_previo,  max_{s≤t} equity_tv(s) )
       drawdown(t)   = equity_tv(t) / hwm_global(t) − 1
       participa(t)  = drawdown(t) > −dd_limit        # en mercado si dentro del límite
       gate          = participa.shift(1)             # anti look-ahead

   La exposición se corta a 0 mientras el drawdown desde el pico GLOBAL supere
   −dd_limit, y se re-activa cuando el equity vol-targeted recupera ese umbral.

ANTI LOOK-AHEAD:
   · Target Vol usa `.shift(1)` (vol de t−1 escala la posición de t).
   · El gate del stop usa `.shift(1)` (drawdown observado en t corta desde t+1).
   · El estado (equity/HWM previos) proviene EXCLUSIVAMENTE de folds anteriores
     (OOS pasado), nunca del futuro.

ESTADO: `_state[strategy_name] = (equity_tv, hwm_global, in_market)`.
   · Se actualiza al final de cada fold y se consume al inicio del siguiente.
   · `reset()` lo limpia → cada torneo nuevo parte de equity=1.0, HWM=1.0
     (evita contaminación entre ejecuciones independientes).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskOverlay:
    """
    Overlay de riesgo STATEFUL aplicable al vector de pesos de cualquier estrategia.

    Args:
        target_vol:   Volatilidad anualizada objetivo (default 0.15 = 15%).
        max_leverage: Cap de exposición (DEFAULT 1.0 = sin apalancamiento).
        vol_window:   Ventana de vol realizada si hay que calcularla al vuelo.
        vol_col:      Columna de vol anualizada preexistente en market_data.
        enable_stop:  Activa el trailing stop-loss global por drawdown.
        dd_limit:     Umbral de drawdown (desde el HWM GLOBAL) que corta a efectivo.
        min_periods:  Mínimo de obs para la vol al vuelo.
    """

    def __init__(
        self,
        target_vol:   float = 0.15,
        max_leverage: float = 1.0,
        vol_window:   int   = 21,
        vol_col:      str   = "realized_vol",
        enable_stop:  bool  = True,
        dd_limit:     float = 0.15,
        min_periods:  int   = 10,
    ) -> None:
        self.target_vol   = target_vol
        self.max_leverage = max_leverage
        self.vol_window   = vol_window
        self.vol_col      = vol_col
        self.enable_stop  = enable_stop
        self.dd_limit     = dd_limit
        self.min_periods  = min_periods
        # Estado global persistente por estrategia: (equity_tv, hwm_global, in_market)
        self._state: Dict[str, Tuple[float, float, bool]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Limpia el estado global. Llamar al inicio de cada torneo nuevo."""
        self._state.clear()

    def _annualized_vol(
        self, market_data: pd.DataFrame, price_ret: pd.Series
    ) -> pd.Series:
        """
        Volatilidad anualizada: usa el feature `realized_vol` si está presente
        (consistencia con el FeatureEngineer); si no, la calcula vectorizada
        como σ rolling de los retornos · √252.
        """
        if self.vol_col in market_data.columns:
            vol = market_data[self.vol_col].astype(float)
        else:
            vol = price_ret.rolling(
                self.vol_window, min_periods=self.min_periods
            ).std() * np.sqrt(252)
        # Saneado: vol nula/NaN no debe inflar la exposición (división por ~0).
        return vol.replace(0.0, np.nan)

    def compute_risk_weight(
        self,
        base_weight: pd.Series,
        price_ret:   pd.Series,
        market_data: pd.DataFrame,
        strategy_name: str = "_default",
    ) -> pd.Series:
        """
        Devuelve el peso de la estrategia ya escalado por el Risk Overlay,
        usando y actualizando el estado GLOBAL de `strategy_name`.

        `base_weight` es la posición fraccional que ENTRA al día t
        (típicamente `positions.shift(1)/capital`, ya sin look-ahead).
        """
        vol = self._annualized_vol(market_data, price_ret)

        # ── 1. Target Volatility (vol de t−1 → posición de t) ───────────────
        tv = (self.target_vol / vol.shift(1)).clip(lower=0.0, upper=self.max_leverage)
        tv = tv.reindex(base_weight.index).fillna(0.0)
        w_tv = base_weight * tv

        if not self.enable_stop:
            return w_tv

        # ── 2. Trailing Stop GLOBAL Y PERSISTENTE ───────────────────────────
        # Estado heredado de folds OOS anteriores (1.0 si es el primer fold).
        prior_eq, prior_hwm, prior_in = self._state.get(
            strategy_name, (1.0, 1.0, True)
        )

        # Equity vol-targeted acumulado DESDE el estado previo (no reinicia a 1.0).
        pre_ret = (w_tv * price_ret).fillna(0.0)
        equity_tv = prior_eq * (1.0 + pre_ret).cumprod()

        # HWM GLOBAL: máximo entre el pico previo y el running-max de este fold.
        hwm = equity_tv.cummax().clip(lower=prior_hwm)
        drawdown = equity_tv / hwm - 1.0

        # Participa si el drawdown desde el pico GLOBAL no excede −dd_limit.
        participate = drawdown > -self.dd_limit
        gate = participate.shift(1)
        # El día 0 del fold hereda el estado de participación del fold anterior.
        gate.iloc[0] = prior_in
        gate = gate.fillna(prior_in).astype(float)

        # ── Persistir estado para el siguiente fold ─────────────────────────
        self._state[strategy_name] = (
            float(equity_tv.iloc[-1]),
            float(hwm.iloc[-1]),
            bool(participate.iloc[-1]),
        )

        return w_tv * gate

    def describe(self) -> str:
        s = f"TargetVol={self.target_vol:.0%} | cap={self.max_leverage:.1f}x"
        if self.enable_stop:
            s += f" | TrailingStop GLOBAL@−{self.dd_limit:.0%}"
        return s
