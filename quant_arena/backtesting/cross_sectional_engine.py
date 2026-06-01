# cross_sectional_engine.py
"""
CrossSectionalEngine — Motor de portafolio long/short de sección cruzada (Tier 1)
=================================================================================
Reemplaza el backtest mono-activo por uno de PANEL: recibe una matriz de scores
para todo el universo y construye, de forma 100% vectorizada (sin bucles sobre
fechas), un vector de pesos por activo con restricciones de portafolio:

  · Dollar-Neutral:        Σᵢ wᵢ = 0           (long top q%, short bottom q%)
  · Apalancamiento bruto:  Σᵢ |wᵢ| = leverage  (p.ej. 2.0 → 100% long / 100% short)
  · Ponderación:           equal-weight | inverse-vol dentro de cada pata.

El `TransactionCostModel` se INYECTA por constructor (Dependency Injection) y
se usa para deducir las fricciones del turnover de panel Σᵢ|Δwᵢ| por fecha.
La salida (`PortfolioResult.net_returns`) es una Serie agregada de portafolio
directamente compatible con `StatisticalSignificanceValidator`.

ALINEACIÓN ANTI LOOK-AHEAD:
  scores(t) conocidos al cierre de t → weights(t). El retorno se materializa en
  t+1: r_p(t+1) = Σᵢ wᵢ(t)·rᵢ(t+1). Implementado como `weights.shift(1) * returns`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quant_arena.backtesting.transaction_costs import TransactionCostModel
from quant_arena.metricas.beta_neutralizer import BetaNeutralizer


@dataclass(frozen=True)
class PortfolioResult:
    """Resultado inmutable de un backtest de panel cross-sectional."""
    weights:       pd.DataFrame   # Date × Ticker (pesos objetivo por fecha)
    gross_returns: pd.Series      # retorno de portafolio bruto (por fecha)
    net_returns:   pd.Series      # NETO de costos → input del validador
    costs:         pd.Series      # costo de portafolio por fecha
    turnover:      pd.Series      # Σᵢ|Δwᵢ| por fecha

    @property
    def avg_turnover(self) -> float:
        return float(self.turnover.mean())

    @property
    def avg_gross_leverage(self) -> float:
        """Σᵢ|wᵢ| medio (debe ≈ leverage objetivo en fechas con universo pleno)."""
        return float(self.weights.abs().sum(axis=1).mean())

    @property
    def avg_net_exposure(self) -> float:
        """Σᵢ wᵢ medio (≈0 si dollar-neutral)."""
        return float(self.weights.sum(axis=1).mean())


class CrossSectionalEngine:
    """
    Motor de backtest cross-sectional long/short con costos inyectados.

    Args:
        cost_model:     Instancia de TransactionCostModel (Dependency Injection).
        long_pct:       Fracción superior del universo a comprar (default 0.20).
        short_pct:      Fracción inferior del universo a vender (default 0.20).
        leverage:       Apalancamiento bruto objetivo Σ|w| (default 2.0).
        dollar_neutral: Si True, long/short con Σw=0; si False, long-only top q%.
        weighting:      'equal' | 'inverse_vol'.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        leverage: float = 2.0,
        dollar_neutral: bool = True,
        weighting: str = "equal",
        beta_neutralizer: Optional[BetaNeutralizer] = None,
    ) -> None:
        if weighting not in ("equal", "inverse_vol"):
            raise ValueError("weighting debe ser 'equal' o 'inverse_vol'.")
        if not (0.0 < long_pct <= 1.0) or not (0.0 < short_pct <= 1.0):
            raise ValueError("long_pct y short_pct deben estar en (0, 1].")
        self.cost_model = cost_model
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.leverage = leverage
        self.dollar_neutral = dollar_neutral
        self.weighting = weighting
        # Tier 1.5: neutralizador inyectado (DI). None ⇒ solo dollar-neutral.
        self.beta_neutralizer = beta_neutralizer

    # ── CONSTRUCCIÓN DE PESOS (vectorizada, row-wise sobre el panel wide) ────────

    def compute_weights(
        self,
        scores: pd.DataFrame,
        vol: Optional[pd.DataFrame] = None,
        beta: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Convierte una matriz de scores Date×Ticker en pesos Date×Ticker.

        Ranking percentil POR FECHA (`rank(axis=1, pct=True)`, robusto a universos
        de tamaño variable y a NaN de activos no disponibles). Selecciona top/bottom
        cuantil y normaliza cada pata al apalancamiento objetivo. Si `beta_neutral`
        y se provee `beta`, proyecta los pesos a la doble restricción (Σw=0 ∧ Σwβ=0).
        """
        ranks = scores.rank(axis=1, pct=True)            # NaN-safe, por fila
        long_mask = ranks > (1.0 - self.long_pct)
        short_mask = ranks <= self.short_pct

        # Base de ponderación: 1 (equal) o 1/vol (inverse-vol) dentro de cada pata.
        if self.weighting == "inverse_vol" and vol is not None:
            basis = (1.0 / vol.replace(0.0, np.nan)).reindex_like(scores)
        else:
            basis = pd.DataFrame(1.0, index=scores.index, columns=scores.columns)
        basis = basis.where(scores.notna(), np.nan)

        if self.dollar_neutral:
            half = self.leverage / 2.0
            w_long = self._normalize_leg(basis.where(long_mask), half)
            w_short = self._normalize_leg(basis.where(short_mask), half)
            weights = w_long.sub(w_short, fill_value=0.0)
        else:  # long-only: top cuantil escalado al apalancamiento pleno
            weights = self._normalize_leg(basis.where(long_mask), self.leverage)

        weights = weights.fillna(0.0)

        # Tier 1.5: inyección del paso de neutralización de beta (DI). El engine
        # NO reimplementa la proyección: delega en BetaNeutralizer.neutralize_weights.
        if self.beta_neutralizer is not None and beta is not None:
            weights = self.beta_neutralizer.neutralize_weights(
                weights, beta, leverage=self.leverage
            )

        return weights

    @staticmethod
    def _normalize_leg(leg_basis: pd.DataFrame, gross_target: float) -> pd.DataFrame:
        """Normaliza una pata para que Σ|pesos| de la fila = gross_target."""
        row_sum = leg_basis.sum(axis=1)
        # div por fila; filas sin miembros (sum=0/NaN) → 0 (sin posición esa fecha).
        normalized = leg_basis.div(row_sum, axis=0).fillna(0.0) * gross_target
        return normalized

    # ── BACKTEST DEL PANEL ──────────────────────────────────────────────────────

    def run(
        self,
        scores: pd.DataFrame,
        asset_returns: pd.DataFrame,
        vol: Optional[pd.DataFrame] = None,
        beta: Optional[pd.DataFrame] = None,
    ) -> PortfolioResult:
        """
        Ejecuta el backtest cross-sectional y devuelve un `PortfolioResult`.

        Args:
            scores:        Matriz Date×Ticker de señales (mayor = más atractivo).
            asset_returns: Matriz Date×Ticker de retornos simples CONTEMPORÁNEOS.
            vol:           (opcional) Matriz Date×Ticker de vol para inverse-vol.
            beta:          (opcional) Matriz Date×Ticker de betas rezagadas; si
                           `beta_neutral=True`, proyecta los pesos a Σwβ=0.

        Returns:
            PortfolioResult con pesos, retornos brutos/netos, costos y turnover.
        """
        weights = self.compute_weights(scores, vol, beta)

        # Alinear columnas e índice de retornos con los pesos.
        rets = asset_returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)

        # Posición de t materializa retorno en t+1 → weights.shift(1).
        shifted = weights.shift(1)
        gross = (shifted * rets).sum(axis=1)

        # Turnover de panel alineado al timeline de retornos: Σᵢ|wᵢ(t-1) − wᵢ(t-2)|.
        turnover = shifted.diff().abs().sum(axis=1)

        # Fricciones vía el modelo inyectado (delegación de la conversión bps→costo).
        report = self.cost_model.apply_with_turnover(gross, turnover)

        return PortfolioResult(
            weights=weights,
            gross_returns=report.gross_returns,
            net_returns=report.net_returns,
            costs=report.costs,
            turnover=report.turnover,
        )

    def __repr__(self) -> str:
        mode = "dollar-neutral" if self.dollar_neutral else "long-only"
        return (f"CrossSectionalEngine({mode}, long={self.long_pct:.0%}, "
                f"short={self.short_pct:.0%}, leverage={self.leverage}, "
                f"weighting='{self.weighting}', {self.cost_model!r})")
