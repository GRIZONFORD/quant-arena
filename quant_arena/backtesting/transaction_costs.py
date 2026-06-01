# transaction_costs.py
"""
TransactionCostModel — Fricciones de trading (comisión + slippage)
==================================================================
Proyecto Paraguay · Tier 0 · Eliminación del sesgo de backtest sin fricciones.

Encapsula el modelado de costos de transacción de forma POO y 100% vectorizada.
El `BacktestEngine` inyecta una instancia de esta clase para descontar los
costos de la curva de retornos ANTES de que el Juez TTT puntúe por Sharpe — de
modo que la habilidad se mide sobre retornos NETOS, no brutos.

MODELO DE COSTOS (lineal en turnover):
    cost(t) = turnover(t) · (commission_bps + slippage_bps) / 1e4
    turnover(t) = |weight(t) − weight(t−1)|          # cambio de exposición
    net_ret(t)  = gross_ret(t) − cost(t)

donde `weight(t)` es la exposición fraccional efectiva durante el día t (ya
post Risk Overlay). El primer día se asume entrada desde efectivo (turnover=|w₀|).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostReport:
    """Resultado inmutable de aplicar el modelo de costos a una serie de retornos."""
    gross_returns: pd.Series
    net_returns:   pd.Series
    costs:         pd.Series
    turnover:      pd.Series

    @property
    def total_cost(self) -> float:
        """Costo acumulado total (suma de fricciones) sobre el periodo."""
        return float(self.costs.sum())

    @property
    def avg_turnover(self) -> float:
        """Turnover medio por periodo (proxy de actividad/rotación)."""
        return float(self.turnover.mean())

    @property
    def annualized_cost_drag(self, periods_per_year: int = 252) -> float:
        """Lastre de costo anualizado (coste medio diario · 252)."""
        return float(self.costs.mean() * periods_per_year)


class TransactionCostModel:
    """
    Modelo de costos de transacción lineal en turnover.

    Args:
        commission_bps: Comisión por unidad de turnover (one-way), en puntos básicos.
        slippage_bps:   Slippage/impacto por unidad de turnover (one-way), en bps.
        name:           Etiqueta para trazabilidad/logging.

    Ejemplo:
        cm = TransactionCostModel(commission_bps=0.5, slippage_bps=1.0)
        report = cm.apply(gross_returns, weight)
        net = report.net_returns
    """

    def __init__(
        self,
        commission_bps: float = 0.5,
        slippage_bps:   float = 1.0,
        name:           str   = "linear_bps",
    ) -> None:
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("Los costos en bps no pueden ser negativos.")
        self.commission_bps = float(commission_bps)
        self.slippage_bps   = float(slippage_bps)
        self.name           = name

    # ── API PÚBLICA ────────────────────────────────────────────────────────────

    @property
    def cost_per_turnover(self) -> float:
        """Fracción de costo por unidad de turnover one-way (bps → fracción)."""
        return (self.commission_bps + self.slippage_bps) / 1.0e4

    def compute_turnover(self, weight: pd.Series) -> pd.Series:
        """
        Turnover por periodo = |Δ exposición|, 100% vectorizado.

        El primer periodo se trata como entrada desde efectivo: turnover = |w₀|.
        """
        w = weight.astype(float).fillna(0.0)
        turnover = w.diff().abs()
        turnover.iloc[0] = abs(w.iloc[0]) if len(w) else 0.0
        return turnover

    def compute_costs(self, weight: pd.Series) -> pd.Series:
        """Serie de costos por periodo = turnover · cost_per_turnover (vectorizado)."""
        return self.compute_turnover(weight) * self.cost_per_turnover

    def apply(self, gross_returns: pd.Series, weight: pd.Series) -> CostReport:
        """
        Descuenta los costos de la serie de retornos brutos.

        Args:
            gross_returns: Retornos brutos del día t (= weight · price_ret).
            weight:        Exposición fraccional efectiva en el día t.

        Returns:
            CostReport con retornos brutos, netos, costos y turnover alineados.
        """
        gross = gross_returns.astype(float).fillna(0.0)
        costs = self.compute_costs(weight).reindex(gross.index).fillna(0.0)
        net   = gross - costs
        return CostReport(
            gross_returns=gross,
            net_returns=net,
            costs=costs,
            turnover=self.compute_turnover(weight).reindex(gross.index).fillna(0.0),
        )

    # ── INTEGRACIÓN CON PANELES (Tier 1) ───────────────────────────────────────

    def costs_from_turnover(self, turnover: pd.Series) -> pd.Series:
        """
        Convierte una serie de turnover YA agregado (p.ej. el turnover de panel
        Σᵢ|Δwᵢ| por fecha) en costos por periodo. Permite que el motor
        cross-sectional calcule su propio turnover de panel y delegue solo la
        conversión bps→costo a esta clase (Single Responsibility).
        """
        return turnover.astype(float).fillna(0.0) * self.cost_per_turnover

    def apply_with_turnover(
        self, gross_returns: pd.Series, turnover: pd.Series
    ) -> CostReport:
        """
        Como `apply`, pero recibe el turnover ya calculado (caso panel) en lugar
        de derivarlo de un único vector de pesos. Alinea, descuenta y reporta.
        """
        gross = gross_returns.astype(float).fillna(0.0)
        costs = self.costs_from_turnover(turnover).reindex(gross.index).fillna(0.0)
        return CostReport(
            gross_returns=gross,
            net_returns=gross - costs,
            costs=costs,
            turnover=turnover.reindex(gross.index).fillna(0.0),
        )

    def __repr__(self) -> str:
        return (f"TransactionCostModel(commission_bps={self.commission_bps}, "
                f"slippage_bps={self.slippage_bps}, "
                f"per_turnover={self.cost_per_turnover*1e4:.2f}bps)")
