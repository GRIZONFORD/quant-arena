# beta_neutralizer.py
"""
BetaNeutralizer — Betas rodantes + proyección ortogonal de doble restricción
============================================================================
Tier 1.5 · Aislamiento del alpha residual (idiosincrásico).

Logra que el portafolio sea SIMULTÁNEAMENTE:
    Σ wᵢ      = 0   (Dollar-Neutral)
    Σ wᵢ·βᵢ   = 0   (Beta-Neutral)

mediante PROYECCIÓN ORTOGONAL ANALÍTICA (sin `scipy.optimize`, sin bucles por
fecha): se proyecta el vector de pesos inicial w₀ sobre el complemento ortogonal
del subespacio generado por 𝟙 (unos) y β:

        w = w₀ − Cᵀ (C Cᵀ)⁻¹ C w₀ ,      C = [ 𝟙ᵀ ; βᵀ ]   (2×N, sobre activos activos)

Como C Cᵀ es 2×2 con elementos (n, Σβ, Σβ, Σβ²), su inversa se calcula en forma
cerrada y se vectoriza sobre todas las fechas con reducciones por fila.

BETAS RODANTES (OLS vectorizado, sin bucle por activo):
    βᵢ = Cov_w(Rᵢ, Rₘ) / Var_w(Rₘ) = (E_w[Rᵢ·Rₘ] − E_w[Rᵢ]·E_w[Rₘ]) / Var_w(Rₘ)
con `shift(1)` estricto ⇒ la β usada en t se estimó solo con datos ≤ t−1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class BetaNeutralizer:
    """
    Estima betas rodantes y neutraliza pesos a doble restricción (Σw=0 ∧ Σwβ=0).

    Args:
        window:      Ventana de la regresión rodante (días).
        min_periods: Mínimo de observaciones para una beta válida (robustez NaN).
        lag:         Rezago de la beta (1 = anti look-ahead estricto).
        ridge:       Regularización sobre Var(Rₘ) (estabilidad numérica).
        det_tol:     Tolerancia del determinante 2×2 (fechas degeneradas → flat).
    """

    def __init__(
        self,
        window: int = 126,
        min_periods: int = 60,
        lag: int = 1,
        ridge: float = 1e-12,
        det_tol: float = 1e-12,
    ) -> None:
        if window < 2:
            raise ValueError("window debe ser ≥ 2.")
        self.window = window
        self.min_periods = min_periods
        self.lag = lag
        self.ridge = ridge
        self.det_tol = det_tol

    # ── 1. BETAS RODANTES (vectorizado, anti look-ahead) ────────────────────────

    def compute_rolling_betas(
        self, returns: pd.DataFrame, market_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Beta rodante OLS de cada activo vs. el mercado (matriz Date×Ticker).

        Vectorizado vía momentos rodantes sobre la matriz wide completa — sin
        iterar por activo. Resultado rezagado `lag` días (β en t estimada ≤ t−1).
        """
        R = returns
        Rm = market_returns.reindex(R.index)
        w, mp = self.window, self.min_periods

        mean_Ri   = R.rolling(w, min_periods=mp).mean()                 # E_w[Rᵢ]
        mean_Rm   = Rm.rolling(w, min_periods=mp).mean()                # E_w[Rₘ]
        mean_RiRm = R.mul(Rm, axis=0).rolling(w, min_periods=mp).mean() # E_w[Rᵢ·Rₘ]
        mean_Rm2  = Rm.pow(2).rolling(w, min_periods=mp).mean()         # E_w[Rₘ²]

        cov   = mean_RiRm.sub(mean_Ri.mul(mean_Rm, axis=0), axis=0)     # Cov_w(Rᵢ,Rₘ)
        var_m = (mean_Rm2 - mean_Rm.pow(2)) + self.ridge               # Var_w(Rₘ)
        beta  = cov.div(var_m, axis=0)

        return beta.shift(self.lag)                                     # anti look-ahead

    # ── 2. PROYECCIÓN ORTOGONAL DE DOBLE RESTRICCIÓN ────────────────────────────

    def neutralize_weights(
        self,
        weights: pd.DataFrame,
        betas: pd.DataFrame,
        leverage: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Proyecta w₀ sobre el complemento ortogonal de span{𝟙, β}, restringido a la
        cesta activa (m = pesos≠0) para no asignar peso fuera del cuantil.

        Implementa, vectorizado sobre fechas:
            w = w₀ − Cᵀ (C Cᵀ)⁻¹ C w₀,   C = [m ; m·β]

        Args:
            weights:  w₀ (Date×Ticker) — típicamente ya dollar-neutral.
            betas:    β (Date×Ticker) rezagadas.
            leverage: Σ|w| objetivo tras proyectar. Si None, conserva el Σ|w₀| por fila.

        Returns:
            Pesos proyectados (Date×Ticker) con Σw=0 ∧ Σwβ=0 por fila.
        """
        w0 = weights
        b = betas.reindex_like(w0).fillna(0.0)
        m = (w0.abs() > 1e-12).astype(float)        # máscara de activos activos (𝟙 restringido)

        # ── Elementos de C Cᵀ (2×2 por fecha): [[n, Σβ], [Σβ, Σβ²]] ───────────────
        n   = m.sum(axis=1)                           # n   = Σ m
        sb  = (m * b).sum(axis=1)                      # Σβ  = Σ m·β
        sbb = (m * b * b).sum(axis=1)                  # Σβ² = Σ m·β²

        # ── C·w₀  (vector 2×1 por fecha): [Σw₀, Σw₀·β] ────────────────────────────
        c0  = w0.sum(axis=1)                           # (C w₀)₁ = Σ w₀   (≈0)
        cb  = (w0 * b).sum(axis=1)                      # (C w₀)₂ = Σ w₀·β

        # ── (C Cᵀ)⁻¹ en forma cerrada 2×2: (1/det)·[[Σβ², −Σβ], [−Σβ, n]] ─────────
        det = (n * sbb - sb * sb)                      # det(C Cᵀ)
        det = det.where(det.abs() > self.det_tol, np.nan)
        # λ = (C Cᵀ)⁻¹ (C w₀)
        lam1 = (sbb * c0 - sb * cb) / det              # λ₁
        lam2 = (n * cb - sb * c0) / det                # λ₂

        # ── Cᵀ·λ = m·(λ₁ + λ₂·β)  →  ajuste solo sobre activos activos ────────────
        adj = m.mul(lam1, axis=0) + (m * b).mul(lam2, axis=0)
        w = w0 - adj                                   # w = w₀ − Cᵀ(CCᵀ)⁻¹C w₀

        # ── Reescalado escalar a Σ|w| objetivo (preserva Σw=0 ∧ Σwβ=0) ───────────
        target = (
            w0.abs().sum(axis=1) if leverage is None
            else pd.Series(float(leverage), index=w0.index)
        )
        gross = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.mul(target / gross, axis=0)
        return w.fillna(0.0)

    # ── UTILIDADES ──────────────────────────────────────────────────────────────

    def residualize(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        betas: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Retornos residuales (alpha): εᵢ(t) = Rᵢ(t) − βᵢ(t)·Rₘ(t)."""
        if betas is None:
            betas = self.compute_rolling_betas(returns, market_returns)
        Rm = market_returns.reindex(returns.index)
        return returns.sub(betas.mul(Rm, axis=0))

    @staticmethod
    def portfolio_beta(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
        """Beta neta del portafolio por fecha: Σᵢ wᵢ(t)·βᵢ(t)."""
        b = betas.reindex_like(weights).fillna(0.0)
        return (weights * b).sum(axis=1)

    def __repr__(self) -> str:
        return (f"BetaNeutralizer(window={self.window}, "
                f"min_periods={self.min_periods}, lag={self.lag})")
