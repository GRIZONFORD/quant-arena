# statistical_validation.py
"""
StatisticalSignificanceValidator — Puertas de significancia robustas
====================================================================
Proyecto Paraguay · Tier 0 · Validación estadística contra el sesgo de selección.

Implementa, en POO estricta y sin dependencias frágiles (usa `statistics.NormalDist`
de la stdlib), las pruebas que distinguen alpha real de suerte / sobreajuste:

  · Probabilistic Sharpe Ratio (PSR)  — Bailey & López de Prado (2012).
  · Deflated Sharpe Ratio (DSR)       — Bailey & López de Prado (2014).
  · Probability of Backtest Overfitting (PBO) vía CSCV — Bailey et al. (2017).

FUNDAMENTO — DEFLATED SHARPE RATIO
──────────────────────────────────────────────────────────────────────────────
El Sharpe muestral está sesgado al alza por (a) no-normalidad de los retornos,
(b) longitud finita del track, y (c) MÚLTIPLES ENSAYOS (probar N estrategias y
quedarse con la mejor infla el máximo esperado bajo la hipótesis nula).

  PSR(SR*) = Φ[ (ŜR − SR*)·√(n−1) / √(1 − γ₃·ŜR + ((γ₄−1)/4)·ŜR²) ]

  donde ŜR = Sharpe muestral (por periodo), γ₃ = asimetría, γ₄ = curtosis
  (no-excesiva, 3 para la normal), n = nº de observaciones, Φ = CDF normal.

  El umbral de deflación SR₀ = E[máx Sharpe] esperado bajo la nula con N ensayos:

  SR₀ = √V · [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

  donde V = varianza de los Sharpe de los N ensayos, γ = constante de
  Euler-Mascheroni (≈0.5772), Z⁻¹ = CDF normal inversa (ppf).

  DSR = PSR(SR₀)  →  P(el Sharpe observado supera el máximo esperado por azar).
  Regla práctica: DSR ≥ 0.95 ⇒ significativo al 95%.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e as _E
from statistics import NormalDist
from typing import List

import numpy as np
import pandas as pd

_EULER_MASCHERONI: float = 0.5772156649015329
_NORM = NormalDist(0.0, 1.0)


@dataclass(frozen=True)
class SignificanceReport:
    """Resultado inmutable de la evaluación de significancia de un track OOS."""
    n_obs:            int
    sharpe_periodic:  float
    sharpe_annual:    float
    skew:             float
    kurtosis:         float
    psr_zero:         float    # PSR contra SR*=0
    sr0_deflation:    float    # umbral E[máx Sharpe] bajo N ensayos
    dsr:              float    # Deflated Sharpe Ratio
    n_trials:         int

    @property
    def is_significant(self, alpha: float = 0.95) -> bool:
        """True si el DSR supera el nivel de confianza (default 95%)."""
        return np.isfinite(self.dsr) and self.dsr >= alpha


class StatisticalSignificanceValidator:
    """
    Validador de significancia estadística para tracks de retornos OOS.

    Diseño POO: métodos puros y componibles. Los estáticos calculan momentos y
    Sharpe; los de instancia encapsulan la convención de anualización y exponen
    PSR, DSR y PBO como puertas de decisión.

    Args:
        periods_per_year: Convención de anualización (252 para diario).
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = periods_per_year

    # ── MOMENTOS Y SHARPE (estáticos, vectorizados) ────────────────────────────

    @staticmethod
    def _clean(returns: pd.Series) -> pd.Series:
        return returns.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def sharpe_periodic(returns: pd.Series) -> float:
        """Sharpe por periodo (sin anualizar): media/desv.estándar."""
        r = StatisticalSignificanceValidator._clean(returns)
        if len(r) < 2:
            return float("nan")
        sd = float(r.std(ddof=1))
        return float(r.mean() / sd) if sd > 1e-12 else float("nan")

    def sharpe_annual(self, returns: pd.Series) -> float:
        """Sharpe anualizado = Sharpe periódico · √(periods_per_year)."""
        sr = self.sharpe_periodic(returns)
        return sr * np.sqrt(self.periods_per_year) if np.isfinite(sr) else float("nan")

    # ── PROBABILISTIC SHARPE RATIO ─────────────────────────────────────────────

    def probabilistic_sharpe_ratio(
        self, returns: pd.Series, sr_benchmark: float = 0.0
    ) -> float:
        """
        PSR(SR*): probabilidad de que el Sharpe real supere `sr_benchmark`,
        ajustada por asimetría, curtosis y longitud del track.
        `sr_benchmark` debe estar en unidades de Sharpe PERIÓDICO.
        """
        r = self._clean(returns)
        n = len(r)
        if n < 3:
            return float("nan")
        sr = self.sharpe_periodic(r)
        if not np.isfinite(sr):
            return float("nan")
        gamma3 = float(r.skew())
        gamma4 = float(r.kurt()) + 3.0            # pandas.kurt = exceso → no-excesiva
        denom = 1.0 - gamma3 * sr + ((gamma4 - 1.0) / 4.0) * sr ** 2
        if denom <= 0:
            return float("nan")
        z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
        return float(_NORM.cdf(z))

    # ── DEFLATED SHARPE RATIO ──────────────────────────────────────────────────

    @staticmethod
    def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
        """
        E[máx Sharpe] esperado bajo la nula con `n_trials` ensayos independientes:
            SR₀ = √V·[(1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(N·e))]
        Devuelve 0.0 si N<2 (sin deflación por múltiples pruebas).
        """
        if n_trials < 2 or sr_variance <= 0:
            return 0.0
        z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
        z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * _E))
        return float(np.sqrt(sr_variance) * (
            (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
        ))

    def deflated_sharpe_ratio(
        self, returns: pd.Series, sr_variance: float, n_trials: int
    ) -> float:
        """
        DSR = PSR(SR₀), con SR₀ = E[máx Sharpe] sobre N ensayos.
        `sr_variance` = varianza de los Sharpe PERIÓDICOS de los N ensayos.
        """
        sr0 = self.expected_max_sharpe(sr_variance, n_trials)
        return self.probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)

    # ── EVALUACIÓN COMPLETA ────────────────────────────────────────────────────

    def evaluate(
        self, returns: pd.Series, sr_variance: float, n_trials: int
    ) -> SignificanceReport:
        """Bundle de métricas de significancia para un track OOS."""
        r = self._clean(returns)
        return SignificanceReport(
            n_obs=len(r),
            sharpe_periodic=self.sharpe_periodic(r),
            sharpe_annual=self.sharpe_annual(r),
            skew=float(r.skew()) if len(r) > 2 else float("nan"),
            kurtosis=float(r.kurt()) + 3.0 if len(r) > 3 else float("nan"),
            psr_zero=self.probabilistic_sharpe_ratio(r, 0.0),
            sr0_deflation=self.expected_max_sharpe(sr_variance, n_trials),
            dsr=self.deflated_sharpe_ratio(r, sr_variance, n_trials),
            n_trials=n_trials,
        )

    def evaluate_from_trials(
        self, trials: pd.DataFrame, target: str
    ) -> SignificanceReport:
        """
        Conveniencia: deriva `sr_variance` y `n_trials` de una matriz de ensayos
        (columnas = estrategias, filas = retornos) y evalúa la estrategia `target`.
        """
        sharpes = trials.apply(self.sharpe_periodic, axis=0).dropna()
        sr_var = float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0
        return self.evaluate(trials[target], sr_var, n_trials=trials.shape[1])

    # ── PROBABILITY OF BACKTEST OVERFITTING (CSCV) ─────────────────────────────

    def probability_of_backtest_overfitting(
        self, trials: pd.DataFrame, n_partitions: int = 16
    ) -> float:
        """
        PBO vía Combinatorially Symmetric Cross-Validation (Bailey et al., 2017).

        Particiona el track en `n_partitions` bloques disjuntos; para cada
        combinación de la mitad como IS, identifica la estrategia óptima IS y mide
        su rango relativo OOS. PBO = P(la mejor IS cae por debajo de la mediana OOS).

        El bucle recorre combinaciones de particiones (algorítmicamente necesario,
        no es iteración fila-a-fila); el cálculo de Sharpe por bloque es vectorizado.

        Args:
            trials:        DataFrame (filas=tiempo, columnas=estrategias).
            n_partitions:  Nº de bloques S (par). Combinaciones evaluadas = C(S, S/2).

        Returns:
            PBO ∈ [0, 1]. Valores altos (>0.5) ⇒ el desempeño IS no se sostiene OOS.
        """
        if n_partitions % 2 != 0:
            raise ValueError("n_partitions debe ser par para CSCV.")
        clean = trials.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        n_cols = clean.shape[1]
        if n_cols < 2 or len(clean) < n_partitions * 2:
            return float("nan")

        # Particionado posicional (np.array_split sobre un DataFrame devuelve
        # ndarrays; partimos los índices enteros y reindexamos con .iloc).
        pos_blocks = np.array_split(np.arange(len(clean)), n_partitions)
        blocks: List[pd.DataFrame] = [
            clean.iloc[b] for b in pos_blocks if len(b) > 1
        ]
        s = len(blocks)
        all_idx = range(s)
        logits: List[float] = []

        for is_idx in combinations(all_idx, s // 2):
            oos_idx = [i for i in all_idx if i not in is_idx]
            is_data = pd.concat([blocks[i] for i in is_idx])
            oos_data = pd.concat([blocks[i] for i in oos_idx])

            is_sharpe = is_data.apply(self.sharpe_periodic, axis=0)
            oos_sharpe = oos_data.apply(self.sharpe_periodic, axis=0)
            if is_sharpe.isna().all() or oos_sharpe.isna().all():
                continue

            best = is_sharpe.idxmax()
            # Rango relativo OOS de la mejor IS (omega ∈ (0,1))
            oos_rank = oos_sharpe.rank().loc[best]
            omega = oos_rank / (n_cols + 1)
            omega = min(max(omega, 1e-6), 1 - 1e-6)
            logits.append(float(np.log(omega / (1.0 - omega))))

        if not logits:
            return float("nan")
        arr = np.asarray(logits)
        return float((arr <= 0).mean())   # fracción con la IS-óptima bajo mediana OOS
