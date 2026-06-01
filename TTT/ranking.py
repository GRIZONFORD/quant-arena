"""
ranking_minimal.py — Reemplazo quirúrgico de ranking.py para el stack TTT v3.0
================================================================================
PROPÓSITO:
    El engine walk_forward_engine con TTT.py importa exactamente DOS símbolos
    de ranking.py:

        from ranking import RankingManager, BayesianELORanking

    El ranking.py original define 4 clases (ELORanking, TrueSkillRanking,
    BradleyTerryRanking, BayesianELORanking) y tiene una dependencia dura sobre
    el paquete 'trueskill' (Microsoft, 2006) — distinto de 'trueskillthroughtime'
    (Landfried, 2024) — que causa NameError en tiempo de carga de clase:

        class TrueSkillRanking(RankingSystem):
            def _update(self, ratings_dict: Dict[str, trueskill.Rating]):  # ← falla
                                                          ^^^^^^^^^^^^^^^
        NameError: name 'trueskill' is not defined

    Este archivo reemplaza ranking.py exponiendo ÚNICAMENTE los dos símbolos
    que el motor TTT usa, sin ninguna dependencia externa:

        ✅ BayesianELORanking  — IS fitness ranking (Calmar/Sharpe por pares)
        ✅ RankingManager      — orquestador multi-régimen/multi-métrica

    Las clases eliminadas (ELORanking, TrueSkillRanking, BradleyTerryRanking)
    no son llamadas en ninguna ruta de ejecución del engine TTT.

USO:
    Sube este archivo a /content/ con el nombre "ranking.py"
    (reemplaza el original). El engine lo importa sin cambio alguno.

REFERENCIA ARQUITECTURAL:
    - El ranking ELO de este archivo opera SOLO sobre el período IS (train).
      Su función es seleccionar qué estrategias van al Top-3 / Bottom-3
      para el split OOS. Luego, TTT realiza la inferencia global sobre
      TODOS los folds OOS para construir μ(t) y σ(t).
    - Ver: WalkForwardELOEngine._train_elo_ranking() en el engine TTT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# RATING DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rating:
    """Contenedor de rating con metadatos de partidos."""
    value: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    last_updated: Optional[pd.Timestamp] = None

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def __repr__(self) -> str:
        return (f"Rating(value={self.value:.1f}, "
                f"games={self.games_played}, "
                f"wr={self.win_rate:.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# BASE ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class RankingSystem(ABC):
    """
    Clase base para sistemas de ranking.
    Mantiene historial de ratings para análisis posterior.
    """

    def __init__(self) -> None:
        # rating_history[strategy] = [(timestamp, rating_value), ...]
        self.rating_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = (
            defaultdict(list)
        )

    @abstractmethod
    def update_ratings(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        """
        Actualiza ratings tras un enfrentamiento.

        Args:
            outcome: 1.0 si A gana, 0.0 si B gana, 0.5 si empate.
        """

    @abstractmethod
    def get_rating(
        self,
        strategy_name: str,
        regime: Optional[str] = None,
    ) -> float:
        """Retorna el rating escalar actual de una estrategia."""

    @abstractmethod
    def get_all_ratings(
        self,
        regime: Optional[str] = None,
    ) -> Dict[str, Rating]:
        """Retorna todos los ratings como objetos Rating."""

    def get_leaderboard(
        self,
        regime: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Tabla ordenada de mayor a menor rating.
        Columna 'strategy' requerida por _train_elo_ranking() del engine.
        """
        ratings = self.get_all_ratings(regime)
        rows = [
            {
                "strategy":  name,
                "rating":    r.value,
                "games":     r.games_played,
                "wins":      r.wins,
                "losses":    r.losses,
                "ties":      r.ties,
                "win_rate":  r.win_rate,
            }
            for name, r in ratings.items()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = (df.sort_values("rating", ascending=False)
                    .reset_index(drop=True))
            df.index = df.index + 1      # ranking 1-based
            if top_n:
                df = df.head(top_n)
        return df

    def get_rating_history(self, strategy_name: str) -> pd.Series:
        """Serie temporal de ratings (para MetaELOAnalyzer)."""
        history = self.rating_history.get(strategy_name, [])
        if not history:
            return pd.Series(dtype=float)
        timestamps, values = zip(*history)
        return pd.Series(list(values), index=list(timestamps))


# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN ELO RANKING  (único sistema usado por el engine TTT)
# ══════════════════════════════════════════════════════════════════════════════

class BayesianELORanking(RankingSystem):
    """
    ELO con seguimiento Bayesiano de incertidumbre (μ, σ) por estrategia.

    Diferencias respecto al ELO clásico:
    ─────────────────────────────────────────────────────────────────────────
    1. K-factor adaptativo: K ∝ σ → estrategias con mayor incertidumbre
       aprenden más rápido (prior débil → actualización agresiva).
    2. σ decrece con cada partido (más información → más certeza), pero
       un término τ (drift) previene colapso total de la incertidumbre.
    3. Rating conservador: μ - k·σ penaliza la incertidumbre para
       decisiones risk-averse (selección Top-3 en modo conservador).

    Rol en el pipeline TTT:
    ─────────────────────────────────────────────────────────────────────────
    Oprea sobre datos IS (train). Su leaderboard final selecciona el
    Top-3 / Bottom-3 para medir el spread OOS. El ranking GLOBAL de
    habilidad se delega a TTT (walk_forward_engine TTT._ttt.fit_ttt()).
    """

    def __init__(
        self,
        initial_mu: float = 1500.0,
        initial_sigma: float = 350.0,
        min_sigma: float = 50.0,
        tau: float = 1.0,
        base_k: float = 32.0,
    ) -> None:
        """
        Args:
            initial_mu:    Rating medio inicial.
            initial_sigma: Incertidumbre inicial (alto → prior no informativo).
            min_sigma:     Suelo de σ — los ratings nunca alcanzan certeza total.
            tau:           Factor de drift — σ nunca colapsa a 0.
            base_k:        K-factor base (se escala por incertidumbre relativa).
        """
        super().__init__()
        self.initial_mu    = initial_mu
        self.initial_sigma = initial_sigma
        self.min_sigma     = min_sigma
        self.tau           = tau
        self.base_k        = base_k

        # ratings[strategy] = (mu, sigma)
        self.ratings: Dict[str, Tuple[float, float]] = defaultdict(
            lambda: (initial_mu, initial_sigma)
        )
        self.game_counts: Dict[str, int] = defaultdict(int)

        # Ratings por régimen de mercado
        self.regime_ratings: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(
            lambda: defaultdict(lambda: (initial_mu, initial_sigma))
        )
        self.regime_game_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ── Actualización pública ─────────────────────────────────────────────────

    def update_ratings(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        ts = timestamp or pd.Timestamp.now()

        # 1. Ratings globales
        self._bayesian_elo_update(
            strategy_a_name, strategy_b_name, outcome,
            self.ratings, self.game_counts, ts,
        )
        # 2. Ratings por régimen (si aplica)
        if regime is not None:
            self._bayesian_elo_update(
                strategy_a_name, strategy_b_name, outcome,
                self.regime_ratings[regime],
                self.regime_game_counts[regime],
                ts,
            )

    # ── Núcleo matemático ─────────────────────────────────────────────────────

    def _bayesian_elo_update(
        self,
        name_a: str,
        name_b: str,
        outcome: float,
        ratings_dict: Dict[str, Tuple[float, float]],
        counts_dict: Dict[str, int],
        timestamp: pd.Timestamp,
    ) -> None:
        """
        Actualización Bayesiana de ELO con incertidumbre adaptativa.

        Ecuaciones:
            E_A = 1 / (1 + 10^((μ_B - μ_A) / 400))      [ELO estándar]
            K   = base_k * (0.5 + 1.5 * avg_uncertainty)  [K adaptativo]
            μ_A' = μ_A + K * (outcome - E_A)
            σ_A' = max(min_sigma, sqrt(σ_A² / (1 + σ_A²/σ_0²)) + τ)
        """
        mu_a, sigma_a = ratings_dict[name_a]
        mu_b, sigma_b = ratings_dict[name_b]

        # Probabilidad esperada de victoria de A (fórmula ELO estándar)
        expected_a = 1.0 / (1.0 + 10.0 ** ((mu_b - mu_a) / 400.0))

        # K-factor adaptativo: mayor incertidumbre → aprendizaje más rápido
        uf_a = sigma_a / self.initial_sigma
        uf_b = sigma_b / self.initial_sigma
        k_adapted = self.base_k * (0.5 + 1.5 * (uf_a + uf_b) / 2.0)

        # Actualización de μ
        error  = outcome - expected_a
        mu_a_new = mu_a + k_adapted * error
        mu_b_new = mu_b - k_adapted * error

        # Actualización de σ (decae con información, drift τ previene colapso)
        def _update_sigma(sigma: float) -> float:
            shrunk = np.sqrt(sigma ** 2 / (1.0 + sigma ** 2 / self.initial_sigma ** 2))
            return max(self.min_sigma, shrunk + self.tau)

        sigma_a_new = _update_sigma(sigma_a)
        sigma_b_new = _update_sigma(sigma_b)

        # Persistir
        ratings_dict[name_a] = (mu_a_new, sigma_a_new)
        ratings_dict[name_b] = (mu_b_new, sigma_b_new)
        counts_dict[name_a]  = counts_dict.get(name_a, 0) + 1
        counts_dict[name_b]  = counts_dict.get(name_b, 0) + 1

        # Historial (μ como señal de rating)
        self.rating_history[name_a].append((timestamp, mu_a_new))
        self.rating_history[name_b].append((timestamp, mu_b_new))

    # ── Consultas ─────────────────────────────────────────────────────────────

    def get_rating(
        self,
        strategy_name: str,
        regime: Optional[str] = None,
        conservative: bool = False,
        conservative_factor: float = 2.0,
    ) -> float:
        """
        Retorna μ (o μ - k·σ en modo conservador).

        El modo conservador penaliza estrategias con alta incertidumbre,
        evitando que estrategias 'afortunadas' con pocos partidos dominen
        el leaderboard IS.
        """
        if regime is None:
            mu, sigma = self.ratings[strategy_name]
        else:
            mu, sigma = self.regime_ratings[regime][strategy_name]
        return (mu - conservative_factor * sigma) if conservative else mu

    def get_uncertainty(
        self,
        strategy_name: str,
        regime: Optional[str] = None,
    ) -> float:
        """Retorna σ (incertidumbre sobre el rating)."""
        if regime is None:
            _, sigma = self.ratings[strategy_name]
        else:
            _, sigma = self.regime_ratings[regime][strategy_name]
        return sigma

    def get_signal_to_noise(
        self,
        strategy_name: str,
        regime: Optional[str] = None,
    ) -> float:
        """SNR = μ/σ — análogo al Sharpe en espacio de habilidad ELO."""
        if regime is None:
            mu, sigma = self.ratings[strategy_name]
        else:
            mu, sigma = self.regime_ratings[regime][strategy_name]
        return mu / sigma if sigma > 1e-9 else 0.0

    def get_all_ratings(
        self,
        regime: Optional[str] = None,
    ) -> Dict[str, Rating]:
        """Retorna todos los ratings como objetos Rating (requerido por base)."""
        rd = self.ratings if regime is None else self.regime_ratings[regime]
        cd = (self.game_counts if regime is None
              else self.regime_game_counts[regime])
        return {
            name: Rating(value=mu, games_played=cd.get(name, 0))
            for name, (mu, _) in rd.items()
        }

    def get_leaderboard_with_uncertainty(
        self,
        regime: Optional[str] = None,
        top_n: Optional[int] = None,
        sort_by: str = "mu",
    ) -> pd.DataFrame:
        """
        Leaderboard enriquecido con μ, σ, rating conservador y SNR.

        Args:
            sort_by: 'mu' | 'conservative' | 'signal_to_noise'
        """
        rd = self.ratings if regime is None else self.regime_ratings[regime]
        cd = (self.game_counts if regime is None
              else self.regime_game_counts[regime])

        rows = []
        for name, (mu, sigma) in rd.items():
            rows.append({
                "strategy":           name,
                "mu":                 mu,
                "sigma":              sigma,
                "conservative_rating": mu - 2.0 * sigma,
                "signal_to_noise":    mu / sigma if sigma > 1e-9 else 0.0,
                "games":              cd.get(name, 0),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = (df.sort_values(sort_by, ascending=False)
                    .reset_index(drop=True))
            df.index = df.index + 1
            if top_n:
                df = df.head(top_n)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# RANKING MANAGER  (orquestador multi-métrica / multi-régimen)
# ══════════════════════════════════════════════════════════════════════════════

class RankingManager:
    """
    Gestiona un sistema de ranking a través de múltiples métricas y regímenes.

    Interfaz principal que consume WalkForwardELOEngine._train_elo_ranking():
        rm = RankingManager(ranking_class=BayesianELORanking, **kwargs)
        rm.update(s_a, s_b, outcome, metric='alpha', regime=r, timestamp=t)
        leaderboard = rm.get_leaderboard('alpha', None)   # → top_3 / bottom_3
    """

    def __init__(
        self,
        ranking_class: type = BayesianELORanking,
        **ranking_kwargs,
    ) -> None:
        self.ranking_class  = ranking_class
        self.ranking_kwargs = ranking_kwargs

        # rankings[metric][regime] = instancia de RankingSystem
        self.rankings: Dict[str, Dict[Optional[str], RankingSystem]] = (
            defaultdict(dict)
        )
        self.metrics: set = set()
        self.regimes: set = set()

    def update(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        metric: str = "sharpe",
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        """Actualiza el ranking para una métrica y régimen específicos."""
        if regime not in self.rankings[metric]:
            self.rankings[metric][regime] = self.ranking_class(
                **self.ranking_kwargs
            )
        self.metrics.add(metric)
        if regime is not None:
            self.regimes.add(regime)

        self.rankings[metric][regime].update_ratings(
            strategy_a_name, strategy_b_name, outcome, regime, timestamp
        )

    def get_leaderboard(
        self,
        metric: str = "sharpe",
        regime: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Leaderboard ordenado para una métrica y régimen.
        Columna 'strategy' requerida por el engine TTT.
        """
        if metric not in self.rankings or regime not in self.rankings[metric]:
            return pd.DataFrame(columns=["strategy", "rating", "games",
                                         "wins", "losses", "ties", "win_rate"])
        return self.rankings[metric][regime].get_leaderboard(regime, top_n)

    def get_all_leaderboards(
        self,
    ) -> Dict[str, Dict[Optional[str], pd.DataFrame]]:
        """Todos los leaderboards indexados por (métrica, régimen)."""
        result: Dict[str, Dict[Optional[str], pd.DataFrame]] = {}
        for metric in self.metrics:
            result[metric] = {}
            for regime in [None] + list(self.regimes):
                if regime in self.rankings[metric]:
                    result[metric][regime] = self.get_leaderboard(
                        metric, regime
                    )
        return result

    def get_rating_history(
        self,
        strategy_name: str,
        metric: str = "sharpe",
        regime: Optional[str] = None,
    ) -> pd.Series:
        """Historial de ratings (para MetaELOAnalyzer)."""
        if metric not in self.rankings or regime not in self.rankings[metric]:
            return pd.Series(dtype=float)
        return self.rankings[metric][regime].get_rating_history(strategy_name)
