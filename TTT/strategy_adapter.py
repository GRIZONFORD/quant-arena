# strategy_adapter.py
"""
Strategy Adapter v3.0 — Puente Strategy Zoo ↔ Motor TTT
=========================================================
El adaptador es la capa de transformación entre las señales crudas del zoo
y el formato que consume el motor Walk-Forward TTT. Sus responsabilidades son:

1. APPLY TRANSACTION COSTS:
   Slippage + comisiones aplicados ANTES de registrar el resultado en TTT.
   La "habilidad" que TTT infiere debe ser NETA de costos reales — de lo
   contrario el ranking favorecería estrategias de alto turnover con alpha
   bruto alto pero alpha neto negativo.

2. SIGNAL DECAY:
   Penalizar señales que llevan mucho tiempo sin renovarse. Una estrategia
   que "se duerme" durante N períodos sin generar señal fresca recibe un
   factor de descuento exponencial sobre su resultado en ese fold.
   factor = exp(-ln(2) * age / halflife)

3. HEALTH-BASED PRIOR PARA TTT:
   Los health_metrics (Sharpe IS, Hit Ratio) de cada estrategia se convierten
   en un prior informativo para TTT: estrategias con Sharpe IS alto inician
   con μ₀ > 0 (creencia a priori de competencia) y σ₀ reducida (prior firme).
   Esto acorta el burn-in bayesiano y produce rankings estables antes.

4. BACKTESTING:
   Dado un DataFrame de datos de mercado y una estrategia, calcula:
   - Retornos diarios netos de costos.
   - Retorno compuesto del fold (para TTT add_fold).
   - Métricas de performance agregadas por fold.

5. TTT FORMAT:
   backtest_to_ttt_format() traduce los retornos de N estrategias en las
   listas composition/results/times/obs que consume TTTAccumulator.add_fold().

COMPATIBILIDAD con walk_forward_engine_con_TTT.py:
  · adapt_strategies(strategies, capital) → Dict[name, AdaptedStrategy]
  · AdaptedStrategy tiene .generate_positions(data) para BacktestEngine
  · StrategyAdapter tiene .get_ttt_prior(health_metrics) para TTTAccumulator

Referencias:
  López de Prado, M. (2018). Advances in Financial Machine Learning.
    Cap. 3 (Meta-Labeling), Cap. 14 (Transaction Costs).
  Landfried, G. & Mocskos, E. (2024). TrueSkill Through Time.
  Almgren, R. & Chriss, N. (2001). Optimal Execution.
    Journal of Risk, 3(2), 5–39.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_zoo import BaseStrategy

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES DE RESULTADO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """
    Resultado de backtesting de UNA estrategia en UN fold.

    Usado internamente por el adaptador y consumido por TTTAccumulator.add_fold().
    """
    strategy_name:    str
    fold_start:       pd.Timestamp
    fold_end:         pd.Timestamp

    # Retornos netos de costos — vector diario
    net_returns:      pd.Series = field(default_factory=pd.Series)

    # Métricas agregadas del fold
    fold_return:      float = 0.0        # retorno compuesto neto del fold
    sharpe_fold:      float = float("nan")
    max_drawdown:     float = 0.0
    n_trades:         int   = 0

    # Costos aplicados
    total_costs_pct:  float = 0.0        # costo total como % del capital

    # Decay factor aplicado
    decay_factor:     float = 1.0        # ∈ [0, 1]; 1 = sin penalización

    # Prior derivado de health_metrics IS
    prior_mu:         float = 0.0        # prior de habilidad para TTT
    prior_sigma:      float = 1.0        # incertidumbre del prior


@dataclass
class TransactionCostModel:
    """
    Modelo de costos de transacción para aplicar antes de TTT.

    COMPONENTES:
    · commission_pct: Comisión fija del broker como fracción del valor operado.
      Valor institucional típico: 0.01–0.05% por lado.
    · slippage_pct:   Deslizamiento esperado al ejecutar la orden.
      Para ETFs líquidos (SPY): ~0.01–0.03%. Para acciones menos líquidas: >0.1%.
    · spread_half_pct: Mitad del bid-ask spread. Costo implícito de liquidez.

    TOTAL COST PER TRADE (one-way):
      cost = commission_pct + slippage_pct + spread_half_pct

    El adaptador multiplica este costo por el número de trades (cambios de posición)
    y lo deduce de los retornos antes de registrar el fold en TTT.

    Ref: Almgren & Chriss (2001), Almgren et al. (2005).
    """
    commission_pct:   float = 0.0005   # 5 bps = 0.05% (institucional)
    slippage_pct:     float = 0.0010   # 10 bps (líquido; ajustar para ilíquidos)
    spread_half_pct:  float = 0.0002   # 2 bps (bid-ask en ETFs)
    market_impact_pct: float = 0.0003  # 3 bps (impacto de mercado)

    @property
    def one_way_cost(self) -> float:
        """Costo total por lado (one-way) como fracción del valor negociado."""
        return (self.commission_pct + self.slippage_pct
                + self.spread_half_pct + self.market_impact_pct)

    @property
    def round_trip_cost(self) -> float:
        """Costo total de ida y vuelta."""
        return 2.0 * self.one_way_cost

    def compute_daily_cost(
        self,
        positions: pd.Series,
        capital: float = 100_000,
    ) -> pd.Series:
        """
        Calcula el costo de transacción diario sobre cambios de posición.

        El costo se incurre SOLO cuando la posición cambia (turnover):
          turnover_t = |position_t - position_{t-1}| / capital
          cost_t     = turnover_t * one_way_cost

        Args:
            positions: Serie de posiciones en $ (diarias).
            capital:   Capital de referencia para normalizar.

        Returns:
            pd.Series de costos como fracción del capital (≥ 0).
        """
        # Cambio absoluto de posición normalizado por capital
        position_change = positions.diff().abs()
        turnover_frac   = (position_change / max(capital, 1.0)).fillna(0.0)

        # Costo diario = turnover × costo one-way por unidad de valor operado
        daily_cost = turnover_frac * self.one_way_cost
        return daily_cost


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTED STRATEGY — Wrapper del zoo para el BacktestEngine
# ══════════════════════════════════════════════════════════════════════════════

class AdaptedStrategy:
    """
    Wrapper que adapta una BaseStrategy al contrato del BacktestEngine.

    El BacktestEngine del motor llama a:
      adapted_strategy.generate_positions(data, capital)

    Este wrapper:
    1. Llama a strategy.generate_signals(data, capital).
    2. Extrae 'positions' del dict de señales.
    3. Aplica el factor de decay a las posiciones.

    El adaptador también expone:
    · .name           → identificador único para el motor.
    · .strategy_type  → tipo para el RankingManager.
    · .health_metrics → dict calculado en IS para prior TTT.
    · .signal_halflife → vida media de la señal (para el adaptador).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        cost_model: TransactionCostModel,
        apply_decay: bool = True,
    ) -> None:
        self._strategy    = strategy
        self.cost_model   = cost_model
        self.apply_decay  = apply_decay

        # Forwarding de atributos del zoo
        self.name          = strategy.name
        self.strategy_type = strategy.strategy_type
        self.signal_halflife = strategy.signal_halflife

        # Calculados en IS por StrategyAdapter.compute_health_priors()
        self.health_metrics: Dict[str, float] = {}
        self.prior_mu:       float = 0.0
        self.prior_sigma:    float = 1.0

    def generate_positions(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> pd.Series:
        """
        Interfaz pública compatible con BacktestEngine.

        Genera posiciones en $ aplicando:
        1. Señales del zoo (positions).
        2. Factor de decay exponencial si apply_decay=True.
        """
        signals = self._strategy.generate_signals(data, capital)
        positions = signals.get("positions", pd.Series(0.0, index=data.index))

        if self.apply_decay:
            age     = signals.get("signal_age",
                                  pd.Series(0, index=data.index, dtype=float))
            halflife = max(self.signal_halflife, 1)
            decay    = np.exp(-np.log(2) * age.astype(float) / halflife)
            positions = positions * decay

        return positions

    def get_raw_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        """Devuelve el dict completo de señales (sin decay) para diagnóstico."""
        return self._strategy.generate_signals(data, capital)

    def warmup_period(self) -> int:
        return self._strategy.warmup_period()

    def __repr__(self) -> str:
        return f"AdaptedStrategy(name={self.name}, halflife={self.signal_halflife})"


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ADAPTER — Orquestador principal
# ══════════════════════════════════════════════════════════════════════════════

class StrategyAdapter:
    """
    Orquestador que conecta el Strategy Zoo con el motor Walk-Forward TTT.

    RESPONSABILIDADES:
    ──────────────────────────────────────────────────────────────────────────
    1. backtest_strategy(strategy, data, capital) → FoldResult
       Ejecuta el backtest neto de costos sobre un período y devuelve
       el FoldResult que contiene retornos, métricas y prior TTT.

    2. backtest_to_ttt_format(fold_results, benchmark_returns) → TTT lists
       Traduce FoldResults a (composition, results, times, obs) para
       TTTAccumulator.add_fold().

    3. compute_health_priors(strategies, is_data, capital) → Dict
       Calcula health_metrics en IS y deriva los priors μ₀/σ₀ para TTT.

    4. apply_signal_decay(positions, signal_age, halflife) → pd.Series
       Factor de descuento exponencial sobre señales viejas.

    5. compute_fold_return(net_returns, decay_factor) → float
       Retorno compuesto neto con ajuste de decay para TTT add_fold().

    FILOSOFÍA DE COSTOS:
    ──────────────────────────────────────────────────────────────────────────
    El alpha NETO de costos es la única medida honesta de habilidad en
    estrategias de alta frecuencia de señal. Un sistema con Sharpe bruto=1.5
    y turnover de 50 operaciones anuales puede tener Sharpe neto < 0.3 si
    los costos son de 0.1% por lado.

    La ecuación de costo por estrategia:
      net_ret_t = gross_ret_t - daily_cost_t
    donde daily_cost_t = |ΔPosition_t| / capital × one_way_cost_rate

    Ref: Almgren & Chriss (2001); Kissell & Glantz (2003); LdP (2018), Cap. 14.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        apply_decay: bool = True,
        min_decay_factor: float = 0.10,
    ) -> None:
        """
        Args:
            cost_model:        Modelo de costos. Default: institucional estándar.
            apply_decay:       Si True, penaliza señales viejas.
            min_decay_factor:  Cota inferior del factor de decay (evita 0 absoluto).
        """
        self.cost_model        = cost_model or TransactionCostModel()
        self.apply_decay       = apply_decay
        self.min_decay_factor  = min_decay_factor

    # ──────────────────────────────────────────────────────────────────────────
    # BACKTEST DE UNA ESTRATEGIA EN UN FOLD
    # ──────────────────────────────────────────────────────────────────────────

    def backtest_strategy(
        self,
        strategy: AdaptedStrategy,
        data: pd.DataFrame,
        capital: float = 100_000,
    ) -> FoldResult:
        """
        Ejecuta el backtest neto de costos de una estrategia sobre un DataFrame.

        PIPELINE:
        ─────────────────────────────────────────────────────────────────────
        1. Generar señales crudas (zoo).
        2. Extraer positions, side, size, signal_age.
        3. Calcular retorno bruto diario:
             gross_ret_t = position_{t-1} / capital × price_ret_t
           (shift(1) obligatorio: se negocia al cierre de t para operar en t+1).
        4. Calcular costos diarios de transacción.
        5. Aplicar decay factor al retorno bruto:
             decayed_ret_t = gross_ret_t × decay_t
        6. Retorno neto:
             net_ret_t = decayed_ret_t - cost_t
        7. Calcular métricas del fold.

        Args:
            strategy: AdaptedStrategy wrapeando una BaseStrategy del zoo.
            data:     DataFrame OHLCV con DatetimeIndex para el período del fold.
            capital:  Capital de referencia en $.

        Returns:
            FoldResult con retornos netos y métricas agregadas.
        """
        if data.empty or "close" not in data.columns:
            logger.warning(f"{strategy.name}: datos vacíos o sin 'close' — FoldResult vacío.")
            return self._empty_fold_result(strategy.name, data)

        # ── 1. Señales brutas ─────────────────────────────────────────────
        try:
            raw = strategy.get_raw_signals(data, capital)
        except Exception as exc:
            logger.error(f"{strategy.name}.get_raw_signals: {exc}")
            return self._empty_fold_result(strategy.name, data)

        positions  = raw.get("positions",  pd.Series(0.0, index=data.index))
        signal_age = raw.get("signal_age", pd.Series(0,   index=data.index))

        # ── 2. Decay factor ───────────────────────────────────────────────
        if self.apply_decay:
            halflife     = max(strategy.signal_halflife, 1)
            decay_factor = np.exp(
                -np.log(2) * signal_age.astype(float) / halflife
            ).clip(self.min_decay_factor, 1.0)
        else:
            decay_factor = pd.Series(1.0, index=data.index)

        # ── 3. Retorno bruto diario ───────────────────────────────────────
        price_ret  = data["close"].astype(float).pct_change()
        weight     = (positions.shift(1) / max(capital, 1.0)).fillna(0.0)
        gross_ret  = (weight * price_ret).fillna(0.0)

        # ── 4. Aplicar decay al retorno bruto ────────────────────────────
        decayed_ret = gross_ret * decay_factor

        # ── 5. Costos de transacción ──────────────────────────────────────
        daily_cost = self.cost_model.compute_daily_cost(positions, capital)

        # ── 6. Retorno neto ───────────────────────────────────────────────
        net_ret = decayed_ret - daily_cost

        # ── 7. Métricas del fold ──────────────────────────────────────────
        fold_return = float((1.0 + net_ret).prod() - 1.0)
        sharpe_fold = self._compute_fold_sharpe(net_ret)
        max_dd      = self._compute_max_drawdown(net_ret)
        n_trades    = int((positions.diff().abs() > capital * 0.01).sum())
        total_cost  = float(daily_cost.sum())

        # Decay escalar representativo del fold (media ponderada por |posición|)
        pos_weight  = positions.abs() / (positions.abs().sum() + 1e-8)
        decay_scalar = float((decay_factor * pos_weight).sum())

        fold_start = data.index[0] if len(data) > 0 else pd.Timestamp("NaT")
        fold_end   = data.index[-1] if len(data) > 0 else pd.Timestamp("NaT")

        return FoldResult(
            strategy_name   = strategy.name,
            fold_start      = fold_start,
            fold_end        = fold_end,
            net_returns     = net_ret,
            fold_return     = fold_return,
            sharpe_fold     = sharpe_fold,
            max_drawdown    = max_dd,
            n_trades        = n_trades,
            total_costs_pct = total_cost,
            decay_factor    = decay_scalar,
            prior_mu        = strategy.prior_mu,
            prior_sigma     = strategy.prior_sigma,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CONVERSIÓN A FORMATO TTT
    # ──────────────────────────────────────────────────────────────────────────

    def backtest_to_ttt_format(
        self,
        fold_results: Dict[str, FoldResult],
        benchmark_net_returns: pd.Series,
    ) -> Tuple[List, List, List, List]:
        """
        Convierte FoldResults a las listas (composition, results, times, obs)
        que consume TTTAccumulator directamente.

        MAPEO TTT (Landfried & Mocskos, 2024, logistic.py):
        ─────────────────────────────────────────────────────────────────────
        Por cada estrategia en fold_results:
          composition[i] = [["strategy_name"], ["_benchmark_"]]
          results[i]     = [strat_fold_ret, bench_fold_ret]
                           Retornos compuestos netos del fold.
                           TTT infiere habilidad de la MAGNITUD del diferencial.
          obs[i]         = "Continuous" — modelo de observación continua.
          times[i]       = timestamp UNIX en días del primer día OOS.

        El retorno compuesto del fold es:
          r_fold = prod(1 + r_t) - 1, ∀t ∈ fold
        Este es el score que TTT usa para el partido estrategia vs benchmark.

        Args:
            fold_results:          Dict[strategy_name → FoldResult].
            benchmark_net_returns: Retornos diarios netos del benchmark.

        Returns:
            Tuple (composition, results, times, obs) listas en formato TTT.
        """
        composition: List = []
        results:     List = []
        times:       List = []
        obs:         List = []

        # Retorno compuesto del benchmark para este fold
        bench_ret = float((1.0 + benchmark_net_returns).prod() - 1.0)

        for name, fr in fold_results.items():
            if fr.net_returns.empty:
                logger.debug(
                    f"backtest_to_ttt_format: '{name}' sin retornos — saltando."
                )
                continue

            # Retorno compuesto neto con decay ya aplicado (incluido en FoldResult)
            strat_ret = fr.fold_return

            # Tiempo en días desde epoch (formato logistic.py línea 19)
            t_days = float(fr.fold_start.timestamp() / (60 * 60 * 24))

            composition.append([[name], ["_benchmark_"]])
            results.append([strat_ret, bench_ret])
            times.append(t_days)
            obs.append("Continuous")

        logger.debug(
            f"backtest_to_ttt_format: {len(composition)} partidos generados | "
            f"bench_ret={bench_ret:.4f}"
        )
        return composition, results, times, obs

    # ──────────────────────────────────────────────────────────────────────────
    # PRIORS TTT BASADOS EN HEALTH METRICS IS
    # ──────────────────────────────────────────────────────────────────────────

    def compute_health_priors(
        self,
        adapted_strategies: Dict[str, AdaptedStrategy],
        is_data: pd.DataFrame,
        capital: float = 100_000,
        sharpe_to_mu_scale: float = 0.20,
        min_sigma: float = 0.30,
        max_sigma: float = 1.00,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula health_metrics IS y deriva priors (μ₀, σ₀) para TTT.

        PRIOR INFORMATIVO TTT:
        ─────────────────────────────────────────────────────────────────────
        Por defecto, TTT usa μ₀ = 0 para todas las estrategias (prior no
        informativo: creemos que todas tienen habilidad cero a priori).

        Con health priors, el prior se actualiza usando el Sharpe IS:
          prior_mu = clip(sharpe_is * sharpe_to_mu_scale, -1.0, 1.0)

        Y la incertidumbre del prior se reduce para estrategias con largo
        historial IS y hit ratio consistente:
          prior_sigma = max_sigma * (1 - clip(n_IS_years * hit_ratio_adj, 0, 1))
                        clipeada a [min_sigma, max_sigma]

        La lógica: si una estrategia tiene Sharpe IS=2.0, creemos a priori
        que su habilidad es μ₀ = 0.4 (positiva), y estamos más seguros de
        este prior (σ₀ reducida) si tiene 5+ años de historial IS.

        ADVERTENCIA: Prior demasiado informativo puede prevenir que TTT
        actualice correctamente ante evidence OOS. Se recomienda sharpe_to_mu_scale
        < 0.3 y min_sigma ≥ 0.3 para mantener el prior "suave".

        Args:
            adapted_strategies: Dict[name → AdaptedStrategy] del zoo.
            is_data:            DataFrame IS para calcular health_metrics.
            capital:            Capital base.
            sharpe_to_mu_scale: Factor de escalado Sharpe IS → μ₀.
            min_sigma:          σ₀ mínima (prior más informativo).
            max_sigma:          σ₀ máxima = prior no informativo.

        Returns:
            Dict[strategy_name → {'prior_mu', 'prior_sigma', **health_metrics}]
            Y actualiza adapted_strategy.prior_mu / .prior_sigma in-place.
        """
        priors: Dict[str, Dict[str, float]] = {}

        n_is_years = len(is_data) / 252.0 if not is_data.empty else 0.0

        for name, adapted in adapted_strategies.items():
            try:
                hm = adapted._strategy.get_health_metrics(is_data, capital)
            except Exception as exc:
                logger.warning(
                    f"compute_health_priors: '{name}' → error en "
                    f"get_health_metrics: {exc}"
                )
                hm = BaseStrategy._empty_health_metrics()

            # Derivar μ₀ desde Sharpe IS
            sharpe_is = hm.get("sharpe_is", 0.0)
            prior_mu  = float(np.clip(
                sharpe_is * sharpe_to_mu_scale,
                -1.0, 1.0
            ))

            # Derivar σ₀: reducir según evidencia IS (años × hit_ratio)
            hit_ratio    = hm.get("hit_ratio", 0.5)
            hit_adj      = max(0.0, hit_ratio - 0.5) * 2.0  # [0, 1]
            evidence_fac = np.clip(n_is_years * hit_adj, 0.0, 1.0)
            prior_sigma  = float(
                max_sigma - evidence_fac * (max_sigma - min_sigma)
            )
            prior_sigma  = float(np.clip(prior_sigma, min_sigma, max_sigma))

            # Actualizar el AdaptedStrategy in-place
            adapted.health_metrics = hm
            adapted.prior_mu       = prior_mu
            adapted.prior_sigma    = prior_sigma

            priors[name] = {
                "prior_mu":    prior_mu,
                "prior_sigma": prior_sigma,
                **hm
            }

            logger.debug(
                f"Prior '{name}': sharpe_is={sharpe_is:.3f} | "
                f"hit_ratio={hit_ratio:.3f} | "
                f"μ₀={prior_mu:.3f} | σ₀={prior_sigma:.3f}"
            )

        return priors

    # ──────────────────────────────────────────────────────────────────────────
    # RANKING COMPARATIVO POR FOLD
    # ──────────────────────────────────────────────────────────────────────────

    def rank_strategies_by_fold(
        self,
        fold_results: Dict[str, FoldResult],
        benchmark_net_returns: pd.Series,
        fitness: str = "calmar",
    ) -> pd.DataFrame:
        """
        Genera un ranking de estrategias para UN fold basado en métricas netas.

        Usado por el motor para construir el leaderboard IS y determinar
        top_3 / bottom_3 antes de evaluar en OOS.

        MÉTRICAS DE FITNESS:
        ─────────────────────────────────────────────────────────────────────
        · 'calmar'  : CAGR fold / |MaxDD fold| — penaliza tail risk.
        · 'sharpe'  : Sharpe anualizado del fold.
        · 'return'  : Retorno compuesto neto (sin ajuste por riesgo).
        · 'alpha'   : Retorno neto − retorno benchmark (exceso neto).

        Args:
            fold_results:          Dict[name → FoldResult].
            benchmark_net_returns: Retornos netos del benchmark en el fold.
            fitness:               Métrica de ranking ('calmar', 'sharpe', etc.).

        Returns:
            DataFrame ordenado de mejor a peor con columnas:
            [rank, strategy, fold_return, sharpe_fold, max_drawdown,
             n_trades, decay_factor, total_costs_pct, fitness_score]
        """
        bench_ret  = float((1.0 + benchmark_net_returns).prod() - 1.0)
        bench_ann  = float(benchmark_net_returns.mean() * 252)

        rows = []
        for name, fr in fold_results.items():
            if fr.net_returns.empty:
                continue

            # Calcular fitness score según métrica elegida
            if fitness == "calmar":
                fs = self._compute_fold_calmar(fr.net_returns)
            elif fitness == "sharpe":
                fs = fr.sharpe_fold
            elif fitness == "return":
                fs = fr.fold_return
            elif fitness == "alpha":
                fs = fr.fold_return - bench_ret
            else:
                logger.warning(f"fitness='{fitness}' desconocido — usando 'return'.")
                fs = fr.fold_return

            rows.append({
                "strategy":        name,
                "fold_return":     fr.fold_return,
                "sharpe_fold":     fr.sharpe_fold,
                "max_drawdown":    fr.max_drawdown,
                "n_trades":        fr.n_trades,
                "decay_factor":    fr.decay_factor,
                "total_costs_pct": fr.total_costs_pct,
                "alpha_vs_bench":  fr.fold_return - bench_ret,
                "fitness_score":   fs if np.isfinite(fs) else -np.inf,
            })

        if not rows:
            return pd.DataFrame()

        rank_df = (
            pd.DataFrame(rows)
            .sort_values("fitness_score", ascending=False)
            .reset_index(drop=True)
        )
        rank_df.insert(0, "rank", rank_df.index + 1)

        return rank_df

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL DECAY STANDALONE
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def apply_signal_decay(
        positions: pd.Series,
        signal_age: pd.Series,
        halflife: int = 20,
        min_factor: float = 0.10,
    ) -> pd.Series:
        """
        Aplica factor de decaimiento exponencial a posiciones según la edad de la señal.

        FÓRMULA:
          decay(t) = exp(-ln(2) × age(t) / halflife)

        Esto modela la degradación de la información contenida en la señal:
        una señal de hace `halflife` días conserva el 50% de su poder predictivo.

        Ejemplo:
          halflife=20, age=20 → decay=0.50  (50% del tamaño original)
          halflife=20, age=40 → decay=0.25  (25% del tamaño original)
          halflife=20, age=60 → decay=0.125 (12.5%, cota inferior=0.10)

        JUSTIFICACIÓN FINANCIERA:
        La información en los mercados financieros se "consume" conforme es
        absorbida por el mercado (efficient market learning). Una señal basada
        en datos de hace 40 días debería tener menor peso en el resultado TTT
        que una señal actualizada ayer.

        Args:
            positions:   Serie de posiciones en $ con DatetimeIndex.
            signal_age:  Serie de edades de señal (enteros ≥ 0).
            halflife:    Días para que la señal conserve el 50% de poder.
            min_factor:  Factor mínimo (cota inferior del decay).

        Returns:
            pd.Series de posiciones ajustadas por decay.
        """
        halflife = max(halflife, 1)
        decay    = np.exp(-np.log(2) * signal_age.astype(float) / halflife)
        decay    = decay.clip(min_factor, 1.0)
        return positions * decay

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTRICAS INTERNAS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_fold_sharpe(
        net_returns: pd.Series, periods_per_year: int = 252
    ) -> float:
        """Sharpe anualizado del fold. NaN si menos de 10 observaciones."""
        r = net_returns.dropna()
        if len(r) < 10:
            return float("nan")
        sigma = r.std(ddof=1)
        if sigma < 1e-12:
            return float("nan")
        return float((r.mean() / sigma) * np.sqrt(periods_per_year))

    @staticmethod
    def _compute_max_drawdown(net_returns: pd.Series) -> float:
        """MDD del fold como número positivo."""
        r = net_returns.dropna()
        if r.empty:
            return 0.0
        wealth = (1.0 + r).cumprod()
        dd = (wealth - wealth.cummax()) / wealth.cummax()
        return float(abs(dd.min()))

    @staticmethod
    def _compute_fold_calmar(net_returns: pd.Series) -> float:
        """Calmar del fold: CAGR anualizado / MDD. NaN si MDD=0."""
        r = net_returns.dropna()
        if len(r) < 10:
            return float("nan")
        wealth  = (1.0 + r).cumprod()
        n_years = len(r) / 252.0
        cagr    = float(wealth.iloc[-1] ** (1.0 / max(n_years, 1e-6)) - 1.0)
        dd      = (wealth - wealth.cummax()) / wealth.cummax()
        mdd     = abs(float(dd.min()))
        if mdd < 1e-10:
            return float("nan")
        return float(cagr / mdd)

    @staticmethod
    def _empty_fold_result(name: str, data: pd.DataFrame) -> FoldResult:
        """Fold vacío cuando hay error en la estrategia."""
        idx   = data.index if not data.empty else pd.DatetimeIndex([])
        start = idx[0]  if len(idx) > 0 else pd.Timestamp("NaT")
        end   = idx[-1] if len(idx) > 0 else pd.Timestamp("NaT")
        return FoldResult(
            strategy_name = name,
            fold_start    = start,
            fold_end      = end,
            net_returns   = pd.Series(0.0, index=idx),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE FÁBRICA — adapt_strategies()
# ══════════════════════════════════════════════════════════════════════════════

def adapt_strategies(
    strategies: List[BaseStrategy],
    capital: float = 100_000,
    cost_model: Optional[TransactionCostModel] = None,
    apply_decay: bool = True,
) -> Dict[str, AdaptedStrategy]:
    """
    Función de fábrica principal: convierte una lista de BaseStrategy en
    AdaptedStrategy listos para el motor Walk-Forward TTT.

    Compatibilidad con walk_forward_engine_con_TTT.py:
        self.adapted_strategies = adapt_strategies(
            list(self.strategy_registry.strategies.values()),
            capital=initial_capital
        )

    Args:
        strategies:  Lista de estrategias del zoo.
        capital:     Capital de referencia (para cálculo de costos y posiciones).
        cost_model:  Modelo de costos. Default: TransactionCostModel() institucional.
        apply_decay: Si True, aplica decay exponencial a señales viejas.

    Returns:
        Dict[strategy_name → AdaptedStrategy] listo para el motor.
    """
    if not cost_model:
        cost_model = TransactionCostModel()

    adapted: Dict[str, AdaptedStrategy] = {}
    for strategy in strategies:
        wrapped = AdaptedStrategy(
            strategy    = strategy,
            cost_model  = cost_model,
            apply_decay = apply_decay,
        )
        adapted[strategy.name] = wrapped
        logger.debug(
            f"adapt_strategies: '{strategy.name}' adaptada | "
            f"halflife={strategy.signal_halflife}d | "
            f"cost={cost_model.one_way_cost*1e4:.1f}bps"
        )

    logger.info(
        f"adapt_strategies: {len(adapted)} estrategias adaptadas | "
        f"one_way_cost={cost_model.one_way_cost*1e4:.1f}bps | "
        f"apply_decay={apply_decay}"
    )
    return adapted
