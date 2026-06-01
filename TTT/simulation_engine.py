# simulation_engine.py
"""
Simulation Engine v1.0 — Motor de Torneo Estocástico TTT
=========================================================
Reemplaza al BacktestEngine financiero tradicional con un marco de
INFERENCIA BAYESIANA DINÁMICA sobre la habilidad latente de los algoritmos.

PARADIGMA DE DISEÑO:
─────────────────────────────────────────────────────────────────────────────
En lugar de medir "retornos" (métrica ex-post sensible a la suerte), este
motor mide la HABILIDAD PREDICTIVA de cada algoritmo: su capacidad de
anticipar correctamente la dirección del estado del sistema en el siguiente
período. Esta métrica — el error cuadrático de predicción (MSE) o la tasa
de acierto direccional (Hit Rate) — es la observable que TTT usa para
inferir la habilidad latente μ(t).

FLUJO PRINCIPAL:
─────────────────────────────────────────────────────────────────────────────
  1. `TTTSimulator` toma el zoo de algoritmos y los datos de mercado.
  2. `run_tournament()` divide los datos en folds temporales con purging.
  3. En cada fold, cada algoritmo genera predicciones sobre el estado siguiente.
  4. Las predicciones se evalúan con MSE y Hit Rate direccional.
  5. Los resultados se convierten en "partidos" TTT: Algoritmo A vs B.
  6. TTTAccumulator acumula todos los partidos.
  7. `History.convergence()` se ejecuta UNA SOLA VEZ al final (backward pass).
  8. Las learning curves resultantes muestran μ(t) y σ(t) por algoritmo.

VENTAJA SOBRE EL BACKTEST CLÁSICO:
─────────────────────────────────────────────────────────────────────────────
El backtest mide performance pasada; TTT infiere habilidad latente futura.
La diferencia es análoga a medir la temperatura de ayer (observación) vs.
inferir el parámetro de difusión del proceso estocástico de temperatura
(inferencia). El backward pass de TTT "suaviza" la curva de habilidad usando
información de todos los folds, produciendo estimaciones más robustas.

Compatibilidad garantizada con walk_forward_engine_con_TTT.py:
  · `BacktestEngine` y `BacktestResult` se mantienen como stubs compatibles.
  · `TTTSimulator` es un reemplazo drop-in de la lógica de evaluación.
  · El `TTTAccumulator` del motor principal se puede alimentar desde aquí.

Referencias:
  Landfried, G. & Mocskos, E. (2024). TrueSkill Through Time. GitHub.
  López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
  Rue, H. & Held, L. (2005). Gaussian Markov Random Fields. Chapman & Hall.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Importación opcional de TTT ─────────────────────────────────────────────
try:
    from trueskillthroughtime import History, Player, Gaussian
    _TTT_AVAILABLE = True
except ImportError:
    _TTT_AVAILABLE = False
    logger.warning(
        "trueskillthroughtime no encontrado. "
        "Instalar: pip install trueskillthroughtime\n"
        "TTTSimulator funcionará en modo STUB (sin inferencia bayesiana)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STUB DE COMPATIBILIDAD — BacktestResult / BacktestEngine
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """
    Contenedor de resultados de evaluación de un algoritmo en un fold.

    Mantiene compatibilidad total con walk_forward_engine_con_TTT.py.
    Extiende el contrato original con métricas de precisión predictiva
    (MSE, hit_rate) que alimentan al motor TTT en lugar de las métricas
    financieras tradicionales.
    """
    # Interface original (compatibilidad)
    returns:      pd.Series = field(default_factory=pd.Series)
    positions:    pd.Series = field(default_factory=pd.Series)
    trades:       pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    metrics:      Dict[str, Any] = field(default_factory=dict)
    alpha_returns: pd.Series = field(default_factory=pd.Series)
    beta:         float = 0.0
    tail_metrics: Dict[str, Any] = field(default_factory=dict)

    # Extensiones TTT (nuevas)
    mse:             float = float("nan")   # Error cuadrático medio de predicción
    hit_rate:        float = float("nan")   # Tasa de acierto direccional [0, 1]
    prediction_bias: float = float("nan")   # Sesgo sistemático de predicción
    n_predictions:   int   = 0              # Número de predicciones válidas
    skill_score:     float = float("nan")   # Brier skill score vs. naïve baseline

    # Extensiones Tier 0 — fricciones (returns ya es NETO de costos)
    gross_returns:   pd.Series = field(default_factory=pd.Series)  # antes de costos
    costs:           pd.Series = field(default_factory=pd.Series)  # costo por periodo
    turnover:        pd.Series = field(default_factory=pd.Series)  # |Δ exposición|


class BacktestEngine:
    """
    Stub de BacktestEngine compatible con walk_forward_engine_con_TTT.py.

    Este stub delega la evaluación real al TTTSimulator.
    Mantiene la interfaz original para no romper el motor principal.
    """

    def __init__(
        self,
        benchmark_symbol: str = "SPY",
        simulator: Optional["TTTSimulator"] = None,
        risk_overlay: Optional[Any] = None,
        cost_model: Optional[Any] = None,
    ) -> None:
        self.benchmark_symbol = benchmark_symbol
        self._simulator = simulator
        # Capa de gestión de riesgo (Target Vol + Trailing Stop). None = exposición desnuda.
        self.risk_overlay = risk_overlay
        # Modelo de fricciones (comisión + slippage). None = backtest sin costos.
        self.cost_model = cost_model

    def run_backtest(
        self,
        strategy: Any,
        market_data: pd.DataFrame,
        initial_capital: float = 100_000,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Ejecuta la evaluación de precisión predictiva del algoritmo.

        En lugar del backtest financiero clásico (P&L), calcula la
        calidad de las predicciones del algoritmo sobre el estado futuro.
        """
        if market_data.empty or "close" not in market_data.columns:
            return self._empty_result(market_data.index)

        # Generar señales del algoritmo
        try:
            signals = strategy.generate_signals(market_data, initial_capital)
        except AttributeError:
            # Retrocompatibilidad: generar_positions legacy
            try:
                positions = strategy.generate_positions(market_data, initial_capital)
                signals = {"positions": positions, "side": np.sign(positions)}
            except Exception as exc:
                logger.warning(f"BacktestEngine: {strategy.name} falló → {exc}")
                return self._empty_result(market_data.index)
        except Exception as exc:
            logger.warning(f"BacktestEngine: {strategy.name} falló → {exc}")
            return self._empty_result(market_data.index)

        positions     = signals.get("positions", pd.Series(0.0, index=market_data.index))
        side          = signals.get("side",      pd.Series(0.0, index=market_data.index))

        # Retornos del activo
        price_ret   = market_data["close"].pct_change().fillna(0.0)
        weight      = (positions.shift(1) / max(initial_capital, 1.0)).fillna(0.0)

        # ── RISK OVERLAY (Target Vol + Trailing Stop) ──────────────────────────
        # Escala el peso ANTES de computar retornos. Vectorizado y sin look-ahead
        # (la vol y el gate del stop usan .shift(1) dentro del overlay).
        if self.risk_overlay is not None:
            weight = self.risk_overlay.compute_risk_weight(
                base_weight=weight, price_ret=price_ret, market_data=market_data,
                strategy_name=getattr(strategy, "name", "_default"),
            )

        gross_ret   = (weight * price_ret)

        # ── COSTOS DE TRANSACCIÓN (Tier 0) ─────────────────────────────────────
        # Descuenta comisión + slippage por turnover. El Juez TTT puntúa el Sharpe
        # sobre retornos NETOS, no brutos. Vectorizado dentro de TransactionCostModel.
        if self.cost_model is not None:
            report   = self.cost_model.apply(gross_ret, weight)
            strat_ret = report.net_returns
            costs_s   = report.costs
            turnover_s = report.turnover
        else:
            strat_ret = gross_ret
            costs_s   = pd.Series(0.0, index=gross_ret.index)
            turnover_s = weight.diff().abs().fillna(0.0)

        # Benchmark (buy-and-hold del activo)
        bench_ret   = price_ret if benchmark_data is None else (
            benchmark_data["close"].pct_change().reindex(market_data.index).fillna(0.0)
        )
        alpha_ret   = strat_ret - bench_ret

        # Métricas de precisión predictiva
        direction_actual    = np.sign(price_ret)
        direction_predicted = side.reindex(price_ret.index).fillna(0.0)
        valid_mask          = direction_actual.abs() > 0

        if valid_mask.sum() > 0:
            hit_rate = float(
                (direction_actual[valid_mask] == direction_predicted[valid_mask]).mean()
            )
            # MSE de las predicciones de dirección (vs la señal de naive drift)
            mse_pred   = float(((direction_actual - direction_predicted) ** 2).mean())
            mse_naive  = float(((direction_actual - direction_actual.mean()) ** 2).mean())
            skill_score = float(1.0 - mse_pred / max(mse_naive, 1e-12))
            pred_bias   = float((direction_predicted - direction_actual).mean())
        else:
            hit_rate = 0.5
            mse_pred = float("nan")
            skill_score = 0.0
            pred_bias = 0.0

        # Equity curve NETA de costos (para compatibilidad visual)
        equity = initial_capital * (1.0 + strat_ret.fillna(0.0)).cumprod()

        return BacktestResult(
            returns      = strat_ret,                  # NETO de costos
            positions    = positions,
            trades       = pd.DataFrame(),
            equity_curve = equity,
            metrics      = {"hit_rate": hit_rate, "skill_score": skill_score},
            alpha_returns= alpha_ret,
            beta         = 0.0,
            tail_metrics = {},
            mse          = mse_pred,
            hit_rate     = hit_rate,
            prediction_bias = pred_bias,
            n_predictions= int(valid_mask.sum()),
            skill_score  = skill_score,
            gross_returns= gross_ret,
            costs        = costs_s,
            turnover     = turnover_s,
        )

    @staticmethod
    def _empty_result(index: pd.Index) -> BacktestResult:
        z = pd.Series(0.0, index=index)
        return BacktestResult(
            returns=z, positions=z, trades=pd.DataFrame(),
            equity_curve=z, metrics={}, alpha_returns=z,
        )


# ══════════════════════════════════════════════════════════════════════════════
# DATOS DE UN ENFRENTAMIENTO (MATCH)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchRecord:
    """
    Registro de un "partido" entre dos algoritmos en un fold.

    En el torneo TTT, cada enfrentamiento es una comparación directa de
    la habilidad predictiva de dos algoritmos en el mismo segmento temporal.
    El resultado [score_A, score_B] es continuo (obs="Continuous") — TTT
    infiere la habilidad latente de la magnitud del diferencial, no solo
    del signo (ganador/perdedor binario).
    """
    fold_idx:       int
    fold_start:     pd.Timestamp
    algo_a:         str
    algo_b:         str
    score_a:        float   # hit_rate u otra métrica de precisión de A
    score_b:        float   # hit_rate u otra métrica de precisión de B
    outcome:        str     # "A_wins" | "B_wins" | "draw"
    time_days:      float   # timestamp en días desde epoch (para TTT)
    n_observations: int     # número de pasos temporales en el fold


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE SIMULACIÓN TTT
# ══════════════════════════════════════════════════════════════════════════════

class TTTSimulator:
    """
    Motor de Torneo Estocástico basado en TrueSkill Through Time.

    DINÁMICA DE SISTEMA:
    ─────────────────────────────────────────────────────────────────────────
    · Los algoritmos son "jugadores" en un sistema dinámico donde la habilidad
      latente μ(t) evoluciona según un proceso de Wiener (random walk gaussiano):
        μ(t+1) = μ(t) + ε,  ε ~ N(0, γ²)
      donde γ (ttt_gamma) controla la volatilidad de la habilidad latente.

    · En cada fold, los algoritmos compiten en parejas (round-robin completo).
      El resultado de cada partido es el DIFERENCIAL de precisión predictiva:
        score_A = hit_rate(A) en fold t
        score_B = hit_rate(B) en fold t
        resultado TTT: [score_A, score_B] con obs="Continuous"

    · Después de acumular todos los partidos, `History.convergence()` realiza
      inferencia bayesiana GLOBAL: el backward pass actualiza μ(t) retrospectivamente
      usando evidencia futura (suavización de Kalman hacia atrás).

    CONTROL DE SESGOS TEMPORALES:
    ─────────────────────────────────────────────────────────────────────────
    · `_randomize_fold_order()`: en cada fold, el orden de evaluación de los
      algoritmos se aleatoriza con semilla reproducible. Esto elimina el
      sesgo de posición (los algoritmos evaluados primero no tienen ventaja
      sistemática por el orden de actualización).
    · Purging: los folds IS/OOS están separados por `purge_days` para
      eliminar correlación serial entre entrenamiento y evaluación.
    · Embargo: buffer adicional de `embargo_days` post-purge.

    Args:
        algorithms:      Lista de estrategias BaseStrategy del zoo.
        ttt_gamma:       Volatilidad de la habilidad latente (proceso estocástico).
                         Valores: 0.01 (casi estacionario) a 0.10 (alta variabilidad).
                         Rec para estrategias de trading: 0.02–0.05.
        ttt_sigma:       Desviación estándar del prior de habilidad (incertidumbre inicial).
                         1.0 = prior no informativo (recomendado si no hay info previa).
        ttt_mu:          Media del prior de habilidad. 0.0 = sin sesgo a priori.
        score_metric:    Métrica de habilidad para los partidos:
                         'hit_rate'    — tasa de acierto direccional [0, 1].
                         'skill_score' — Brier Skill Score vs naive baseline [-∞, 1].
                         'mse_diff'    — diferencia de MSE (negativo → mejor).
        purge_days:      Días de purging en bordes IS/OOS.
        embargo_days:    Días de embargo post-purge.
        random_seed:     Semilla para reproducibilidad del orden de evaluación.
    """

    def __init__(
        self,
        algorithms: List[Any],
        ttt_gamma:    float = 0.03,
        ttt_sigma:    float = 1.0,
        ttt_mu:       float = 0.0,
        score_metric: str   = "hit_rate",
        purge_days:   int   = 5,
        embargo_days: int   = 10,
        random_seed:  int   = 42,
        apply_risk_overlay: bool = True,
        risk_overlay: Optional[Any] = None,
        apply_costs:  bool = True,
        cost_model:   Optional[Any] = None,
    ) -> None:
        self.algorithms    = algorithms
        self.ttt_gamma     = ttt_gamma
        self.ttt_sigma     = ttt_sigma
        self.ttt_mu        = ttt_mu
        self.score_metric  = score_metric
        self.purge_days    = purge_days
        self.embargo_days  = embargo_days
        self.rng           = np.random.default_rng(random_seed)

        # ── Capa de Gestión de Riesgo (Target Vol + Trailing Stop) ────────────
        # Activa por defecto: toda estrategia se evalúa con exposición gestionada.
        if apply_risk_overlay and risk_overlay is None:
            from risk_overlay import RiskOverlay
            risk_overlay = RiskOverlay()
        elif not apply_risk_overlay:
            risk_overlay = None
        self.risk_overlay = risk_overlay

        # ── Modelo de costos de transacción (Tier 0) ──────────────────────────
        # Activo por defecto: el Juez puntúa el Sharpe sobre retornos NETOS.
        if apply_costs and cost_model is None:
            from transaction_costs import TransactionCostModel
            cost_model = TransactionCostModel()
        elif not apply_costs:
            cost_model = None
        self.cost_model = cost_model

        self._backtest_engine = BacktestEngine(
            risk_overlay=risk_overlay, cost_model=cost_model
        )

        # Acumuladores del torneo (llenados por run_tournament)
        self._matches:     List[MatchRecord] = []
        self._composition: List = []
        self._results:     List = []
        self._times:       List = []
        self._obs:         List = []
        self._algo_names:  set  = {a.name for a in algorithms}

        # Resultados post-convergencia
        self.history_:        Optional[Any]              = None
        self.learning_curves_: Dict[str, pd.DataFrame]  = {}
        self.tournament_log_:  pd.DataFrame              = pd.DataFrame()

        overlay_desc = self.risk_overlay.describe() if self.risk_overlay else "OFF (exposición desnuda)"
        cost_desc = repr(self.cost_model) if self.cost_model else "OFF (sin fricciones)"
        logger.info(
            f"TTTSimulator: {len(algorithms)} algoritmos | "
            f"γ={ttt_gamma} | σ={ttt_sigma} | μ={ttt_mu} | "
            f"metric={score_metric} | purge={purge_days}d | embargo={embargo_days}d | "
            f"RiskOverlay={overlay_desc} | Costs={cost_desc}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RUN TOURNAMENT — Orquestador principal
    # ──────────────────────────────────────────────────────────────────────────

    def run_tournament(
        self,
        market_data:    pd.DataFrame,
        n_folds:        int   = 10,
        min_is_days:    int   = 252,
        fold_size_days: int   = 21,
        capital:        float = 100_000,
        verbose:        bool  = True,
    ) -> "TTTSimulator":
        """
        Orquesta el torneo completo sobre todos los folds temporales.

        PROTOCOLO DE TORNEO:
        ─────────────────────────────────────────────────────────────────────
        Para cada fold t = 1..n_folds:
          1. Extraer segmento OOS con purging+embargo correctos.
          2. Aleatorizar el orden de evaluación de los algoritmos (anti-sesgo).
          3. Evaluar cada algoritmo sobre el segmento OOS:
             score(algo, fold) = hit_rate u otra métrica de precisión.
          4. Registrar partidos round-robin: para cada par (A, B):
             partido = [score_A, score_B] con obs="Continuous".
          5. Acumular en las listas TTT.

        Post-torneo:
          6. `_fit_ttt()` ejecuta `History.convergence()` UNA SOLA VEZ.
          7. Extraer learning curves por algoritmo.

        Args:
            market_data:    DataFrame OHLCV con DatetimeIndex.
            n_folds:        Número de folds OOS.
            min_is_days:    Mínimo de días IS antes del primer fold OOS.
            fold_size_days: Días por fold OOS.
            capital:        Capital para generar señales (normalizado internamente).
            verbose:        Imprimir progreso del torneo.

        Returns:
            self (para chaining): TTTSimulator.run_tournament(...).get_curves()
        """
        if market_data.empty or "close" not in market_data.columns:
            raise ValueError("market_data vacío o sin columna 'close'.")

        # Reset del estado global del Risk Overlay: cada torneo parte de HWM=1.0
        # (evita contaminación entre ejecuciones independientes del simulador).
        if self.risk_overlay is not None and hasattr(self.risk_overlay, "reset"):
            self.risk_overlay.reset()

        dates    = market_data.index
        n_total  = len(dates)
        buffer   = self.purge_days + self.embargo_days

        # Verificar datos suficientes
        required = min_is_days + n_folds * fold_size_days + buffer
        if n_total < required:
            raise ValueError(
                f"Datos insuficientes: {n_total}d disponibles, "
                f"~{required}d requeridos para {n_folds} folds."
            )

        if verbose:
            print("=" * 80)
            print(f"TORNEO TTT — {len(self.algorithms)} algoritmos × {n_folds} folds")
            print(f"γ={self.ttt_gamma} | σ={self.ttt_sigma} | "
                  f"métrica={self.score_metric} | purge={self.purge_days}d")
            print(f"Datos: {dates[0].date()} → {dates[-1].date()} ({n_total}d)")
            print("=" * 80)

        algo_names = [a.name for a in self.algorithms]
        all_pairs  = list(combinations(algo_names, 2))

        # Generar folds OOS (ventana deslizante desde el final del dataset)
        folds_generated = 0
        for fold_idx in range(n_folds):

            # Anclar OOS desde el final hacia atrás
            oos_end_idx   = n_total - 1 - (n_folds - 1 - fold_idx) * fold_size_days
            oos_start_idx = oos_end_idx - fold_size_days + 1

            if oos_start_idx < 0 or oos_end_idx >= n_total:
                logger.debug(f"Fold {fold_idx}: índices OOS fuera de rango — saltando.")
                continue

            # IS: todo lo que está antes del OOS minus buffer
            is_end_idx = oos_start_idx - 1 - buffer
            is_start_idx = max(0, is_end_idx - min_is_days - fold_idx * fold_size_days)

            if is_end_idx - is_start_idx < min_is_days // 2:
                logger.debug(f"Fold {fold_idx}: IS demasiado corto — saltando.")
                continue

            oos_data = market_data.iloc[oos_start_idx:oos_end_idx + 1]
            if oos_data.empty:
                continue

            # Ventana IN-SAMPLE estricta (anterior al OOS, ya con purge+embargo).
            # Se entrega a fit() para las estrategias ML — SIN solapamiento con OOS.
            is_data = market_data.iloc[is_start_idx:is_end_idx + 1]

            fold_start = oos_data.index[0]

            # Tiempo en días desde epoch (formato logistic.py)
            t_days = float(fold_start.timestamp() / 86_400.0)

            if verbose:
                print(
                    f"\n  Fold {fold_idx + 1}/{n_folds} | "
                    f"OOS: {fold_start.date()} → {oos_data.index[-1].date()} "
                    f"({len(oos_data)}d)"
                )

            # ── Aleatorizar orden de evaluación (anti-sesgo de posición) ────────
            eval_order = self._randomize_fold_order(
                self.algorithms, seed_offset=fold_idx
            )

            # ── Evaluar cada algoritmo en el fold OOS ─────────────────────────
            fold_scores: Dict[str, float] = {}
            for algo in eval_order:
                # RE-FIT WALK-FORWARD: entrenar solo con la ventana IS (anti-fuga).
                # Las estrategias basadas en reglas heredan un fit() no-op.
                try:
                    algo.fit(is_data, capital)
                except Exception as exc:
                    logger.warning(f"{getattr(algo,'name','?')}.fit falló → {exc}")
                score = self._evaluate_algorithm(algo, oos_data, capital)
                fold_scores[algo.name] = score
                if verbose:
                    print(f"    {algo.name:40s} → {self.score_metric}={score:.4f}")

            # ── Registrar partidos round-robin ────────────────────────────────
            for name_a, name_b in all_pairs:
                sc_a = fold_scores.get(name_a, 0.5)
                sc_b = fold_scores.get(name_b, 0.5)

                # Determinar resultado categórico (para log)
                eps = 1e-6
                if sc_a > sc_b + eps:
                    outcome = "A_wins"
                elif sc_b > sc_a + eps:
                    outcome = "B_wins"
                else:
                    outcome = "draw"

                match = MatchRecord(
                    fold_idx       = fold_idx,
                    fold_start     = fold_start,
                    algo_a         = name_a,
                    algo_b         = name_b,
                    score_a        = sc_a,
                    score_b        = sc_b,
                    outcome        = outcome,
                    time_days      = t_days,
                    n_observations = len(oos_data),
                )
                self._matches.append(match)

                # Acumular en formato TTT
                self._composition.append([[name_a], [name_b]])
                self._results.append([sc_a, sc_b])
                self._times.append(t_days)
                self._obs.append("Continuous")

            folds_generated += 1

        if folds_generated == 0:
            raise RuntimeError(
                "No se generó ningún fold válido. "
                "Aumentar el dataset o reducir n_folds."
            )

        logger.info(
            f"Torneo: {folds_generated} folds × "
            f"{len(all_pairs)} pares = "
            f"{len(self._matches)} partidos acumulados."
        )

        if verbose:
            print(f"\n  {len(self._matches)} partidos acumulados en {folds_generated} folds.")
            print("  Ejecutando convergencia global TTT...")

        # ── Inferencia bayesiana global — ÚNICA llamada a convergence() ───────
        self._fit_ttt(verbose=verbose)
        self._build_tournament_log()

        if verbose:
            print("\n[TORNEO COMPLETADO]")
            if self.learning_curves_:
                print(f"  Learning curves disponibles: {list(self.learning_curves_.keys())}")

        return self

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUACIÓN DE UN ALGORITMO EN UN FOLD
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_algorithm(
        self,
        algo: Any,
        oos_data: pd.DataFrame,
        capital: float,
    ) -> float:
        """
        Evalúa la precisión predictiva de un algoritmo en un fold OOS.

        MÉTRICAS DISPONIBLES:
        ─────────────────────────────────────────────────────────────────────
        · hit_rate:    Fracción de períodos en que la dirección predicha coincide
                       con la dirección real del sistema.
                         hit_rate = P(sign(pred_t) == sign(actual_t))
                       Esta es la medida más pura de habilidad predictiva binaria.

        · skill_score: Brier Skill Score normalizado respecto a un predictor naïve
                       (media histórica):
                         BSS = 1 - MSE_pred / MSE_naive
                       BSS=1 → predictor perfecto; BSS=0 → equivalente al naïve;
                       BSS<0 → peor que adivinar al azar.

        · mse_diff:    Diferencia de MSE respecto al naïve baseline. Valores
                       negativos indican que el algoritmo supera al baseline.
                       Para partidos TTT, se invierte el signo (mayor es mejor).

        Args:
            algo:     Algoritmo del zoo (con generate_signals()).
            oos_data: DataFrame OOS con OHLCV.
            capital:  Capital de referencia.

        Returns:
            float — score del algoritmo en este fold (mayor = mejor).
        """
        result = self._backtest_engine.run_backtest(
            algo, oos_data, initial_capital=capital
        )

        if self.score_metric == "hit_rate":
            score = result.hit_rate if np.isfinite(result.hit_rate) else 0.5
        elif self.score_metric == "skill_score":
            score = result.skill_score if np.isfinite(result.skill_score) else 0.0
        elif self.score_metric == "mse_diff":
            # Para TTT: mayor = mejor; MSE más bajo → invertir signo
            mse = result.mse if np.isfinite(result.mse) else 1.0
            score = float(1.0 - mse)  # mapear a [−∞, 1] con 1=perfecto
        elif self.score_metric in ("sharpe", "sharpe_fold"):
            # Sharpe anualizado OOS — premia retorno ajustado por riesgo, NO beta.
            score = self._fold_risk_score(result.returns, kind="sharpe")
        elif self.score_metric in ("sortino", "sortino_fold"):
            # Sortino anualizado OOS — penaliza solo la volatilidad a la baja.
            score = self._fold_risk_score(result.returns, kind="sortino")
        else:
            logger.warning(
                f"score_metric='{self.score_metric}' desconocido. Usando hit_rate."
            )
            score = result.hit_rate if np.isfinite(result.hit_rate) else 0.5

        # ── SALVAGUARDA NaN/inf ────────────────────────────────────────────────
        # Estrategias degeneradas (señal plana → retornos todo-cero → Sharpe
        # indefinido) NO deben inyectar NaN/inf al Juez TTT: se mapean a 0.0
        # (rendimiento neutro), preservando la estabilidad del belief-propagation.
        return float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))

    @staticmethod
    def _fold_risk_score(
        returns: pd.Series,
        kind: str = "sharpe",
        periods_per_year: int = 252,
    ) -> float:
        """
        Métrica de riesgo OOS para el Juez TTT (mayor = mejor).

        `returns` son los retornos diarios netos de la estrategia en el fold,
        ya calculados con posición rezagada (`positions.shift(1)`) → sin
        look-ahead. Devuelve 0.0 si hay <10 observaciones o vol nula
        (estrategia plana), de modo que un algoritmo inactivo no gana ni pierde.
        """
        r = returns.dropna() if returns is not None else pd.Series(dtype=float)
        if len(r) < 10:
            return 0.0
        mean = float(r.mean())
        if kind == "sortino":
            downside = float(r[r < 0].std(ddof=1))
            denom = downside
        else:
            denom = float(r.std(ddof=1))
        if not np.isfinite(denom) or denom < 1e-12:
            return 0.0
        return float((mean / denom) * np.sqrt(periods_per_year))

    # ──────────────────────────────────────────────────────────────────────────
    # CONTROL DE SESGOS — Aleatorización del orden de evaluación
    # ──────────────────────────────────────────────────────────────────────────

    def _randomize_fold_order(
        self,
        algorithms: List[Any],
        seed_offset: int = 0,
    ) -> List[Any]:
        """
        Aleatoriza el orden de evaluación de los algoritmos en cada fold.

        PROPÓSITO — ELIMINACIÓN DEL SESGO DE POSICIÓN:
        ─────────────────────────────────────────────────────────────────────
        Si los algoritmos siempre se evaluaran en el mismo orden, el primero
        tendría ventaja sistemática en sistemas con recursos compartidos
        (acceso a caché, warm-up de datos, etc.). La aleatorización con
        semilla determinista por fold garantiza:
        1. Reproducibilidad: dado el mismo seed + fold_idx → mismo orden.
        2. Independencia: orden diferente en cada fold.
        3. Balance asintótico: sobre N folds, cada algoritmo aparece
           uniformemente en todas las posiciones del orden de evaluación.

        La semilla es: self.rng.integers(0, 2**32) XOR seed_offset.
        Esto garantiza que folds distintos tengan órdenes distintos.

        Args:
            algorithms:  Lista de algoritmos a aleatorizar.
            seed_offset: Offset de fold para variar la semilla por fold.

        Returns:
            Lista de algoritmos en orden aleatorizado.
        """
        indices = list(range(len(algorithms)))
        # Mezcla de Fisher-Yates con semilla determinista por fold
        fold_rng = np.random.default_rng(
            seed=int(self.rng.integers(1, 2**31)) ^ seed_offset
        )
        fold_rng.shuffle(indices)
        return [algorithms[i] for i in indices]

    # ──────────────────────────────────────────────────────────────────────────
    # INFERENCIA BAYESIANA GLOBAL — History.convergence()
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_ttt(
        self,
        iterations: int  = 6,
        epsilon:    float = 1e-3,
        verbose:    bool  = False,
    ) -> None:
        """
        Ajusta el modelo TTT sobre todos los partidos acumulados.

        FUNDAMENTO ESTADÍSTICO:
        ─────────────────────────────────────────────────────────────────────
        `History.convergence()` implementa un algoritmo de paso de mensajes
        (belief propagation) sobre un grafo de factores bayesianos. El proceso:

        1. FORWARD PASS: propaga creencias temporalmente hacia adelante.
           Para cada partido t, la creencia posterior sobre μ(t) se actualiza
           usando la verosimilitud del resultado observado.

        2. BACKWARD PASS (Suavización): propaga información futura hacia atrás.
           Este es el motivo de inferir globalmente (no fold a fold): la
           información de partidos futuros mejora la estimación de μ en
           períodos pasados. Es equivalente al smoother de Kalman-Rauch-Tung-Striebel.

        3. CONVERGENCIA: itera hasta que el cambio en las creencias sea < epsilon.
           Landfried (2024) recomienda ≥ 3 iteraciones para series largas.

        La complejidad es O(N_matches × N_iterations) en tiempo y
        O(N_players × T_folds) en espacio — eficiente para torneos de cientos
        de folds con docenas de jugadores.

        Ref: Landfried & Mocskos (2024), Algorithm 1 (sec. 3.1).
             Rue & Held (2005), Gaussian Markov Random Fields, Cap. 4.
        """
        if not _TTT_AVAILABLE:
            logger.warning(
                "TTT no disponible. Instalar: pip install trueskillthroughtime\n"
                "Las learning curves no estarán disponibles."
            )
            return

        if not self._composition:
            logger.warning("_fit_ttt: ningún partido acumulado. Abortando.")
            return

        logger.info(
            f"TTT fit: {len(self._composition)} partidos | "
            f"{len(self._algo_names)} algoritmos | "
            f"γ={self.ttt_gamma} | σ={self.ttt_sigma} | obs=Continuous"
        )

        try:
            h = History(
                composition = self._composition,
                results     = self._results,
                times       = self._times,
                mu          = self.ttt_mu,
                sigma       = self.ttt_sigma,
                gamma       = self.ttt_gamma,
                p_draw      = 0.10,
            )
            result_conv = h.convergence(
                iterations = iterations,
                epsilon    = epsilon,
                verbose    = verbose,
            )
            step_tuple, n_iter = result_conv
            step = max(step_tuple) if isinstance(step_tuple, (tuple, list)) else float(step_tuple)
            if step > epsilon:
                logger.warning(
                    f"TTT no convergió en {n_iter} iteraciones (step={step:.2e})."
                )
            else:
                logger.info(
                    f"TTT convergió en {n_iter} iteraciones (step={step:.2e})."
                )

            self.history_ = h
            self.learning_curves_ = self._extract_learning_curves(h)

        except Exception as exc:
            logger.error(f"_fit_ttt: error durante convergencia → {exc}")
            self.history_ = None

    def _extract_learning_curves(
        self, h: Any
    ) -> Dict[str, pd.DataFrame]:
        """
        Extrae las curvas de aprendizaje por algoritmo del objeto History.

        Sintaxis idéntica a logistic.py de Landfried:
          lc = h.learning_curves()
          mu    = [tp[1].mu    for tp in lc["nombre_algo"]]
          sigma = [tp[1].sigma for tp in lc["nombre_algo"]]

        Convierte al formato DataFrame con columnas:
          [time_days, time_date, ttt_mu, ttt_sigma]

        Retrocompatible con ttt_curves_ del motor principal.

        Returns:
            Dict[algo_name → DataFrame] con las curvas de habilidad.
        """
        lc_raw = h.learning_curves()
        curves: Dict[str, pd.DataFrame] = {}

        for name in self._algo_names:
            if name not in lc_raw:
                continue
            entries = lc_raw[name]
            if not entries:
                continue

            times_d  = [tp[0] for tp in entries]
            mus      = [float(tp[1].mu)    for tp in entries]
            sigmas   = [float(tp[1].sigma) for tp in entries]

            curves[name] = pd.DataFrame({
                "time_days":  times_d,
                "time_date":  pd.to_datetime(
                    [t * 86_400 for t in times_d], unit="s", utc=True
                ).tz_localize(None),
                "ttt_mu":    mus,
                "ttt_sigma": sigmas,
            })

        return curves

    # ──────────────────────────────────────────────────────────────────────────
    # LOG DE TORNEO
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tournament_log(self) -> None:
        """Construye el DataFrame de log de todos los partidos del torneo."""
        if not self._matches:
            return
        rows = [
            {
                "fold_idx":       m.fold_idx,
                "fold_start":     m.fold_start,
                "algo_a":         m.algo_a,
                "algo_b":         m.algo_b,
                "score_a":        m.score_a,
                "score_b":        m.score_b,
                "outcome":        m.outcome,
                "time_days":      m.time_days,
                "n_observations": m.n_observations,
            }
            for m in self._matches
        ]
        self.tournament_log_ = pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────────────────
    # INTERFAZ PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def get_curves(self) -> Dict[str, pd.DataFrame]:
        """Devuelve las learning curves por algoritmo (post-convergencia)."""
        return self.learning_curves_

    def get_final_skill(
        self, k_conservative: float = 3.0
    ) -> pd.DataFrame:
        """
        Habilidad final de cada algoritmo al término del torneo.

        Calcula μ_final y σ_final (última estimación post-convergencia) y el
        skill score conservador = μ - k*σ (Herbrich et al., 2007).

        Args:
            k_conservative: Factor de conservadurismo (default 3.0 = límite 99.7%).

        Returns:
            DataFrame con [rank, algorithm, mu_final, sigma_final, conservative_skill]
            ordenado de mayor a menor habilidad conservadora.
        """
        records = []
        for name, curve in self.learning_curves_.items():
            if curve.empty:
                continue
            mu_f    = float(curve["ttt_mu"].iloc[-1])
            sigma_f = float(curve["ttt_sigma"].iloc[-1])
            cs      = mu_f - k_conservative * sigma_f
            records.append({
                "algorithm":          name,
                "mu_final":           mu_f,
                "sigma_final":        sigma_f,
                "conservative_skill": cs,
            })

        if not records:
            return pd.DataFrame()

        df = (
            pd.DataFrame(records)
            .sort_values("conservative_skill", ascending=False)
            .reset_index(drop=True)
        )
        df.insert(0, "rank", df.index + 1)
        return df

    def get_win_matrix(self) -> pd.DataFrame:
        """
        Matriz de victorias directas A vs B en el torneo (head-to-head).

        Elemento [i, j] = número de folds en que el algoritmo i superó al j.
        Útil como complemento de la inferencia TTT para verificar consistencia.

        Returns:
            DataFrame cuadrado con algoritmos en filas y columnas.
        """
        if self.tournament_log_.empty:
            return pd.DataFrame()

        names = sorted(self._algo_names)
        mat   = pd.DataFrame(0, index=names, columns=names)

        for _, row in self.tournament_log_.iterrows():
            if row["outcome"] == "A_wins":
                mat.at[row["algo_a"], row["algo_b"]] += 1
            elif row["outcome"] == "B_wins":
                mat.at[row["algo_b"], row["algo_a"]] += 1

        return mat

    def export_ttt_data(
        self
    ) -> Tuple[List, List, List, List]:
        """
        Exporta las listas TTT acumuladas para uso externo.

        Permite alimentar un TTTAccumulator externo (del motor principal)
        con los partidos del simulador, combinando ambos sistemas.

        Returns:
            Tuple (composition, results, times, obs) en formato logistic.py.
        """
        return (
            self._composition.copy(),
            self._results.copy(),
            self._times.copy(),
            self._obs.copy(),
        )

    def summary(self) -> None:
        """Imprime un resumen del torneo en consola."""
        n_matches = len(self._matches)
        n_folds   = len(set(m.fold_idx for m in self._matches))
        converged = self.history_ is not None

        print("\n" + "=" * 70)
        print("RESUMEN DEL TORNEO TTT")
        print("=" * 70)
        print(f"  Algoritmos:      {len(self._algo_names)}")
        print(f"  Folds evaluados: {n_folds}")
        print(f"  Partidos totales:{n_matches}")
        print(f"  TTT convergido:  {'Sí ✓' if converged else 'No ✗'}")
        print(f"  Métrica usada:   {self.score_metric}")
        print(f"  γ (dinámica):    {self.ttt_gamma}")
        print(f"  σ (prior):       {self.ttt_sigma}")

        if converged:
            skill_df = self.get_final_skill()
            if not skill_df.empty:
                print(f"\n  Ranking de habilidad final (μ - 3σ):")
                print(
                    skill_df[["rank", "algorithm", "mu_final",
                              "sigma_final", "conservative_skill"]]
                    .to_string(index=False)
                )
        print("=" * 70)
