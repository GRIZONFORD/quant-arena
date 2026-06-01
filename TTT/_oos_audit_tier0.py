# _oos_audit_tier0.py — Auditoría OOS NETA de costos + puerta DSR/PBO
"""
Re-audita el track walk-forward 2005-2024 con el motor Tier 0 completo
(RiskOverlay global + TransactionCostModel) y aplica las puertas de
significancia estadística: Deflated Sharpe Ratio y Probability of Backtest
Overfitting sobre la matriz de ensayos del zoo.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from strategy_zoo import StrategyRegistry
from simulation_engine import BacktestEngine
from risk_overlay import RiskOverlay
from transaction_costs import TransactionCostModel
from statistical_validation import StatisticalSignificanceValidator
from run_ttt_analysis import _load_market_data

ANN = 252
N_FOLDS, FOLD, MIN_IS, PURGE, EMBARGO = 200, 21, 252, 5, 10


def equity_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"sharpe": float("nan"), "max_dd": float("nan"), "cagr": float("nan")}
    sd = r.std(ddof=1)
    eq = (1 + r).cumprod()
    return {
        "sharpe": (r.mean() / sd) * np.sqrt(ANN) if sd > 0 else float("nan"),
        "max_dd": (eq / eq.cummax() - 1).min(),
        "cagr":   eq.iloc[-1] ** (ANN / len(r)) - 1,
    }


def main():
    md = _load_market_data("SPY", "2005-01-01", "2024-12-31")
    md = md[~md.index.duplicated(keep="last")].sort_index()
    n = len(md); buffer = PURGE + EMBARGO

    strategies = list(StrategyRegistry().create_default_universe().strategies.values())
    # Motor Tier 0 COMPLETO: overlay global + costos (retornos NETOS).
    engine = BacktestEngine(
        risk_overlay=RiskOverlay(),
        cost_model=TransactionCostModel(commission_bps=0.5, slippage_bps=1.0),
    )

    acc = {s.name: [] for s in strategies}
    for fi in range(N_FOLDS):
        oos_end = n - 1 - (N_FOLDS - 1 - fi) * FOLD
        oos_start = oos_end - FOLD + 1
        if oos_start < 0 or oos_end >= n:
            continue
        is_end = oos_start - 1 - buffer
        is_start = max(0, is_end - MIN_IS - fi * FOLD)
        if is_end - is_start < MIN_IS // 2:
            continue
        is_data, oos_data = md.iloc[is_start:is_end + 1], md.iloc[oos_start:oos_end + 1]
        for s in strategies:
            try:
                s.fit(is_data)
            except Exception:
                pass
            acc[s.name].append(engine.run_backtest(s, oos_data).returns)

    # Matriz de ensayos (filas=tiempo OOS, columnas=estrategias), NETA de costos.
    trials = pd.DataFrame({
        name: pd.concat(parts).sort_index() for name, parts in acc.items()
    })
    trials = trials[~trials.index.duplicated(keep="last")].fillna(0.0)

    # Filtrar estrategias degeneradas (nunca operan → retorno idénticamente 0).
    active = [c for c in trials.columns if trials[c].abs().sum() > 1e-9]
    degenerate = [c for c in trials.columns if c not in active]
    trials_active = trials[active]

    val = StatisticalSignificanceValidator(periods_per_year=ANN)
    sharpes = trials_active.apply(val.sharpe_periodic, axis=0).dropna()
    sr_var = float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0
    n_trials = trials_active.shape[1]

    print(f"\nVentana OOS NETA: {trials.index[0].date()} → {trials.index[-1].date()} "
          f"({len(trials)} días) | N ensayos activos = {n_trials}")
    print(f"Costos: 1.50 bps/turnover | Overlay: TargetVol 15% / cap 1.0x / Stop GLOBAL -15%")
    print(f"Var(Sharpe periódico) entre ensayos V = {sr_var:.6e}")
    print(f"Estrategias degeneradas excluidas: {degenerate}\n")

    print(f"{'Estrategia':24s} | {'Sharpe':>7s} {'MaxDD':>8s} {'CAGR':>7s} | "
          f"{'PSR0':>6s} {'DSR':>6s} {'Signif':>7s}")
    print("-" * 80)
    for name in active:
        m = equity_metrics(trials_active[name])
        rep = val.evaluate(trials_active[name], sr_var, n_trials)
        signif = "SÍ" if rep.is_significant else "no"
        print(f"{name:24s} | {m['sharpe']:7.3f} {m['max_dd']*100:7.2f}% {m['cagr']*100:6.2f}% | "
              f"{rep.psr_zero:6.3f} {rep.dsr:6.3f} {signif:>7s}")

    sr0 = val.expected_max_sharpe(sr_var, n_trials)
    pbo = val.probability_of_backtest_overfitting(trials_active, n_partitions=16)
    print("-" * 80)
    print(f"Umbral de deflación SR0 (periódico) = {sr0:.4f}  "
          f"(≈ {sr0*np.sqrt(ANN):.3f} anualizado)")
    print(f"PBO (CSCV, S=16) = {pbo:.3f}   "
          f"→ {'SOBREAJUSTE probable' if pbo > 0.5 else 'aceptable'}")
    print("\nNota: DSR ≥ 0.95 ⇒ significativo al 95% tras deflación por N ensayos.")


if __name__ == "__main__":
    main()
