# _audit_run.py — Ejecución de auditoría con DATOS REALES (sin fallback sintético)
"""
Corre el torneo TTT sobre SPY REAL (yfinance) — aborta si la descarga falla
en lugar de caer en datos sintéticos (PROHIBIDOS para un backtest válido).
Además computa métricas de riesgo REALES (Sharpe, Sortino, Max Drawdown) sobre
los retornos diarios de SPY: periodo completo y ventanas de crisis 2008 / 2020.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

ANN = 252


def fetch_real_spy(start="1997-01-01", end="2024-12-31") -> pd.DataFrame:
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(start=start, end=end, tickers="SPY",
                          auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Descarga yfinance vacía — ABORTADO (sin sintéticos).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.dropna(how="all").sort_index()


def risk_metrics(close: pd.Series) -> dict:
    """Sharpe/Sortino anualizados y Max Drawdown sobre retornos simples diarios."""
    ret = close.pct_change().dropna()
    if len(ret) < 10:
        return {"n": len(ret), "sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan}
    mu, sd = ret.mean(), ret.std()
    downside = ret[ret < 0].std()
    sharpe = (mu / sd) * np.sqrt(ANN) if sd > 0 else np.nan
    sortino = (mu / downside) * np.sqrt(ANN) if downside and downside > 0 else np.nan
    equity = (1.0 + ret).cumprod()
    max_dd = (equity / equity.cummax() - 1.0).min()
    cagr = equity.iloc[-1] ** (ANN / len(ret)) - 1.0
    return {"n": len(ret), "cagr": cagr, "sharpe": sharpe,
            "sortino": sortino, "max_dd": max_dd}


def main():
    print("=" * 78)
    print("AUDITORÍA V2 — DATOS REALES (SPY, yfinance) | sin fallback sintético")
    print("=" * 78)

    spy = fetch_real_spy()
    px = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    print(f"\nSPY REAL: {len(spy)} días | {spy.index[0].date()} → {spy.index[-1].date()}")

    # ── Métricas de riesgo REALES (Buy & Hold SPY) ───────────────────────────
    print("\n[RIESGO REAL · Buy&Hold SPY]")
    windows = {
        "Periodo completo (1997-2024)": px,
        "Crisis GFC (2007-07 → 2009-06)": px.loc["2007-07-01":"2009-06-30"],
        "Crisis COVID (2020-01 → 2020-06)": px.loc["2020-01-01":"2020-06-30"],
    }
    for label, series in windows.items():
        m = risk_metrics(series)
        print(f"  {label:38s} | n={m['n']:5d} | "
              f"Sharpe={m['sharpe']:.3f} | Sortino={m['sortino']:.3f} | "
              f"MaxDD={m['max_dd']*100:6.2f}%")

    # ── Torneo TTT sobre datos REALES ────────────────────────────────────────
    from strategy_zoo import StrategyRegistry
    from simulation_engine import TTTSimulator
    from analyzer import run_full_analysis

    registry = StrategyRegistry().create_default_universe()
    strategies = list(registry.strategies.values())
    print(f"\n[TORNEO TTT REAL] {len(strategies)} algoritmos | "
          f"folds que alcanzan 2008 y 2020")

    sim = TTTSimulator(algorithms=strategies, ttt_gamma=0.03, ttt_sigma=1.0,
                       ttt_mu=0.0, score_metric="hit_rate",
                       purge_days=5, embargo_days=10, random_seed=42)
    sim.run_tournament(market_data=spy, n_folds=180, min_is_days=252,
                       fold_size_days=21, verbose=False)
    curves = sim.get_curves()
    if not curves:
        print("⚠ Sin curvas TTT.")
        return
    analysis = run_full_analysis(curves, output_html="ttt_dashboard_real.html",
                                 verbose=False, k_conservative=3.0,
                                 dominance_threshold=0.80)
    summary = analysis.get("skill_summary")
    if summary is not None:
        summary.to_csv("ttt_skill_summary_real.csv", index=False)
        print("\n[RANKING DE HABILIDAD TTT — DATOS REALES]")
        cols = [c for c in ["rank", "algorithm", "mu_final", "sigma_final",
                            "conservative_skill"] if c in summary.columns]
        print(summary[cols].to_string(index=False))
    print("\n✓ Auditoría real completada (ttt_skill_summary_real.csv).")


if __name__ == "__main__":
    main()
