# _oos_metrics.py — Métricas de EQUITY OOS reales (overlay vs desnudo vs B&H)
"""
Reconstruye la pista de retornos OOS (2005-2024) stitcheando los 200 folds
walk-forward EXACTAMENTE como TTTSimulator (mismo split, mismo BacktestEngine,
mismo RiskOverlay, mismo fit() por fold). Computa Sharpe/Sortino/MaxDD reales
del equity — lo que el ranking de habilidad TTT no expone — y aísla el drawdown
en las crisis de 2008 (GFC) y 2020 (COVID).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from strategy_zoo import StrategyRegistry
from simulation_engine import BacktestEngine
from risk_overlay import RiskOverlay
from run_ttt_analysis import _load_market_data

ANN = 252
N_FOLDS, FOLD, MIN_IS, PURGE, EMBARGO = 200, 21, 252, 5, 10


def metrics(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {"n": len(r), "sharpe": float("nan"), "sortino": float("nan"),
                "max_dd": float("nan"), "cagr": float("nan")}
    sd = r.std(ddof=1); dn = r[r < 0].std(ddof=1)
    eq = (1 + r).cumprod()
    return {
        "n": len(r),
        "sharpe":  (r.mean()/sd)*np.sqrt(ANN) if sd > 0 else float("nan"),
        "sortino": (r.mean()/dn)*np.sqrt(ANN) if dn and dn > 0 else float("nan"),
        "max_dd":  (eq/eq.cummax() - 1).min(),
        "cagr":    eq.iloc[-1]**(ANN/len(r)) - 1,
    }


def dd_window(ret: pd.Series, start: str, end: str) -> float:
    r = ret.loc[start:end].dropna()
    if len(r) < 5:
        return float("nan")
    eq = (1 + r).cumprod()
    return (eq/eq.cummax() - 1).min()


def main():
    md = _load_market_data("SPY", "2005-01-01", "2024-12-31")
    md = md[~md.index.duplicated(keep="last")].sort_index()
    n = len(md); buffer = PURGE + EMBARGO
    print(f"market_data: {n} días | {md.index[0].date()} → {md.index[-1].date()}")

    strategies = list(StrategyRegistry().create_default_universe().strategies.values())
    eng_ov = BacktestEngine(risk_overlay=RiskOverlay())   # 15% / 1.5x / -15%
    eng_nk = BacktestEngine(risk_overlay=None)            # exposición desnuda

    acc_ov = {s.name: [] for s in strategies}
    acc_nk = {s.name: [] for s in strategies}

    for fi in range(N_FOLDS):
        oos_end = n - 1 - (N_FOLDS - 1 - fi) * FOLD
        oos_start = oos_end - FOLD + 1
        if oos_start < 0 or oos_end >= n:
            continue
        is_end = oos_start - 1 - buffer
        is_start = max(0, is_end - MIN_IS - fi * FOLD)
        if is_end - is_start < MIN_IS // 2:
            continue
        is_data = md.iloc[is_start:is_end + 1]
        oos_data = md.iloc[oos_start:oos_end + 1]

        for s in strategies:
            try:
                s.fit(is_data)
            except Exception:
                pass
            acc_ov[s.name].append(eng_ov.run_backtest(s, oos_data).returns)
            acc_nk[s.name].append(eng_nk.run_backtest(s, oos_data).returns)

    # Stitch OOS contiguo
    span_start = md.index[max(0, n - 1 - (N_FOLDS - 1) * FOLD - FOLD + 1)]
    bh = md["close"].pct_change().loc[span_start:]
    print(f"\nVentana OOS stitcheada: {span_start.date()} → {md.index[-1].date()} "
          f"({len(bh)} días)\n")

    print(f"{'Estrategia':24s} | {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>8s} "
          f"{'CAGR':>7s} | {'DD2008':>8s} {'DD2020':>8s}")
    print("-" * 86)

    rows = []
    for s in strategies:
        ov = pd.concat(acc_ov[s.name]).sort_index()
        ov = ov[~ov.index.duplicated(keep="last")]
        m = metrics(ov)
        d08 = dd_window(ov, "2008-01-01", "2009-06-30")
        d20 = dd_window(ov, "2020-01-01", "2020-06-30")
        rows.append((s.name, m, d08, d20))
        print(f"{s.name:24s} | {m['sharpe']:7.3f} {m['sortino']:8.3f} "
              f"{m['max_dd']*100:7.2f}% {m['cagr']*100:6.2f}% | "
              f"{d08*100:7.2f}% {d20*100:7.2f}%")

    mbh = metrics(bh)
    print("-" * 86)
    print(f"{'SPY Buy&Hold (desnudo)':24s} | {mbh['sharpe']:7.3f} {mbh['sortino']:8.3f} "
          f"{mbh['max_dd']*100:7.2f}% {mbh['cagr']*100:6.2f}% | "
          f"{dd_window(bh,'2008-01-01','2009-06-30')*100:7.2f}% "
          f"{dd_window(bh,'2020-01-01','2020-06-30')*100:7.2f}%")

    # XGBoost: overlay vs desnudo (efecto del Risk Overlay)
    print("\n[XGBoost: efecto del Risk Overlay]")
    for label, acc, eng in [("CON overlay", acc_ov, eng_ov), ("SIN overlay", acc_nk, eng_nk)]:
        r = pd.concat(acc["XGBoostTrend_d5"]).sort_index()
        r = r[~r.index.duplicated(keep="last")]
        m = metrics(r)
        print(f"  {label:12s} | Sharpe={m['sharpe']:.3f} | MaxDD={m['max_dd']*100:.2f}% "
              f"| CAGR={m['cagr']*100:.2f}%")

    # Guardar CSV
    out = pd.DataFrame([{
        "strategy": name, **m, "dd_2008": d08, "dd_2020": d20
    } for name, m, d08, d20 in rows])
    out.to_csv("oos_equity_metrics_2005_2024.csv", index=False)
    print("\n✓ Guardado: oos_equity_metrics_2005_2024.csv")


if __name__ == "__main__":
    main()
