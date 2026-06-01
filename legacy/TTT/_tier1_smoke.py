# _tier1_smoke.py — Validación end-to-end del motor cross-sectional (Tier 1)
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from panel_data_manager import PanelDataManager
from cross_sectional_engine import CrossSectionalEngine
from transaction_costs import TransactionCostModel
from statistical_validation import StatisticalSignificanceValidator

ANN = 252
UNIVERSE = ["XLK", "XLF", "XLV", "XLE", "XLU", "XLI", "XLP", "XLY", "XLB"]


def main():
    pdm = PanelDataManager(UNIVERSE).load("2005-01-01", "2024-12-31")
    close = pdm.wide("close")
    asset_ret = pdm.simple_returns()
    vol = pdm.realized_vol(window=21)

    # Score cross-sectional: momentum 12-1 (252d) — conocido en t, sin look-ahead.
    momentum = close.pct_change(252)

    val = StatisticalSignificanceValidator(periods_per_year=ANN)
    print(f"Panel: {pdm.n_assets} activos | {close.index[0].date()} → {close.index[-1].date()}")
    print(f"{'Config':28s} | {'Sharpe':>7s} {'MaxDD':>8s} {'CAGR':>7s} {'Turn':>6s} {'Lev':>5s} {'Net':>6s} {'PSR0':>6s}")
    print("-" * 92)

    for tag, kw in [
        ("L/S equal 20/20 lev2.0",  dict(weighting="equal")),
        ("L/S inv-vol 20/20 lev2.0", dict(weighting="inverse_vol")),
        ("L/S equal 30/30 lev1.0",  dict(long_pct=0.3, short_pct=0.3, leverage=1.0)),
    ]:
        eng = CrossSectionalEngine(
            cost_model=TransactionCostModel(commission_bps=0.5, slippage_bps=1.0), **kw
        )
        res = eng.run(scores=momentum, asset_returns=asset_ret, vol=vol)
        r = res.net_returns.dropna()
        eq = (1 + r).cumprod()
        sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(ANN)
        maxdd = (eq / eq.cummax() - 1).min()
        cagr = eq.iloc[-1] ** (ANN / len(r)) - 1
        psr0 = val.probabilistic_sharpe_ratio(r, 0.0)
        print(f"{tag:28s} | {sharpe:7.3f} {maxdd*100:7.2f}% {cagr*100:6.2f}% "
              f"{res.avg_turnover:6.3f} {res.avg_gross_leverage:5.2f} "
              f"{res.avg_net_exposure:+6.3f} {psr0:6.3f}")

    print("\n✓ Verificación de restricciones (config base):")
    eng = CrossSectionalEngine(cost_model=TransactionCostModel())
    res = eng.run(scores=momentum, asset_returns=asset_ret, vol=vol)
    w = res.weights.loc[res.weights.abs().sum(axis=1) > 0]
    print(f"  Σw (dollar-neutral, debe ≈0):   media={w.sum(axis=1).mean():+.2e}  max|.|={w.sum(axis=1).abs().max():.2e}")
    print(f"  Σ|w| (leverage, debe ≈2.0):     media={w.abs().sum(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
