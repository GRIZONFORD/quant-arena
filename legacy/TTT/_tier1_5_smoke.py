# _tier1_5_smoke.py — Validación del motor Beta-Neutral (Tier 1.5)
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from panel_data_manager import PanelDataManager
from cross_sectional_engine import CrossSectionalEngine
from transaction_costs import TransactionCostModel
from beta_neutralizer import BetaNeutralizer
from statistical_validation import StatisticalSignificanceValidator

ANN = 252
UNIVERSE = ["AAPL","MSFT","JNJ","JPM","XOM","PG","KO","PFE","WMT","CVX","HD",
            "INTC","CSCO","ORCL","MCD","DIS","BA","CAT","MMM","IBM","WFC","C",
            "VZ","T","MRK","ABT","PEP","COST","LOW","HON"]


def realized_beta(port: pd.Series, mkt: pd.Series) -> float:
    df = pd.concat([port, mkt], axis=1).dropna()
    if len(df) < 50:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


def stats(r: pd.Series) -> tuple:
    r = r.dropna()
    eq = (1 + r).cumprod()
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(ANN)
    maxdd = (eq / eq.cummax() - 1).min()
    return sharpe, maxdd


def main():
    pdm = PanelDataManager(UNIVERSE + ["SPY"]).load("2005-01-01", "2024-12-31")
    asset_ret_all = pdm.simple_returns()
    mkt = asset_ret_all["SPY"]
    asset_ret = asset_ret_all.drop(columns=["SPY"])
    close = pdm.wide("close").drop(columns=["SPY"])
    vol = pdm.realized_vol(21).drop(columns=["SPY"])

    bn = BetaNeutralizer(window=126, min_periods=60, lag=1)
    beta = bn.compute_rolling_betas(asset_ret, mkt)       # Date×Ticker, rezagada

    # Score: momentum residual (alpha sin mercado) 126d sobre retornos residuales.
    resid = bn.residualize(asset_ret, mkt, betas=beta)
    score = (1.0 + resid).rolling(126, min_periods=60).apply(np.prod, raw=True) - 1.0

    cm = TransactionCostModel(commission_bps=0.5, slippage_bps=1.0)
    val = StatisticalSignificanceValidator(periods_per_year=ANN)

    print(f"Panel: {pdm.n_assets-1} activos + SPY | {close.index[0].date()} → {close.index[-1].date()}\n")
    print(f"{'Motor':26s} | {'Sharpe':>7s} {'MaxDD':>8s} | {'βport(real)':>11s} {'Σwβ|avg|':>9s} {'Σw|avg|':>8s} {'Turn':>6s}")
    print("-" * 92)

    for tag, bneutral in [("Dollar-Neutral (Tier 1)", False),
                          ("Beta-Neutral (Tier 1.5)", True)]:
        eng = CrossSectionalEngine(
            cost_model=cm, long_pct=0.2, short_pct=0.2, leverage=2.0,
            weighting="inverse_vol",
            beta_neutralizer=bn if bneutral else None,
        )
        res = eng.run(scores=score, asset_returns=asset_ret, vol=vol, beta=beta)
        sharpe, maxdd = stats(res.net_returns)
        rbeta = realized_beta(res.net_returns, mkt)
        sw_beta = BetaNeutralizer.portfolio_beta(res.weights, beta).abs().mean()
        print(f"{tag:26s} | {sharpe:7.3f} {maxdd*100:7.2f}% | {rbeta:11.4f} "
              f"{sw_beta:9.4f} {res.avg_net_exposure:+8.1e} {res.avg_turnover:6.3f}")

    print("\n✓ Restricciones (Beta-Neutral, fechas con cesta plena):")
    eng = CrossSectionalEngine(cost_model=cm, weighting="inverse_vol", beta_neutralizer=bn)
    res = eng.run(scores=score, asset_returns=asset_ret, vol=vol, beta=beta)
    w = res.weights.loc[res.weights.abs().sum(axis=1) > 1e-9]
    swb = BetaNeutralizer.portfolio_beta(w, beta).loc[w.index]
    print(f"  Σw   (dollar-neutral, ≈0): max|.| = {w.sum(axis=1).abs().max():.2e}")
    print(f"  Σwβ  (beta-neutral, ≈0):   media|.| = {swb.abs().mean():.2e}  max|.| = {swb.abs().max():.2e}")
    print(f"  Σ|w| (leverage, ≈2.0):     media = {w.abs().sum(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
