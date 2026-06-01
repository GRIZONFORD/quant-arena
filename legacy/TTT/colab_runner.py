# %% [markdown]
"""
# Quant Arena — Colab Runner v1.0
### Walk-Forward Engine + TrueSkill Through Time (Continuo + Calibración Bayesiana)

**Instrucciones de uso en Google Colab:**
1. Sube todos los módulos del proyecto al directorio `/content/`:
   `walk_forward_engine.py`, `data_manager.py`, `regime_detector.py`,
   `strategy_zoo.py`, `strategy_adapter.py`, `backtest_engine.py`,
   `ranking.py`, `utils.py`
2. Ejecuta las celdas en orden.
3. La **Parte A** (celdas 1–6) es completamente autónoma: no necesita los módulos
   del proyecto. Demuestra TTT puro con yfinance.
4. La **Parte B** (celdas 7–8) arranca el `WalkForwardELOEngine` completo y requiere
   todos los módulos.

**Referencia:**
Landfried, G. & Mocskos, E. (2024). TrueSkill Through Time.
github.com/glandfried/TrueSkillThroughTime.py
"""

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 1: Instalación de dependencias
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys

_PACKAGES = [
    "trueskillthroughtime",
    "yfinance",
    "matplotlib",
    "seaborn",
    "pandas",
    "numpy",
    "scipy",
]

print("Instalando dependencias…")
for pkg in _PACKAGES:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", pkg]
    )
print("✓ Instalación completada.")

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 2: Descarga de datos (SPY, 10 años) via yfinance
#          → DataFrame con columnas en minúsculas listo para el engine
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import numpy as np

SYMBOL     = "SPY"
START_DATE = "2014-01-01"
END_DATE   = "2024-01-01"

print(f"Descargando {SYMBOL} ({START_DATE} → {END_DATE})…")
raw = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Normalizar al contrato del engine: columnas en minúsculas, índice DatetimeIndex
market_data = raw.copy()
market_data.columns = [c.lower() for c in market_data.columns]
market_data.index = pd.to_datetime(market_data.index)
market_data = market_data[["open", "high", "low", "close", "volume"]].dropna()

print(f"✓ {len(market_data):,} días de trading | "
      f"{market_data.index[0].date()} → {market_data.index[-1].date()}")
print(market_data.tail(3).to_string())

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 3: Estrategia de prueba — SMA 50/200 Crossover
#          Genera retornos diarios de estrategia y benchmark
# ─────────────────────────────────────────────────────────────────────────────

prices = market_data["close"]

# ── Señales (sin look-ahead: shift(1)) ───────────────────────────────────────
sma_fast = prices.rolling(50).mean()
sma_slow = prices.rolling(200).mean()
signal   = (sma_fast > sma_slow).astype(float).shift(1)   # 1 = long, 0 = flat

# ── Retornos diarios ──────────────────────────────────────────────────────────
bench_returns  = prices.pct_change()                        # Buy & Hold SPY
strat_returns  = signal * bench_returns                     # SMA strategy

# Eliminar NaN del warmup (200 días)
first_valid = max(strat_returns.first_valid_index(),
                  bench_returns.first_valid_index())
bench_returns = bench_returns.loc[first_valid:].fillna(0.0)
strat_returns = strat_returns.loc[first_valid:].fillna(0.0)

print(f"Retorno total SPY  B&H : {(1 + bench_returns).prod() - 1:.2%}")
print(f"Retorno total SMA 50/200: {(1 + strat_returns).prod() - 1:.2%}")
print(f"Días de trading con señal: {signal.loc[first_valid:].sum():.0f} / {len(signal.loc[first_valid:])}")

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 4: Acumulador TTT autónomo + Calibración bayesiana de (gamma, sigma)
#          ── Replica TTTAccumulator del engine, sin dependencias externas ──
# ─────────────────────────────────────────────────────────────────────────────

from itertools import product as iproduct
from typing import Dict, List, Optional, Tuple
from trueskillthroughtime import History

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("TTT-Colab")

# ────────────────────────────────────────────────────────────────────────────
# Clase autónoma (espejo de TTTAccumulator sin importar el engine completo)
# ────────────────────────────────────────────────────────────────────────────

class StandaloneTTTAccumulator:
    """
    Versión standalone de TTTAccumulator para uso en Colab sin dependencias
    del proyecto. Idéntica lógica: retorno compuesto + obs='Continuous'.
    """

    def __init__(self, ttt_gamma: float = 0.03, ttt_sigma: float = 1.0):
        self.ttt_gamma = ttt_gamma
        self.ttt_sigma = ttt_sigma
        self.composition: List = []
        self.results:     List = []
        self.times:       List = []
        self.obs:         List = []
        self.strategy_names: set = set()

    def add_fold(
        self,
        strategy_name: str,
        strategy_oos_returns: pd.Series,
        benchmark_oos_returns: pd.Series,
        test_start: pd.Timestamp,
    ) -> None:
        """Retorno compuesto del fold: prod(1+r_t) - 1."""
        strat_ret = float((1.0 + strategy_oos_returns).prod() - 1.0)
        bench_ret = float((1.0 + benchmark_oos_returns).prod() - 1.0)
        t_days    = float(test_start.timestamp() / 86_400)

        self.composition.append([[strategy_name], ["_benchmark_"]])
        self.results.append([strat_ret, bench_ret])
        self.obs.append("Continuous")
        self.times.append(t_days)
        self.strategy_names.add(strategy_name)

    def _build_history(self, gamma: float, sigma: float) -> History:
        return History(
            composition=self.composition,
            results=self.results,
            times=self.times,
            mu=0.0,
            sigma=sigma,
            gamma=gamma,
            obs=self.obs,
        )

    def calibrate(
        self,
        grid_gamma: List[float] = None,
        grid_sigma: List[float] = None,
        calib_iterations: int = 3,
        calib_epsilon:    float = 0.01,
    ) -> Dict:
        """
        Grid search sobre (gamma, sigma) maximizando h.geometric_mean().

        Criterio: Log-Evidencia Marginal — exp(mean(log P(result_t | history_<t))).
        Convergencia relajada durante calibración para eficiencia computacional.
        """
        if grid_gamma is None:
            grid_gamma = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        if grid_sigma is None:
            grid_sigma = [0.5, 1.0, 1.5, 2.0]

        best_score = -float("inf")
        best_gamma = self.ttt_gamma
        best_sigma = self.ttt_sigma
        all_results: List[Tuple] = []

        n = len(grid_gamma) * len(grid_sigma)
        log.info(f"Calibración: {n} combinaciones | gamma={grid_gamma} | sigma={grid_sigma}")

        for gamma, sigma in iproduct(grid_gamma, grid_sigma):
            try:
                h = self._build_history(gamma, sigma)
                h.convergence(iterations=calib_iterations,
                               epsilon=calib_epsilon, verbose=False)
                score = h.geometric_mean()
                all_results.append((gamma, sigma, score))

                if score > best_score:
                    best_score, best_gamma, best_sigma = score, gamma, sigma

                log.debug(f"  γ={gamma:.4f} σ={sigma:.4f} → gm={score:.6f}")
            except Exception as exc:
                log.debug(f"  γ={gamma:.4f} σ={sigma:.4f} → error: {exc}")

        self.ttt_gamma, self.ttt_sigma = best_gamma, best_sigma
        log.info(f"Mejor: γ={best_gamma:.4f} σ={best_sigma:.4f} gm={best_score:.6f}")

        return {
            "gamma": best_gamma, "sigma": best_sigma,
            "geometric_mean": best_score, "all_results": all_results,
        }

    def fit(self, iterations: int = 6, epsilon: float = 1e-3,
            auto_calibrate: bool = True) -> History:
        """Ajuste final (estricto). Si auto_calibrate=True llama calibrate() antes."""
        if auto_calibrate:
            self.calibrate()

        h = self._build_history(self.ttt_gamma, self.ttt_sigma)
        step, n_iter = h.convergence(iterations=iterations, epsilon=epsilon, verbose=False)

        if step > epsilon:
            log.warning(
                f"TTT no convergió en {n_iter} iteraciones "
                f"(step={step:.6f} > epsilon={epsilon:.6f})."
            )
        return h


# ────────────────────────────────────────────────────────────────────────────
# Generación de folds OOS con ventana rolling de 21 días
# ────────────────────────────────────────────────────────────────────────────

FOLD_DAYS   = 21       # ventana OOS por fold
MIN_TRAIN   = 200      # warmup mínimo antes del primer fold

dates = strat_returns.index

acc = StandaloneTTTAccumulator(ttt_gamma=0.03, ttt_sigma=1.0)

fold_count = 0
for fold_start_idx in range(MIN_TRAIN, len(dates) - FOLD_DAYS, FOLD_DAYS):
    fold_end_idx  = fold_start_idx + FOLD_DAYS
    fold_dates    = dates[fold_start_idx:fold_end_idx]
    fold_strat    = strat_returns.iloc[fold_start_idx:fold_end_idx]
    fold_bench    = bench_returns.iloc[fold_start_idx:fold_end_idx]

    if len(fold_dates) < FOLD_DAYS:
        break

    acc.add_fold(
        strategy_name         = "SMA_50_200",
        strategy_oos_returns  = fold_strat,
        benchmark_oos_returns = fold_bench,
        test_start            = fold_dates[0],
    )
    fold_count += 1

print(f"✓ {fold_count} folds OOS acumulados en el TTTAccumulator.")

# Calibración bayesiana + ajuste final
print("\nEjecutando calibración bayesiana de (gamma, sigma)…")
calib_result = acc.calibrate(
    grid_gamma=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
    grid_sigma=[0.5, 1.0, 1.5, 2.0],
    calib_iterations=3,
    calib_epsilon=0.01,
)

print(f"\nMejor combinación encontrada:")
print(f"  gamma           = {calib_result['gamma']:.4f}")
print(f"  sigma           = {calib_result['sigma']:.4f}")
print(f"  geometric_mean  = {calib_result['geometric_mean']:.6f}")

print("\nAjuste final (convergencia estricta)…")
h_final = acc.fit(iterations=6, epsilon=1e-3, auto_calibrate=False)
print("✓ Modelo TTT ajustado.")

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 5: Extraer learning curves para visualización
# ─────────────────────────────────────────────────────────────────────────────

lc_raw = h_final.learning_curves()

def extract_curve(lc_raw: dict, name: str) -> pd.DataFrame:
    """Convierte la curva TTT a DataFrame con columnas date / mu / sigma."""
    entries = lc_raw.get(name, [])
    if not entries:
        return pd.DataFrame()
    times_d  = [tp[0] for tp in entries]
    mus      = [tp[1].mu    for tp in entries]
    sigmas   = [tp[1].sigma for tp in entries]
    dates_dt = pd.to_datetime(
        [t * 86_400 for t in times_d], unit="s", utc=True
    ).tz_localize(None)
    return pd.DataFrame({"date": dates_dt, "mu": mus, "sigma": sigmas})

curve_sma   = extract_curve(lc_raw, "SMA_50_200")
curve_bench = extract_curve(lc_raw, "_benchmark_")

print(f"Curva SMA_50_200 : {len(curve_sma)} puntos | "
      f"mu_final={curve_sma['mu'].iloc[-1]:.4f} | "
      f"sigma_final={curve_sma['sigma'].iloc[-1]:.4f}")

print(f"Curva benchmark  : {len(curve_bench)} puntos | "
      f"mu_final={curve_bench['mu'].iloc[-1]:.4f}")

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 6: Visualizaciones institucionales
#   Fig 1 – Learning curves TTT (mu ± 2σ)
#   Fig 2 – Heatmap calibración bayesiana (gamma vs sigma)
#   Fig 3 – Retorno compuesto acumulado: estrategia vs benchmark
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PALETTE = {"strategy": "#2563EB", "benchmark": "#DC2626", "band": "#93C5FD"}
sns.set_theme(style="darkgrid", font_scale=1.1)
fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    "Quant Arena — TrueSkill Through Time (obs=Continuous)\n"
    f"SPY SMA 50/200 Crossover | {START_DATE} → {END_DATE}",
    fontsize=15, fontweight="bold", y=0.98
)

# ── Fig 1: Learning curves ──────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 2, (1, 2))

if not curve_sma.empty:
    mu_s, sg_s, dt_s = curve_sma["mu"], curve_sma["sigma"], curve_sma["date"]
    ax1.plot(dt_s, mu_s, color=PALETTE["strategy"], lw=2,
             label="SMA 50/200 — μ(t)")
    ax1.fill_between(dt_s,
                     mu_s - 2 * sg_s,
                     mu_s + 2 * sg_s,
                     color=PALETTE["band"], alpha=0.35, label="±2σ")

if not curve_bench.empty:
    ax1.plot(curve_bench["date"], curve_bench["mu"],
             color=PALETTE["benchmark"], lw=1.5, ls="--",
             label="Benchmark (_benchmark_) — μ(t)")

ax1.axhline(0, color="gray", lw=0.8, ls=":")
ax1.set_title("Learning Curves TTT — Habilidad Latente μ(t) con Bandas ±2σ",
              fontweight="bold")
ax1.set_ylabel("μ — Habilidad TTT")
ax1.legend(loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# Anotar el mu final
if not curve_sma.empty:
    ax1.annotate(
        f"μ_final = {mu_s.iloc[-1]:.4f}",
        xy=(dt_s.iloc[-1], mu_s.iloc[-1]),
        xytext=(-80, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=PALETTE["strategy"]),
        color=PALETTE["strategy"], fontsize=10,
    )

# ── Fig 2: Heatmap calibración ──────────────────────────────────────────────
ax2 = fig.add_subplot(3, 2, 3)

if calib_result["all_results"]:
    hm_df = pd.DataFrame(
        calib_result["all_results"], columns=["gamma", "sigma", "geometric_mean"]
    )
    hm_pivot = hm_df.pivot(index="sigma", columns="gamma", values="geometric_mean")
    # Ordenar sigma descendente para que el eje Y sea intuitivo
    hm_pivot = hm_pivot.sort_index(ascending=False)

    sns.heatmap(
        hm_pivot,
        ax=ax2,
        annot=True, fmt=".4f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "geometric_mean (log-evidencia)"},
    )
    # Marcar la celda óptima
    best_row = hm_pivot.index.get_loc(calib_result["sigma"])
    best_col = hm_pivot.columns.get_loc(calib_result["gamma"])
    ax2.add_patch(plt.Rectangle(
        (best_col, best_row), 1, 1,
        fill=False, edgecolor="#1D4ED8", lw=3, label="Óptimo"
    ))
    ax2.set_title(
        f"Calibración Bayesiana — geometric_mean\n"
        f"Óptimo: γ={calib_result['gamma']:.4f}, σ={calib_result['sigma']:.4f}",
        fontweight="bold"
    )
    ax2.set_xlabel("gamma (volatilidad de habilidad)")
    ax2.set_ylabel("sigma (incertidumbre prior)")

# ── Fig 3: Curva de riqueza acumulada ───────────────────────────────────────
ax3 = fig.add_subplot(3, 2, 4)

cum_strat = (1 + strat_returns).cumprod()
cum_bench = (1 + bench_returns).cumprod()

ax3.plot(cum_strat.index, cum_strat, color=PALETTE["strategy"],
         lw=2, label="SMA 50/200")
ax3.plot(cum_bench.index, cum_bench, color=PALETTE["benchmark"],
         lw=1.5, ls="--", label="SPY B&H")
ax3.set_title("Wealth Index — Retorno Compuesto Acumulado", fontweight="bold")
ax3.set_ylabel("Crecimiento de $1")
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── Fig 4: Distribución del Alpha por fold ──────────────────────────────────
ax4 = fig.add_subplot(3, 2, 5)

fold_alphas = [
    r[0] - r[1]                    # strat_ret - bench_ret por fold
    for r in acc.results
]
ax4.hist(fold_alphas, bins=30, color=PALETTE["strategy"],
         edgecolor="white", alpha=0.85)
ax4.axvline(0, color=PALETTE["benchmark"], lw=1.5, ls="--", label="α=0")
ax4.axvline(np.mean(fold_alphas), color="gold", lw=2,
            label=f"Media α={np.mean(fold_alphas):.4f}")
ax4.set_title("Distribución del Alpha por Fold OOS\n(strat_ret − bench_ret)",
              fontweight="bold")
ax4.set_xlabel("Alpha compuesto del fold")
ax4.set_ylabel("Frecuencia")
ax4.legend()

# ── Fig 5: Evolución de sigma (incertidumbre TTT) ───────────────────────────
ax5 = fig.add_subplot(3, 2, 6)

if not curve_sma.empty:
    ax5.plot(curve_sma["date"], curve_sma["sigma"],
             color="#7C3AED", lw=2, label="σ(t) — SMA 50/200")
    ax5.fill_between(curve_sma["date"],
                     0, curve_sma["sigma"],
                     color="#7C3AED", alpha=0.2)
    ax5.set_title("Incertidumbre TTT — σ(t) a lo largo del tiempo",
                  fontweight="bold")
    ax5.set_ylabel("σ — Incertidumbre de habilidad")
    ax5.set_xlabel("Fecha")
    ax5.legend()
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("quant_arena_ttt_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Dashboard guardado en quant_arena_ttt_dashboard.png")

# %% [markdown]
"""
---
## PARTE B — Motor Completo `WalkForwardELOEngine`

**Prerrequisitos:** Los siguientes archivos deben estar en `/content/`:
- `walk_forward_engine.py`
- `data_manager.py`
- `regime_detector.py`
- `strategy_zoo.py`
- `strategy_adapter.py`
- `backtest_engine.py`
- `ranking.py`
- `utils.py`

Para subirlos desde tu PC usa la celda siguiente o el panel lateral de Colab
(ícono de carpeta → botón "Upload").
"""

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 7: Upload interactivo de módulos (solo en Colab)
#          Si ya los subiste manualmente, puedes saltar esta celda.
# ─────────────────────────────────────────────────────────────────────────────

import sys, os

_IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if _IN_COLAB:
    try:
        from google.colab import files as _colab_files
        print("Sube los módulos del proyecto (selección múltiple con Ctrl+Click):")
        _uploaded = _colab_files.upload()
        for fname in _uploaded:
            print(f"  ✓ {fname} subido ({len(_uploaded[fname])/1024:.1f} KB)")
    except Exception as _e:
        print(f"Saltar upload interactivo: {_e}")
else:
    print("Entorno local detectado — asegúrate de que los módulos estén en el PATH.")

# %% ─────────────────────────────────────────────────────────────────────────
# CELDA 8: Ejecución del WalkForwardELOEngine completo + visualización
# ─────────────────────────────────────────────────────────────────────────────

# Añadir /content al path para Colab
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

_REQUIRED_MODULES = [
    "walk_forward_engine", "data_manager", "regime_detector",
    "strategy_zoo", "strategy_adapter", "backtest_engine",
    "ranking", "utils",
]

def _check_modules() -> bool:
    missing = []
    for mod in _REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[ERROR] Módulos faltantes: {missing}")
        print("Sube los archivos y vuelve a ejecutar esta celda.")
        return False
    return True


if _check_modules():
    from walk_forward_engine import WalkForwardELOEngine, compute_elo_predictive_power
    import json

    print("=" * 80)
    print("Iniciando WalkForwardELOEngine v3.0 — TTT Continuo + Calibración Bayesiana")
    print("=" * 80)

    engine = WalkForwardELOEngine(
        n_splits           = 10,       # Reducido para demo rápida en Colab
        min_train_days     = 252,
        test_days          = 21,
        purge_days         = 5,
        embargo_days       = 10,
        use_cpcv           = False,
        fitness_function   = "calmar",
        ttt_gamma          = 0.03,
        ttt_sigma          = 1.0,
        ttt_iterations     = 6,
        ttt_auto_calibrate = True,     # ← Grid search bayesiano activo
    )

    results_df = engine.run_walk_forward_validation(
        symbol          = SYMBOL,
        start_date      = START_DATE,
        end_date        = END_DATE,
        initial_capital = 100_000,
    )

    stats = compute_elo_predictive_power(results_df)

    results_df.to_csv("walk_forward_results.csv", index=False)
    with open("elo_stats.json", "w") as _f:
        json.dump(stats, _f, indent=2)
    print("\n✓ Resultados → walk_forward_results.csv | Estadísticas → elo_stats.json")

    # ── Visualización del output del engine completo ─────────────────────────
    if hasattr(engine, "ttt_curves_") and engine.ttt_curves_:

        fig2, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=False)
        fig2.suptitle(
            "WalkForwardELOEngine — TTT Learning Curves (todas las estrategias)",
            fontsize=14, fontweight="bold"
        )

        ax_mu, ax_sigma = axes

        # Colores automáticos por estrategia
        cmap = plt.cm.get_cmap("tab20", len(engine.ttt_curves_))

        for idx, (strat_name, curve_df) in enumerate(engine.ttt_curves_.items()):
            if curve_df.empty or strat_name == "_benchmark_":
                continue
            color = cmap(idx)
            ax_mu.plot(
                curve_df["time_date"], curve_df["ttt_mu"],
                lw=1.8, color=color, label=strat_name, alpha=0.85
            )
            ax_mu.fill_between(
                curve_df["time_date"],
                curve_df["ttt_mu"] - 2 * curve_df["ttt_sigma"],
                curve_df["ttt_mu"] + 2 * curve_df["ttt_sigma"],
                color=color, alpha=0.08
            )
            ax_sigma.plot(
                curve_df["time_date"], curve_df["ttt_sigma"],
                lw=1.5, color=color, label=strat_name, alpha=0.85
            )

        ax_mu.axhline(0, color="gray", lw=0.8, ls=":")
        ax_mu.set_title("Habilidad latente μ(t) — todas las estrategias", fontweight="bold")
        ax_mu.set_ylabel("μ TTT")
        ax_mu.legend(ncol=3, fontsize=8, loc="upper left")
        ax_mu.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax_sigma.set_title("Incertidumbre σ(t) — todas las estrategias", fontweight="bold")
        ax_sigma.set_ylabel("σ TTT")
        ax_sigma.legend(ncol=3, fontsize=8, loc="upper left")
        ax_sigma.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        plt.savefig("engine_ttt_all_strategies.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✓ Dashboard engine → engine_ttt_all_strategies.png")

        # ── Tabla resumen TTT: ranking final por mu ──────────────────────────
        final_mus = {
            name: df["ttt_mu"].iloc[-1]
            for name, df in engine.ttt_curves_.items()
            if not df.empty
        }
        final_sigmas = {
            name: df["ttt_sigma"].iloc[-1]
            for name, df in engine.ttt_curves_.items()
            if not df.empty
        }
        ranking_df = pd.DataFrame({
            "strategy":  list(final_mus.keys()),
            "ttt_mu":    list(final_mus.values()),
            "ttt_sigma": [final_sigmas[k] for k in final_mus],
        }).sort_values("ttt_mu", ascending=False).reset_index(drop=True)
        ranking_df.index += 1   # Rango 1-based

        print("\nRanking final TTT (por μ_final):")
        print(ranking_df.to_string())

    # ── Calibración bayesiana del engine ────────────────────────────────────
    _acc = engine._ttt
    if _acc.ttt_gamma and _acc.ttt_sigma:
        print(f"\n[Calibración engine] γ={_acc.ttt_gamma:.4f} | σ={_acc.ttt_sigma:.4f}")

else:
    print("\nParte B omitida — sube los módulos del proyecto y vuelve a ejecutar.")

# %% [markdown]
"""
---
## Resumen de outputs generados

| Archivo | Contenido |
|---|---|
| `quant_arena_ttt_dashboard.png` | Dashboard TTT standalone (Parte A) |
| `engine_ttt_all_strategies.png` | Learning curves de todas las estrategias del engine |
| `walk_forward_results.csv`      | DataFrame completo de métricas por split |
| `elo_stats.json`                | Estadísticas de poder predictivo del ELO |

### Interpretación rápida de las visualizaciones

**Learning curve μ(t):** Un μ > 0 sostenido indica que la estrategia genera Alpha
consistente frente al benchmark. La banda ±2σ muestra la incertidumbre del modelo
bayesiano — se estrecha a medida que acumula más partidos OOS.

**Heatmap de calibración:** La celda marcada en azul es el par (γ, σ) óptimo según
la Log-Evidencia Marginal `h.geometric_mean()`. Un γ alto implica que el modelo
espera cambios rápidos de régimen; un γ bajo, una habilidad más estacionaria.

**Distribución del Alpha por fold:** Muestra si el exceso de retorno es estable
o ruidoso. Una media positiva con baja varianza indica Alpha estructural.
"""
