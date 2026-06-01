"""
run_ttt_analysis.py — Ejemplo de Invocación del Sistema TTT
=============================================================
Script de ejemplo que conecta todos los módulos del ecosistema:

  strategy_zoo.py      → Universo de algoritmos con Meta-Labeling
  strategy_adapter.py  → Adaptador con costos y signal decay
  data_manager.py      → Carga y segmentación temporal (purging+embargo)
  simulation_engine.py → Motor de torneo estocástico TTT
  analyzer.py          → Interpretación bayesiana y dashboard

Dos modos de uso:
  1. MODO STANDALONE: solo simulation_engine + analyzer (sin walk_forward_engine)
  2. MODO INTEGRADO: alimenta el TTTAccumulator del WalkForwardELOEngine

Ejecutar:
  python run_ttt_analysis.py
"""

import logging
import os
import pandas as pd
import numpy as np

logging.basicConfig(
    level   = logging.WARNING,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_ttt_analysis")


# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — STANDALONE: TTTSimulator directo
# ══════════════════════════════════════════════════════════════════════════════

def run_standalone_tournament(
    symbol:         str   = "SPY",
    start:          str   = "2010-01-01",
    end:            str   = "2024-01-01",
    n_folds:        int   = 12,
    fold_size_days: int   = 21,
    min_is_days:    int   = 252,
    ttt_gamma:      float = 0.03,
    ttt_sigma:      float = 1.0,
    score_metric:   str   = "sharpe",   # Juez TTT por Sharpe OOS (no hit_rate/beta)
    output_html:    str   = "ttt_dashboard.html",
) -> dict:
    """
    Pipeline completo end-to-end del torneo TTT en modo standalone.

    PASOS:
    1. Cargar datos (Yahoo Finance o datos sintéticos si yfinance no está).
    2. Construir universo de algoritmos con StrategyRegistry.
    3. Ejecutar torneo con TTTSimulator.
    4. Analizar resultados con run_full_analysis().
    5. Guardar dashboard HTML.

    Returns:
        Dict con curvas TTT y DataFrames de análisis.
    """
    print("\n" + "═" * 72)
    print("  SISTEMA DE EVALUACIÓN BAYESIANA — TrueSkill Through Time")
    print("  Landfried & Mocskos (2024) | López de Prado (2018)")
    print("═" * 72)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print(f"\n[1/4] Cargando datos: {symbol} [{start} → {end}]...")
    market_data = _load_market_data(symbol, start, end)
    print(f"      {len(market_data)} períodos | "
          f"{market_data.index[0].date()} → {market_data.index[-1].date()}")

    # ── 2. Construir universo de algoritmos ───────────────────────────────────
    print("\n[2/4] Construyendo universo de algoritmos...")
    from strategy_zoo import StrategyRegistry
    registry   = StrategyRegistry().create_default_universe()
    strategies = list(registry.strategies.values())
    print(f"      {len(strategies)} algoritmos registrados:")
    for s in strategies:
        print(f"       · {s.name:40s} [{s.strategy_type}]  halflife={s.signal_halflife}d")

    # ── 3. Ejecutar torneo TTT ────────────────────────────────────────────────
    print(f"\n[3/4] Iniciando torneo TTT ({n_folds} folds × {len(strategies)} algos)...")
    from simulation_engine import TTTSimulator
    simulator = TTTSimulator(
        algorithms    = strategies,
        ttt_gamma     = ttt_gamma,
        ttt_sigma     = ttt_sigma,
        ttt_mu        = 0.0,
        score_metric  = score_metric,
        purge_days    = 5,
        embargo_days  = 10,
        random_seed   = 42,
    )
    simulator.run_tournament(
        market_data    = market_data,
        n_folds        = n_folds,
        min_is_days    = min_is_days,
        fold_size_days = fold_size_days,
        verbose        = True,
    )

    curves = simulator.get_curves()
    if not curves:
        print("\n⚠ TTT no produjo curvas. Verificar instalación: pip install trueskillthroughtime")
        return {}

    # ── 4. Análisis completo ──────────────────────────────────────────────────
    print(f"\n[4/4] Ejecutando análisis bayesiano...")
    from analyzer import run_full_analysis
    analysis = run_full_analysis(
        curves             = curves,
        output_html        = output_html,
        verbose            = True,
        k_conservative     = 3.0,
        dominance_threshold= 0.80,
    )

    print(f"\n✓ Dashboard guardado: {output_html}")

    # Exportar CSVs
    for key, df in analysis.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            fname = f"ttt_{key}.csv"
            df.to_csv(fname, index=False)
    print("✓ CSVs exportados: ttt_skill_summary.csv, ttt_stability.csv, etc.")

    # Exportar learning curves individuales por algoritmo
    curves_dir = "ttt_curves"
    os.makedirs(curves_dir, exist_ok=True)
    for name, df in curves.items():
        safe = name.replace(" ", "_").replace("/", "_")
        df.to_csv(f"{curves_dir}/curve_{safe}.csv", index=False)
    print(f"✓ Curvas TTT en: {curves_dir}/")

    return {"curves": curves, "analysis": analysis, "simulator": simulator}


# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — INTEGRADO: Alimentar TTTAccumulator del WalkForwardELOEngine
# ══════════════════════════════════════════════════════════════════════════════

def run_integrated_with_engine(engine) -> dict:
    """
    Conecta el TTTSimulator al WalkForwardELOEngine existente.

    Después de que el motor principal ejecuta run_walk_forward_validation(),
    esta función:
    1. Extrae las ttt_curves_ del motor.
    2. Las analiza con run_full_analysis().
    3. Exporta el dashboard y los DataFrames.

    Uso típico desde el script principal del motor:

        engine = WalkForwardELOEngine(...)
        results_df = engine.run_walk_forward_validation('SPY', ...)

        # Análisis TTT de las curvas producidas por el motor
        from run_ttt_analysis import run_integrated_with_engine
        ttt_analysis = run_integrated_with_engine(engine)

    Args:
        engine: Instancia de WalkForwardELOEngine post-ejecución.

    Returns:
        Dict con DataFrames de análisis y path del dashboard.
    """
    curves = getattr(engine, "ttt_curves_", {})
    if not curves:
        print("⚠ engine.ttt_curves_ vacío. "
              "Verificar que TTT esté instalado y el motor haya convergido.")
        return {}

    print(f"\n[TTT Integrado] Analizando {len(curves)} curvas del WalkForwardELOEngine...")

    from analyzer import run_full_analysis
    analysis = run_full_analysis(
        curves          = curves,
        output_html     = "ttt_engine_dashboard.html",
        verbose         = True,
        k_conservative  = 3.0,
        dominance_threshold = 0.80,
    )
    return {"curves": curves, "analysis": analysis}


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS — Fallback a datos sintéticos si yfinance no está
# ══════════════════════════════════════════════════════════════════════════════

FEATURES_PARQUET = "sp500_daily_1997_to_today_features.parquet"


def _load_market_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Carga datos de mercado para el torneo, con prioridad al dataset ENRIQUECIDO.

    ORDEN DE PREFERENCIA:
      1. `sp500_daily_1997_to_today_features.parquet` (OHLCV + 32 features) →
         es lo que consume XGBoostTrendStrategy. Se filtra al rango [start, end].
      2. yfinance (OHLCV crudo) — las estrategias ML quedarán planas sin features.
      3. Datos sintéticos — SOLO último recurso (no válido para backtest real).
    """
    import os
    for path in (FEATURES_PARQUET, os.path.join("..", FEATURES_PARQUET)):
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            df = df.sort_index().loc[start:end]
            logger.warning(
                f"Dataset ENRIQUECIDO cargado: {path} | {len(df)} días | "
                f"{df.shape[1]} columnas (OHLCV + features)."
            )
            return df

    logger.warning(
        f"'{FEATURES_PARQUET}' no encontrado — recurriendo a yfinance "
        "(las estrategias ML quedarán planas sin features)."
    )
    try:
        import yfinance as yf
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            raw = yf.download(symbol, start=start, end=end,
                              auto_adjust=False, progress=False)
        if raw.empty:
            raise ValueError("Descarga vacía")

        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        if "adj_close" not in raw.columns and "adj close" in raw.columns:
            raw = raw.rename(columns={"adj close": "adj_close"})
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "date"
        return raw.dropna(how="all").sort_index()

    except Exception as exc:
        logger.warning(f"yfinance falló ({exc}) — usando datos sintéticos.")
        return _synthetic_market_data(start, end)


def _synthetic_market_data(start: str, end: str) -> pd.DataFrame:
    """
    Genera datos OHLCV sintéticos con 3 regímenes para pruebas.
    Sin dependencias externas.
    """
    np.random.seed(2024)
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)

    # 3 regímenes: bull calmo, crash, lateral
    t1, t2 = n // 3, 2 * n // 3
    ret = np.concatenate([
        np.random.randn(t1)         * 0.008 + 0.0004,   # bull calmo
        np.random.randn(t2 - t1)   * 0.020 - 0.0008,   # crash
        np.random.randn(n - t2)    * 0.010,             # lateral
    ])
    close = 100.0 * np.exp(np.cumsum(ret))

    return pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.003),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.008),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.008),
        "close":  close,
        "adj_close": close,
        "volume": np.random.randint(int(1e6), int(5e6), n).astype(float),
    }, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Configurar nivel de log para ver mensajes TTT ─────────────────────────
    logging.getLogger("simulation_engine").setLevel(logging.INFO)
    logging.getLogger("analyzer").setLevel(logging.INFO)

    # ── Modo 1: Torneo standalone ─────────────────────────────────────────────
    output = run_standalone_tournament(
        symbol         = "SPY",
        start          = "2010-01-01",
        end            = "2024-01-01",
        n_folds        = 12,
        fold_size_days = 21,       # ~1 mes por fold OOS
        min_is_days    = 252,      # ~1 año mínimo IS
        ttt_gamma      = 0.03,     # Volatilidad de habilidad latente
        ttt_sigma      = 1.0,      # Prior no informativo
        score_metric   = "sharpe", # Juez TTT por Sharpe OOS (riesgo-ajustado)
        output_html    = "ttt_dashboard.html",
    )

    if output:
        sim = output["simulator"]
        print("\n[Modo 2 - Integración con WalkForwardELOEngine]")
        print("  Para integrar con el motor principal:")
        print("  from run_ttt_analysis import run_integrated_with_engine")
        print("  ttt_analysis = run_integrated_with_engine(engine)")
        print()
        print("  Los datos TTT del simulador también se pueden exportar")
        print("  al TTTAccumulator del motor:")
        comp, res, times, obs = sim.export_ttt_data()
        print(f"  sim.export_ttt_data() → {len(comp)} partidos listos para TTTAccumulator")
