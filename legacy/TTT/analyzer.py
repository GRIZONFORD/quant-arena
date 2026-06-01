# analyzer.py
"""
Analyzer v1.0 — Interpretación Bayesiana de Habilidades TTT
=============================================================
Módulo de análisis e interpretación post-inferencia para el motor
TrueSkill Through Time (Landfried & Mocskos, 2024).

PERSPECTIVA DE SISTEMAS DINÁMICOS:
─────────────────────────────────────────────────────────────────────────────
Cada algoritmo del zoo es un sistema dinámico con estado latente μ(t) que
evoluciona según un proceso estocástico. El analizador examina:

1. CONVERGENCIA DE LA HABILIDAD (Learning Curves):
   Las curvas μ(t) ± σ(t) muestran cómo la estimación bayesiana de habilidad
   se actualiza a medida que acumula evidencia. Una σ(t) decreciente indica
   que el modelo se vuelve más "seguro" de su estimación — equivalente a
   la reducción de varianza del filtro de Kalman ante datos sucesivos.

2. DOMINANCIA ESTOCÁSTICA (Dominance Matrix):
   La probabilidad P(μ_A > μ_B) se calcula analíticamente desde las
   distribuciones normales post-convergencia. Esto es dominancia de primer
   orden bajo gaussianidad — la probabilidad de que un sorteo de μ_A supere
   un sorteo de μ_B dado el estado actual del modelo.

3. VOLATILIDAD DE HABILIDAD (Skill Volatility):
   La varianza temporal de μ(t) cuantifica la inestabilidad del algoritmo:
   un sistema con μ(t) errático tiene alta volatilidad de habilidad, lo que
   implica que su "habilidad" depende del régimen y no es generalizable.
   Se calcula como: Var[μ(t)] sobre todos los folds del torneo.

4. VELOCIDAD DE APRENDIZAJE (Learning Speed):
   La pendiente de σ(t): dσ/dt < 0 indica reducción de incertidumbre —
   el algoritmo "aprende" en el sentido de que el modelo se vuelve más
   preciso. Una pendiente cercana a cero indica que el algoritmo no genera
   evidencia suficiente para reducir la incertidumbre del prior.

COMPATIBILIDAD:
─────────────────────────────────────────────────────────────────────────────
· Compatible con learning_curves_ del TTTSimulator.
· Compatible con ttt_curves_ del WalkForwardELOEngine.
· El formato de entrada es siempre Dict[algo_name → DataFrame(time_date, ttt_mu, ttt_sigma)].

Referencias:
  Landfried, G. & Mocskos, E. (2024). TrueSkill Through Time. GitHub.
  Herbrich, R., Minka, T. & Graepel, T. (2007). TrueSkill™. NIPS.
  Kalman, R.E. (1960). A New Approach to Linear Filtering. ASME J. Basic Eng.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TIPO DE CURVAS
# ══════════════════════════════════════════════════════════════════════════════

# Dict[algorithm_name → DataFrame con columnas: time_date, ttt_mu, ttt_sigma]
SkillCurves = Dict[str, pd.DataFrame]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TABLA DE HABILIDAD FINAL
# ══════════════════════════════════════════════════════════════════════════════

def summarize_final_skills(
    curves: SkillCurves,
    k_conservative: float = 3.0,
) -> pd.DataFrame:
    """
    Tabla resumen de habilidad final de cada algoritmo post-convergencia.

    COLUMNAS:
    ─────────────────────────────────────────────────────────────────────────
    · mu_final:           μ al último punto de la curva (estimación puntual).
    · sigma_final:        σ al último punto (incertidumbre residual).
    · conservative_skill: μ - k*σ — límite inferior de la región de credibilidad.
                          Con k=3: el 99.7% de las realizaciones del proceso
                          estarían por encima de este valor.
    · mu_max / mu_min:    Extremos de μ durante el torneo (rango dinámico).
    · skill_volatility:   Desviación estándar temporal de μ(t).
    · sigma_reduction_pct: Reducción porcentual de σ desde el inicio al fin.
                           Positivo → el modelo ganó certeza (aprendizaje).
    · n_observations:     Número de folds en que el algoritmo tuvo estimaciones.

    Args:
        curves:          Learning curves por algoritmo.
        k_conservative:  Factor k para el score conservador.

    Returns:
        DataFrame ordenado de mayor a menor conservative_skill.
    """
    records = []
    for name, df in curves.items():
        if df.empty or "ttt_mu" not in df.columns:
            continue

        mu    = df["ttt_mu"].values
        sigma = df["ttt_sigma"].values

        mu_f     = float(mu[-1])
        sigma_f  = float(sigma[-1])
        sigma_0  = float(sigma[0])
        sigma_red = float((sigma_0 - sigma_f) / max(sigma_0, 1e-12)) * 100.0

        records.append({
            "algorithm":            name,
            "mu_final":             mu_f,
            "sigma_final":          sigma_f,
            "conservative_skill":   mu_f - k_conservative * sigma_f,
            "mu_max":               float(mu.max()),
            "mu_min":               float(mu.min()),
            "mu_mean":              float(mu.mean()),
            "skill_volatility":     float(mu.std(ddof=1)) if len(mu) > 1 else 0.0,
            "sigma_reduction_pct":  sigma_red,
            "n_observations":       len(df),
        })

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("conservative_skill", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda d: d.index + 1)
        [["rank", "algorithm", "mu_final", "sigma_final",
          "conservative_skill", "mu_max", "mu_min",
          "skill_volatility", "sigma_reduction_pct", "n_observations"]]
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. MATRIZ DE DOMINANCIA ESTOCÁSTICA
# ══════════════════════════════════════════════════════════════════════════════

def compute_dominance_matrix(
    curves: SkillCurves,
    at_time: Optional[pd.Timestamp] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Matriz de probabilidad de dominancia estocástica P(μ_A > μ_B).

    FUNDAMENTO MATEMÁTICO:
    ─────────────────────────────────────────────────────────────────────────
    Dado que las habilidades son distribuciones normales post-convergencia:
      μ_A ~ N(m_A, s_A²)    y    μ_B ~ N(m_B, s_B²)

    La variable aleatoria (μ_A − μ_B) ~ N(m_A − m_B, s_A² + s_B²)

    La probabilidad de que A supere a B es:
      P(μ_A > μ_B) = Φ( (m_A − m_B) / sqrt(s_A² + s_B²) )

    donde Φ es la CDF de la distribución normal estándar.

    Esta es la probabilidad de que, si ambos algoritmos se enfrentan en un
    período futuro no observado, A genere una habilidad superior a B.

    INTERPRETACIÓN:
    · P(A,B) > 0.95 → A domina a B con 95% de confianza estadística.
    · P(A,B) ≈ 0.50 → habilidades estadísticamente indistinguibles.
    · La matriz es antisimétrica: P(A,B) + P(B,A) = 1.

    Args:
        curves:           Learning curves por algoritmo.
        at_time:          Timestamp de evaluación. None → última estimación.
        confidence_level: Umbral para marcar dominancia estadística en el output.

    Returns:
        DataFrame cuadrado P(row_algo > col_algo) con algoritmos en filas/columnas.
        Diagonal = 0.50 (P(A > A) = 0.50 por simetría).
    """
    # Extraer (mu, sigma) para cada algoritmo en at_time
    skill_params: Dict[str, Tuple[float, float]] = {}
    for name, df in curves.items():
        if df.empty:
            continue
        if at_time is not None:
            past = df[df["time_date"] <= at_time]
            row  = past.iloc[-1] if not past.empty else df.iloc[-1]
        else:
            row = df.iloc[-1]
        skill_params[name] = (float(row["ttt_mu"]), float(row["ttt_sigma"]))

    names = sorted(skill_params.keys())
    n     = len(names)
    mat   = pd.DataFrame(np.nan, index=names, columns=names)

    for i, name_a in enumerate(names):
        m_a, s_a = skill_params[name_a]
        for j, name_b in enumerate(names):
            if i == j:
                mat.at[name_a, name_b] = 0.50
                continue
            m_b, s_b = skill_params[name_b]
            # P(μ_A > μ_B)
            diff_std = np.sqrt(s_a**2 + s_b**2)
            if diff_std < 1e-12:
                prob = 1.0 if m_a > m_b else (0.5 if m_a == m_b else 0.0)
            else:
                prob = float(sp_stats.norm.cdf((m_a - m_b) / diff_std))
            mat.at[name_a, name_b] = round(prob, 4)

    return mat


def get_dominant_pairs(
    dominance_matrix: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Extrae pares (A, B) donde P(A > B) supera el umbral de dominancia.

    Args:
        dominance_matrix: Resultado de compute_dominance_matrix().
        threshold:        Probabilidad mínima para declarar dominancia.

    Returns:
        DataFrame con [winner, loser, p_dominance] ordenado de mayor a menor.
    """
    rows = []
    for winner in dominance_matrix.index:
        for loser in dominance_matrix.columns:
            if winner == loser:
                continue
            prob = dominance_matrix.at[winner, loser]
            if prob >= threshold:
                rows.append({
                    "winner":      winner,
                    "loser":       loser,
                    "p_dominance": prob,
                })
    if not rows:
        return pd.DataFrame(columns=["winner", "loser", "p_dominance"])
    return (
        pd.DataFrame(rows)
        .sort_values("p_dominance", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRICAS DE ESTABILIDAD DEL SISTEMA DINÁMICO
# ══════════════════════════════════════════════════════════════════════════════

def compute_skill_stability_metrics(
    curves: SkillCurves,
) -> pd.DataFrame:
    """
    Métricas de estabilidad de la habilidad latente como sistema dinámico.

    MARCO CONCEPTUAL:
    ─────────────────────────────────────────────────────────────────────────
    Cada curva μ(t) es una realización de un proceso estocástico discreto.
    La "estabilidad" del sistema se mide a través de:

    1. VOLATILIDAD DE HABILIDAD: σ[μ(t)]
       Varianza temporal de las estimaciones μ. Alta volatilidad → el algoritmo
       tiene habilidad fuertemente dependiente del régimen. Baja → robusto.

    2. DRIFT NETO: (μ_final − μ_inicial) / T
       Tendencia secular de la habilidad. Positivo → aprendizaje acumulativo;
       negativo → degradación; ≈0 → estacionario.

    3. VELOCIDAD DE REDUCCIÓN DE INCERTIDUMBRE: dσ/dt
       Pendiente (por regresión OLS) de σ(t). Valores negativos indican
       convergencia epistémica del estimador bayesiano — el sistema aprende.

    4. RATIO SEÑAL/RUIDO: μ_media / σ_media
       Análogo al ratio de Sharpe pero para la habilidad latente.
       Mide cuánto de la habilidad estimada es señal (μ) vs. incertidumbre (σ).

    5. MÁXIMO RETROCESO DE HABILIDAD:
       Análogo al drawdown, pero sobre μ(t): el mayor descenso desde un
       máximo previo de μ. Un algoritmo que colapsa en habilidad durante
       ciertos regímenes tendrá un alto "skill drawdown".

    6. LONGITUD DE CORRELACIÓN: lag donde autocorr(μ) cae por debajo de 0.5.
       Mide la persistencia temporal de la habilidad estimada. Alta longitud
       de correlación → la habilidad es persistente (explotar para trading).
       Baja → la habilidad fluctúa sin memoria (aleatoria).

    Args:
        curves: Learning curves por algoritmo.

    Returns:
        DataFrame con métricas de estabilidad por algoritmo.
    """
    records = []
    for name, df in curves.items():
        if df.empty or len(df) < 3:
            continue

        mu    = df["ttt_mu"].values.astype(float)
        sigma = df["ttt_sigma"].values.astype(float)
        t_idx = np.arange(len(mu))

        # 1. Volatilidad de habilidad
        skill_vol = float(np.std(mu, ddof=1)) if len(mu) > 1 else 0.0

        # 2. Drift neto por período
        drift = float((mu[-1] - mu[0]) / max(len(mu) - 1, 1))

        # 3. Velocidad de reducción de σ (pendiente OLS de sigma vs t)
        if len(sigma) > 2:
            slope_sigma, _, r_sigma, _, _ = sp_stats.linregress(t_idx, sigma)
        else:
            slope_sigma, r_sigma = 0.0, 0.0
        sigma_reduction_speed = float(-slope_sigma)  # negativo = incertidumbre decrece

        # 4. Ratio señal/ruido
        mu_mean    = float(np.mean(np.abs(mu)))
        sigma_mean = float(np.mean(sigma))
        snr = mu_mean / max(sigma_mean, 1e-12)

        # 5. Máximo retroceso de habilidad (skill drawdown)
        cummax_mu = np.maximum.accumulate(mu)
        skill_dd  = float(np.min(mu - cummax_mu))  # siempre ≤ 0

        # 6. Longitud de correlación
        corr_length = _compute_correlation_length(mu, threshold=0.5)

        # 7. Estabilidad del ranking: fracción de tiempo con μ > 0
        pct_positive = float((mu > 0).mean())

        records.append({
            "algorithm":              name,
            "skill_volatility":       round(skill_vol, 6),
            "drift_per_period":       round(drift, 6),
            "sigma_reduction_speed":  round(sigma_reduction_speed, 6),
            "signal_noise_ratio":     round(snr, 4),
            "max_skill_drawdown":     round(skill_dd, 6),
            "correlation_length":     corr_length,
            "pct_time_positive_mu":   round(pct_positive, 4),
            "n_periods":              len(mu),
        })

    if not records:
        return pd.DataFrame()

    df_out = pd.DataFrame(records).sort_values("skill_volatility")
    df_out.insert(0, "stability_rank", range(1, len(df_out) + 1))
    return df_out.reset_index(drop=True)


def _compute_correlation_length(
    series: np.ndarray,
    threshold: float = 0.5,
    max_lag: int = 20,
) -> int:
    """
    Primer lag k donde la autocorrelación de la serie cae por debajo del umbral.

    Mide la persistencia temporal del proceso μ(t). Implementación vectorizada
    sin loops usando correlación de Pearson para cada lag.

    Returns:
        int — lag de correlación. 0 si la serie no tiene correlación significativa.
    """
    n = len(series)
    if n < 4:
        return 0

    mu_centered = series - series.mean()
    var_0 = np.dot(mu_centered, mu_centered)
    if var_0 < 1e-12:
        return 0

    for lag in range(1, min(max_lag + 1, n)):
        autocorr = np.dot(mu_centered[:-lag], mu_centered[lag:]) / var_0
        if autocorr < threshold:
            return lag

    return max_lag  # correlación persistente en todo el rango


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANÁLISIS DE VELOCIDAD DE APRENDIZAJE
# ══════════════════════════════════════════════════════════════════════════════

def compute_learning_speed(
    curves: SkillCurves,
    n_initial_folds: int = 3,
) -> pd.DataFrame:
    """
    Velocidad de convergencia epistémica de cada algoritmo.

    MÉTRICA PRINCIPAL — Reducción de Incertidumbre Acumulada (RIA):
    ─────────────────────────────────────────────────────────────────────────
    RIA = σ₀ − σ_T  (reducción absoluta de la desviación estándar)

    Normalizada: RIA% = (σ₀ − σ_T) / σ₀ × 100%

    La velocidad de aprendizaje también se mide como el número de folds
    necesarios para que σ caiga al 50% de su valor inicial (half-time).

    Un algoritmo con half-time bajo genera evidencia informativa rápidamente —
    cada fold que compite aporta mucha información nueva sobre su habilidad.
    Un algoritmo con half-time alto es "opaco" — sus resultados son difíciles
    de distinguir del ruido, requiriendo muchos folds para establecer su nivel.

    Args:
        curves:          Learning curves por algoritmo.
        n_initial_folds: Folds iniciales para medir σ₀ (suavizado del prior).

    Returns:
        DataFrame con métricas de velocidad de aprendizaje.
    """
    records = []
    for name, df in curves.items():
        if df.empty or len(df) < 2:
            continue

        sigma = df["ttt_sigma"].values.astype(float)
        mu    = df["ttt_mu"].values.astype(float)

        sigma_0 = float(np.mean(sigma[:n_initial_folds]))
        sigma_T = float(sigma[-1])

        # Reducción absoluta y porcentual
        reduction_abs = sigma_0 - sigma_T
        reduction_pct = (reduction_abs / max(sigma_0, 1e-12)) * 100.0

        # Half-time: cuántos folds para σ < σ₀ / 2
        half_sigma = sigma_0 / 2.0
        half_time_idx = np.argmax(sigma <= half_sigma)
        half_time = int(half_time_idx) if sigma[half_time_idx] <= half_sigma else len(sigma)

        # Mu en el fold n_initial_folds (creencia inicial estabilizada)
        mu_initial = float(np.mean(mu[:n_initial_folds])) if len(mu) >= n_initial_folds else float(mu[0])
        mu_final   = float(mu[-1])

        # "Cambio de creencia" — cuánto se actualizó la estimación de habilidad
        belief_update = abs(mu_final - mu_initial)

        records.append({
            "algorithm":         name,
            "sigma_initial":     round(sigma_0, 5),
            "sigma_final":       round(sigma_T, 5),
            "reduction_abs":     round(reduction_abs, 5),
            "reduction_pct":     round(reduction_pct, 2),
            "half_time_folds":   half_time,
            "mu_initial":        round(mu_initial, 5),
            "mu_final":          round(mu_final, 5),
            "belief_update":     round(belief_update, 5),
        })

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("reduction_pct", ascending=False)
        .reset_index(drop=True)
        .assign(learning_rank=lambda d: d.index + 1)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN — Dashboard HTML (sin Matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def render_skill_dashboard(
    curves: SkillCurves,
    title: str = "TTT Skill Analysis Dashboard",
    output_path: Optional[str] = None,
) -> str:
    """
    Genera un dashboard HTML interactivo de las curvas de habilidad TTT.

    COMPONENTES DEL DASHBOARD:
    ─────────────────────────────────────────────────────────────────────────
    1. Gráfico de curvas μ(t) ± σ(t) por algoritmo (Chart.js).
    2. Tabla de habilidad final ordenada por conservative_skill.
    3. Mapa de calor de dominancia estocástica P(A > B).
    4. Tabla de métricas de estabilidad del sistema dinámico.

    Usa Chart.js desde CDN (sin dependencias Python de visualización).
    Compatible con cualquier navegador moderno.

    Args:
        curves:      Learning curves por algoritmo.
        title:       Título del dashboard.
        output_path: Si se especifica, guarda el HTML en disco.

    Returns:
        str — HTML completo del dashboard.
    """
    # Tablas analíticas
    skill_df     = summarize_final_skills(curves)
    dom_matrix   = compute_dominance_matrix(curves)
    stability_df = compute_skill_stability_metrics(curves)

    # Preparar datos para Chart.js
    chart_datasets = _build_chartjs_datasets(curves)
    dom_heatmap    = _build_dominance_heatmap_data(dom_matrix)

    # Tabla de habilidad → HTML
    skill_table_html = _df_to_html_table(
        skill_df[["rank", "algorithm", "mu_final", "sigma_final",
                  "conservative_skill", "skill_volatility",
                  "sigma_reduction_pct"]].round(5),
        css_class="skill-table"
    )
    stability_table_html = _df_to_html_table(
        stability_df[["stability_rank", "algorithm", "skill_volatility",
                      "drift_per_period", "signal_noise_ratio",
                      "max_skill_drawdown", "correlation_length"]].round(5),
        css_class="stability-table"
    )

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
      :root {{
        --bg:         #0b0e1a;
        --surface:    #111624;
        --border:     #1e2540;
        --accent:     #4f8ef7;
        --accent2:    #c97afc;
        --text:       #d4daf2;
        --text-dim:   #6b7499;
        --green:      #38e8a0;
        --red:        #f75a5a;
        --gold:       #f4c254;
        --font-mono:  'Courier New', monospace;
        --font-main:  'Segoe UI', system-ui, sans-serif;
        --radius:     8px;
      }}
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: var(--bg);
        color: var(--text);
        font-family: var(--font-main);
        font-size: 14px;
        line-height: 1.6;
        padding: 24px;
      }}
      h1 {{
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: .04em;
        color: #fff;
        margin-bottom: 4px;
      }}
      .subtitle {{
        color: var(--text-dim);
        font-size: .85rem;
        margin-bottom: 28px;
        font-family: var(--font-mono);
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
      }}
      .grid-full {{ grid-column: 1 / -1; }}
      .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px;
      }}
      .card h2 {{
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .12em;
        color: var(--text-dim);
        margin-bottom: 16px;
      }}
      .chart-wrap {{
        position: relative;
        height: 320px;
      }}
      /* Tables */
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
      }}
      th {{
        background: var(--border);
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: .07em;
        font-size: 10px;
        padding: 8px 10px;
        text-align: right;
      }}
      th:first-child, th:nth-child(2) {{ text-align: left; }}
      td {{
        padding: 7px 10px;
        border-bottom: 1px solid var(--border);
        text-align: right;
        font-family: var(--font-mono);
        font-size: 12px;
      }}
      td:first-child, td:nth-child(2) {{
        text-align: left;
        font-family: var(--font-main);
        font-weight: 500;
      }}
      tr:hover td {{ background: rgba(79,142,247,.06); }}
      td.pos {{ color: var(--green); }}
      td.neg {{ color: var(--red);   }}
      td.rank1 {{ color: var(--gold); font-weight: 700; }}

      /* Dominance heatmap */
      .heatmap-grid {{
        display: grid;
        gap: 3px;
        font-size: 11px;
        overflow-x: auto;
      }}
      .hm-cell {{
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 3px;
        height: 34px;
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: 11px;
      }}
      .hm-header {{
        color: var(--text-dim);
        font-family: var(--font-main);
        font-size: 10px;
        font-weight: 600;
        writing-mode: vertical-rl;
        text-orientation: mixed;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        justify-content: flex-end;
        height: 70px;
        background: transparent !important;
      }}
      .hm-row-label {{
        text-align: right;
        justify-content: flex-end;
        padding-right: 6px;
        color: var(--text-dim);
        font-family: var(--font-main);
        font-size: 11px;
        background: transparent !important;
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}
      .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
      }}
      .badge-green {{ background: rgba(56,232,160,.15); color: var(--green); }}
      .badge-red   {{ background: rgba(247,90,90,.15);  color: var(--red); }}
      .badge-dim   {{ background: rgba(107,116,153,.1); color: var(--text-dim); }}
    </style>
    </head>
    <body>
    <h1>⬡ {title}</h1>
    <div class="subtitle">
      TrueSkill Through Time — Inferencia Bayesiana Global |
      Landfried &amp; Mocskos (2024)
    </div>

    <div class="grid">

      <!-- ── Learning Curves ───────────────────────────────────────────── -->
      <div class="card grid-full">
        <h2>Curvas de Habilidad μ(t) ± σ(t) — Proceso Estocástico por Algoritmo</h2>
        <div class="chart-wrap">
          <canvas id="lcChart"></canvas>
        </div>
      </div>

      <!-- ── Tabla habilidad final ─────────────────────────────────────── -->
      <div class="card">
        <h2>Habilidad Final — Ranking por Score Conservador (μ − 3σ)</h2>
        {skill_table_html}
      </div>

      <!-- ── Mapa de dominancia ────────────────────────────────────────── -->
      <div class="card">
        <h2>Matriz de Dominancia Estocástica P(fila &gt; columna)</h2>
        <div id="dominanceHeatmap"></div>
      </div>

      <!-- ── Volatilidad σ(t) ──────────────────────────────────────────── -->
      <div class="card grid-full">
        <h2>Evolución de Incertidumbre σ(t) — Velocidad de Convergencia Epistémica</h2>
        <div class="chart-wrap">
          <canvas id="sigmaChart"></canvas>
        </div>
      </div>

      <!-- ── Estabilidad ───────────────────────────────────────────────── -->
      <div class="card grid-full">
        <h2>Métricas de Estabilidad del Sistema Dinámico</h2>
        {stability_table_html}
      </div>

    </div>

    <script>
    // ── Paleta de colores ────────────────────────────────────────────────────
    const PALETTE = [
      '#4f8ef7','#c97afc','#38e8a0','#f4c254','#f75a5a',
      '#56d4f8','#fc9c7a','#a8fc7a','#fc7ad0','#7ab8fc'
    ];

    // ── Datos de curvas ──────────────────────────────────────────────────────
    const chartData = {chart_datasets};
    const domData   = {dom_heatmap};

    // ── Learning Curves ──────────────────────────────────────────────────────
    (function() {{
      const ctx = document.getElementById('lcChart').getContext('2d');
      const datasets = [];
      chartData.names.forEach((name, i) => {{
        const c = PALETTE[i % PALETTE.length];
        const mu    = chartData.mu[name];
        const sigma = chartData.sigma[name];
        const labels = chartData.labels[name];

        // μ line
        datasets.push({{
          label: name + ' μ',
          data: mu,
          borderColor: c,
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.35,
        }});

        // σ band (μ + 1σ dashed)
        datasets.push({{
          label: name + ' +σ',
          data: mu.map((v,j) => v + (sigma[j]||0)),
          borderColor: c + '55',
          backgroundColor: c + '18',
          borderWidth: 1,
          borderDash: [4,3],
          pointRadius: 0,
          tension: 0.35,
          fill: '+1',
        }});
        datasets.push({{
          label: name + ' -σ',
          data: mu.map((v,j) => v - (sigma[j]||0)),
          borderColor: c + '55',
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderDash: [4,3],
          pointRadius: 0,
          tension: 0.35,
          fill: false,
        }});
      }});

      const allLabels = chartData.all_labels;
      new Chart(ctx, {{
        type: 'line',
        data: {{ labels: allLabels, datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{
              labels: {{
                color: '#6b7499', font: {{ size: 11 }},
                filter: item => !item.text.includes('±') &&
                                !item.text.includes('+σ') &&
                                !item.text.includes('-σ'),
              }}
            }},
            tooltip: {{ mode: 'index', intersect: false }}
          }},
          scales: {{
            x: {{ ticks: {{ color:'#6b7499', maxTicksLimit:10 }},
                  grid: {{ color:'#1e2540' }} }},
            y: {{ ticks: {{ color:'#6b7499' }},
                  grid: {{ color:'#1e2540' }},
                  title: {{ display:true, text:'Habilidad μ(t)', color:'#6b7499' }} }}
          }}
        }}
      }});
    }})();

    // ── Sigma Chart ──────────────────────────────────────────────────────────
    (function() {{
      const ctx = document.getElementById('sigmaChart').getContext('2d');
      const datasets = chartData.names.map((name,i) => ({{
        label: name,
        data: chartData.sigma[name],
        borderColor: PALETTE[i % PALETTE.length],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 2.5,
        tension: 0.3,
      }}));
      new Chart(ctx, {{
        type: 'line',
        data: {{ labels: chartData.all_labels, datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color:'#6b7499', font:{{ size:11 }} }} }},
            tooltip: {{ mode:'index', intersect:false }}
          }},
          scales: {{
            x: {{ ticks:{{ color:'#6b7499', maxTicksLimit:10 }},
                  grid:{{ color:'#1e2540' }} }},
            y: {{ ticks:{{ color:'#6b7499' }}, grid:{{ color:'#1e2540' }},
                  title:{{ display:true, text:'Incertidumbre σ(t)', color:'#6b7499' }} }}
          }}
        }}
      }});
    }})();

    // ── Dominance Heatmap ────────────────────────────────────────────────────
    (function() {{
      const container = document.getElementById('dominanceHeatmap');
      const names = domData.names;
      const n = names.length;
      const values = domData.values;  // flat array row-major

      const cols = n + 1;  // row-label + n algo headers
      const grid = document.createElement('div');
      grid.className = 'heatmap-grid';
      grid.style.gridTemplateColumns = '110px ' + ' 1fr'.repeat(n);

      // Header row
      const emptyCorner = document.createElement('div');
      emptyCorner.className = 'hm-cell';
      grid.appendChild(emptyCorner);
      names.forEach(name => {{
        const h = document.createElement('div');
        h.className = 'hm-cell hm-header';
        h.title = name;
        h.textContent = name.length > 14 ? name.slice(0, 13) + '…' : name;
        grid.appendChild(h);
      }});

      // Data rows
      names.forEach((rowName, ri) => {{
        const rowLabel = document.createElement('div');
        rowLabel.className = 'hm-cell hm-row-label';
        rowLabel.title = rowName;
        rowLabel.textContent = rowName.length > 16 ? rowName.slice(0, 15) + '…' : rowName;
        grid.appendChild(rowLabel);

        names.forEach((_, ci) => {{
          const val = values[ri * n + ci];
          const cell = document.createElement('div');
          cell.className = 'hm-cell';
          cell.textContent = val.toFixed(2);
          cell.title = rowName + ' vs ' + names[ci] + ' = ' + val.toFixed(4);

          // Color: verde = dominancia alta, rojo = dominancia baja
          if (ri === ci) {{
            cell.style.background = '#1e2540';
            cell.style.color = '#6b7499';
          }} else {{
            const t = Math.max(0, Math.min(1, (val - 0.3) / 0.6));
            const r = Math.round(255 * (1 - t));
            const g = Math.round(232 * t);
            const b = Math.round(140 * t + 90 * (1-t));
            cell.style.background = `rgba(${{r}},${{g}},${{b}},0.25)`;
            cell.style.color = val >= 0.75 ? '#38e8a0' :
                               val <= 0.35 ? '#f75a5a' : '#d4daf2';
          }}
          grid.appendChild(cell);
        }});
      }});

      container.appendChild(grid);
    }})();
    </script>
    </body>
    </html>
    """).strip()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Dashboard guardado en: {output_path}")

    return html


# ── Helpers para Chart.js ────────────────────────────────────────────────────

def _build_chartjs_datasets(curves: SkillCurves) -> str:
    """Serializa las curvas al formato JSON para Chart.js."""
    import json

    names   = sorted(curves.keys())
    mu_dict = {}
    si_dict = {}
    lb_dict = {}

    all_dates: List[str] = []
    for name in names:
        df = curves[name]
        if df.empty:
            mu_dict[name] = []
            si_dict[name] = []
            lb_dict[name] = []
            continue
        dates = [str(d.date()) if hasattr(d, "date") else str(d)
                 for d in df["time_date"]]
        mu_dict[name] = [round(v, 6) for v in df["ttt_mu"].tolist()]
        si_dict[name] = [round(v, 6) for v in df["ttt_sigma"].tolist()]
        lb_dict[name] = dates
        all_dates.extend(dates)

    # Índice temporal unificado (unión de todas las fechas)
    all_labels = sorted(set(all_dates))

    data = {
        "names":      names,
        "mu":         mu_dict,
        "sigma":      si_dict,
        "labels":     lb_dict,
        "all_labels": all_labels,
    }
    return json.dumps(data)


def _build_dominance_heatmap_data(dom_matrix: pd.DataFrame) -> str:
    """Serializa la matriz de dominancia al formato JSON para el heatmap."""
    import json
    names  = dom_matrix.index.tolist()
    values = []
    for name_a in names:
        for name_b in names:
            v = dom_matrix.at[name_a, name_b]
            values.append(round(float(v), 4) if not np.isnan(v) else 0.5)
    return json.dumps({"names": names, "values": values})


def _df_to_html_table(df: pd.DataFrame, css_class: str = "data-table") -> str:
    """Convierte un DataFrame a tabla HTML con clases semánticas."""
    if df.empty:
        return "<p style='color:#6b7499'>Sin datos disponibles.</p>"

    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows_html = []
    for i, row in df.iterrows():
        cells = []
        for j, (col, val) in enumerate(row.items()):
            if col in ("rank", "stability_rank", "learning_rank"):
                css = "rank1" if val == 1 else ""
                cells.append(f'<td class="{css}">{val}</td>')
            elif isinstance(val, float):
                css = "pos" if val > 0 else ("neg" if val < 0 else "")
                cells.append(f'<td class="{css}">{val:.5f}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f'<div style="overflow-x:auto">'
        f'<table class="{css_class}">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. FUNCIÓN PRINCIPAL DE ANÁLISIS — Punto de entrada único
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    curves: SkillCurves,
    output_html: Optional[str] = "ttt_dashboard.html",
    verbose: bool = True,
    k_conservative: float = 3.0,
    dominance_threshold: float = 0.80,
) -> Dict[str, pd.DataFrame]:
    """
    Ejecuta el análisis completo TTT y genera todos los artefactos.

    PIPELINE DE ANÁLISIS:
    ─────────────────────────────────────────────────────────────────────────
    1. Tabla de habilidad final (ranking + métricas bayesianas).
    2. Matriz de dominancia estocástica P(A > B).
    3. Pares con dominancia estadísticamente significativa.
    4. Métricas de estabilidad del sistema dinámico.
    5. Velocidad de aprendizaje (reducción de incertidumbre).
    6. Dashboard HTML interactivo.

    Esta función es el punto de entrada recomendado desde el script principal.

    Args:
        curves:               Dict[algo → DataFrame(time_date, ttt_mu, ttt_sigma)].
        output_html:          Ruta donde guardar el dashboard. None = no guardar.
        verbose:              Imprimir resumen en consola.
        k_conservative:       Factor k para el score conservador (default 3.0).
        dominance_threshold:  Umbral P(A>B) para declarar dominancia (default 0.80).

    Returns:
        Dict con todos los DataFrames de análisis:
          'skill_summary'   — habilidad final y métricas bayesianas.
          'dominance_matrix'— P(A > B) para todos los pares.
          'dominant_pairs'  — pares con dominancia > threshold.
          'stability'       — métricas de estabilidad dinámica.
          'learning_speed'  — velocidad de convergencia epistémica.
    """
    if not curves:
        logger.warning("run_full_analysis: curves vacío. Abortando.")
        return {}

    # ── 1. Habilidad final ────────────────────────────────────────────────────
    skill_df = summarize_final_skills(curves, k_conservative=k_conservative)

    # ── 2. Dominancia ─────────────────────────────────────────────────────────
    dom_matrix = compute_dominance_matrix(curves)
    dom_pairs  = get_dominant_pairs(dom_matrix, threshold=dominance_threshold)

    # ── 3. Estabilidad ────────────────────────────────────────────────────────
    stability_df = compute_skill_stability_metrics(curves)

    # ── 4. Velocidad de aprendizaje ───────────────────────────────────────────
    speed_df = compute_learning_speed(curves)

    # ── 5. Dashboard HTML ─────────────────────────────────────────────────────
    render_skill_dashboard(curves, output_path=output_html)

    # ── 6. Imprimir resumen en consola ────────────────────────────────────────
    if verbose:
        _print_console_summary(
            skill_df, dom_pairs, stability_df, speed_df,
            dominance_threshold, k_conservative
        )

    return {
        "skill_summary":    skill_df,
        "dominance_matrix": dom_matrix,
        "dominant_pairs":   dom_pairs,
        "stability":        stability_df,
        "learning_speed":   speed_df,
    }


def _print_console_summary(
    skill_df: pd.DataFrame,
    dom_pairs: pd.DataFrame,
    stability_df: pd.DataFrame,
    speed_df: pd.DataFrame,
    dom_threshold: float,
    k: float,
) -> None:
    """Imprime el resumen de análisis TTT en consola con formato legible."""

    SEP = "─" * 70

    print(f"\n{'═'*70}")
    print("  ANÁLISIS BAYESIANO TTT — TrueSkill Through Time")
    print(f"  Landfried & Mocskos (2024)")
    print(f"{'═'*70}")

    # Habilidad final
    print(f"\n▸ HABILIDAD LATENTE FINAL (Score Conservador = μ − {k}σ)")
    print(SEP)
    if not skill_df.empty:
        cols = ["rank", "algorithm", "mu_final", "sigma_final",
                "conservative_skill", "skill_volatility"]
        print(skill_df[cols].to_string(index=False))
    else:
        print("  Sin datos de habilidad disponibles.")

    # Dominancia
    print(f"\n▸ PARES CON DOMINANCIA ESTOCÁSTICA P(A>B) ≥ {dom_threshold:.0%}")
    print(SEP)
    if not dom_pairs.empty:
        print(dom_pairs.to_string(index=False))
    else:
        print(f"  Ningún par supera el umbral de dominancia {dom_threshold:.0%}.")

    # Estabilidad
    print(f"\n▸ RANKING DE ESTABILIDAD DINÁMICA (menor volatilidad de μ = más estable)")
    print(SEP)
    if not stability_df.empty:
        cols = ["stability_rank", "algorithm", "skill_volatility",
                "signal_noise_ratio", "drift_per_period",
                "max_skill_drawdown", "correlation_length"]
        print(stability_df[cols].to_string(index=False))

    # Velocidad de aprendizaje
    print(f"\n▸ VELOCIDAD DE CONVERGENCIA EPISTÉMICA (mayor σ_reduction → aprende más rápido)")
    print(SEP)
    if not speed_df.empty:
        cols = ["learning_rank", "algorithm", "sigma_initial",
                "sigma_final", "reduction_pct", "half_time_folds"]
        print(speed_df[cols].to_string(index=False))

    print(f"\n{'═'*70}\n")
