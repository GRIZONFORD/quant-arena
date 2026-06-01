# utils.py
"""
Utilidades Core — TTT/CPCV Walk-Forward Engine
===============================================
Funciones de soporte para el motor Walk-Forward con TrueSkill Through Time:

1. ALINEACIÓN TEMPORAL:
   align_series()       — sincroniza múltiples Series/DataFrames al mismo índice.
   align_series_to_index() — retrocompatibilidad con data_manager_v2.py.

2. MÉTRICAS TTT (TrueSkill Through Time):
   extract_ttt_skill()          — mu y sigma de un jugador en un timestamp dado.
   compute_conservative_skill() — habilidad conservadora = mu - k*sigma.
   rank_strategies_by_ttt()     — ranking de estrategias post-convergencia.
   compute_ttt_geometric_mean() — log-evidencia marginal del modelo ajustado.

3. MÉTRICAS FINANCIERAS DE APOYO:
   compute_log_returns()    — vectorizado, con manejo estricto de NaNs.
   compute_sharpe_ratio()   — Sharpe anualizado.
   compute_calmar_ratio()   — Calmar = CAGR / |max_drawdown|.
   compute_max_drawdown()   — drawdown máximo desde pico.

4. VALIDACIÓN ANTI-LEAKAGE:
   check_for_leakage()      — verifica solapamiento IS/OOS por fechas.
   sanity_check_market_data() — retrocompatibilidad con data_manager_v2.py.

5. INTEGRIDAD TTT:
   validate_ttt_inputs()    — valida composition/results/times antes del fit.

Referencias:
  López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
  Landfried, G. & Mocskos, E. (2024). TrueSkill Through Time.
    GitHub: github.com/glandfried/TrueSkillThroughTime
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ALINEACIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════

def align_series(
    series_list: List[Union[pd.Series, pd.DataFrame]],
    method: str = "ffill",
    dropna_threshold: float = 0.0,
) -> List[Union[pd.Series, pd.DataFrame]]:
    """
    Alinea múltiples Series/DataFrames al índice común (intersección) de todos.

    En el motor TTT, todos los "jugadores" (estrategias) deben competir sobre
    los MISMOS timestamps OOS. Esta función garantiza esa condición sin
    introducir leakage (no usa fechas futuras para rellenar).

    DISEÑO:
    · La intersección de índices es la elección correcta para el motor TTT:
      garantiza que cada "partido" tiene información completa de ambas partes.
    · El forward-fill es intra-serie (propaga el último valor conocido), lo
      que es apropiado para precios y retornos acumulados, pero NO para
      retornos diarios (donde un NaN indica día no operado, no un dato igual
      al anterior). En ese caso usar method='none'.

    Args:
        series_list:      Lista de pd.Series o pd.DataFrame con DatetimeIndex.
        method:           Método de relleno de NaNs tras reindexación:
                          'ffill' (default), 'bfill', 'none' (sin relleno).
        dropna_threshold: Fracción máxima de NaNs permitida por serie tras
                          alineación. Si se supera, se lanza ValueError.

    Returns:
        Lista de Series/DataFrames alineados al índice común, mismo orden
        que series_list.

    Raises:
        ValueError: Si series_list está vacía, los índices no se solapan,
                    o alguna serie supera dropna_threshold de NaNs.
    """
    if not series_list:
        raise ValueError("align_series: series_list está vacía.")

    if len(series_list) == 1:
        return [series_list[0].copy()]

    # Verificar que todos tienen DatetimeIndex
    for i, s in enumerate(series_list):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(
                f"align_series: elemento {i} no tiene DatetimeIndex "
                f"(tipo: {type(s.index).__name__})."
            )

    # Intersección de índices (timestamps en todos los elementos)
    common_idx = series_list[0].index
    for s in series_list[1:]:
        common_idx = common_idx.intersection(s.index)

    if len(common_idx) == 0:
        ranges = [(s.index.min(), s.index.max()) for s in series_list]
        raise ValueError(
            f"align_series: intersección de índices vacía. "
            f"Rangos: {ranges}. Las series no se solapan temporalmente."
        )

    logger.debug(
        f"align_series: índice común = {len(common_idx)} timestamps | "
        f"rango [{common_idx[0].date()} → {common_idx[-1].date()}]"
    )

    aligned: List[Union[pd.Series, pd.DataFrame]] = []
    for i, s in enumerate(series_list):
        # Reindexar a la intersección
        re = s.reindex(common_idx)

        # Aplicar método de relleno
        if method == "ffill":
            re = re.ffill().bfill()
        elif method == "bfill":
            re = re.bfill().ffill()
        elif method == "none":
            pass
        else:
            raise ValueError(
                f"align_series: method='{method}' inválido. "
                "Usar 'ffill', 'bfill' o 'none'."
            )

        # Verificar umbral de NaNs post-alineación
        if dropna_threshold > 0.0:
            if isinstance(re, pd.DataFrame):
                nan_frac = re.isna().any(axis=1).mean()
            else:
                nan_frac = re.isna().mean()
            if nan_frac > dropna_threshold:
                raise ValueError(
                    f"align_series: serie {i} tiene {nan_frac:.2%} NaNs "
                    f"tras alineación (umbral: {dropna_threshold:.2%})."
                )

        aligned.append(re)

    return aligned


def align_series_to_index(
    series: Union[pd.Series, pd.DataFrame],
    index: pd.DatetimeIndex,
    method: str = "ffill",
) -> Union[pd.Series, pd.DataFrame]:
    """
    Reindexar `series` a `index` con relleno.
    Retrocompatibilidad con walk_forward_engine_con_TTT.py y data_manager_v2.py.

    Args:
        series: pd.Series o pd.DataFrame con DatetimeIndex.
        index:  Índice destino (pd.DatetimeIndex).
        method: 'ffill' (default), 'bfill', o 'none'.

    Returns:
        Serie/DataFrame reindexado.
    """
    if series is None:
        raise ValueError("align_series_to_index: series es None.")

    if series.index.equals(index):
        return series.copy()

    re = series.reindex(index)

    if method == "none":
        return re
    elif method == "ffill":
        re = re.ffill().bfill()
    elif method == "bfill":
        re = re.bfill().ffill()
    else:
        raise ValueError(f"method='{method}' inválido. Usar 'ffill', 'bfill' o 'none'.")

    # Inferir dtype para evitar object columns post-fill
    if isinstance(re, pd.DataFrame):
        re = re.infer_objects()
    else:
        re = pd.Series(re).infer_objects()

    return re


# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS TTT
# ══════════════════════════════════════════════════════════════════════════════

def extract_ttt_skill(
    learning_curves: Dict[str, pd.DataFrame],
    strategy_name: str,
    at_timestamp: Optional[pd.Timestamp] = None,
) -> Tuple[float, float]:
    """
    Extrae (mu, sigma) de la curva de aprendizaje TTT de una estrategia.

    Las learning curves son DataFrames con columnas ['time_date', 'ttt_mu',
    'ttt_sigma'] producidos por TTTAccumulator.extract_learning_curves().

    Args:
        learning_curves: Dict[strategy_name → DataFrame con ttt_mu, ttt_sigma].
        strategy_name:   Nombre de la estrategia a consultar.
        at_timestamp:    Timestamp de referencia. Si None, usa el último punto
                         disponible (habilidad estimada al final del período).

    Returns:
        Tuple (mu, sigma). Si la estrategia no tiene curva disponible,
        devuelve (0.0, np.inf) indicando prior no informativo.
    """
    if strategy_name not in learning_curves:
        logger.warning(
            f"extract_ttt_skill: '{strategy_name}' no encontrado en "
            f"learning_curves. Devolviendo prior (0.0, inf)."
        )
        return 0.0, np.inf

    curve = learning_curves[strategy_name]
    required_cols = {"time_date", "ttt_mu", "ttt_sigma"}
    if not required_cols.issubset(curve.columns):
        raise ValueError(
            f"extract_ttt_skill: curva de '{strategy_name}' falta columnas. "
            f"Requeridas: {required_cols}. Disponibles: {set(curve.columns)}"
        )

    if curve.empty:
        return 0.0, np.inf

    if at_timestamp is None:
        # Último punto disponible — habilidad post-convergencia final
        row = curve.iloc[-1]
    else:
        # Punto más cercano al timestamp (sin usar información futura)
        past_mask = pd.to_datetime(curve["time_date"]) <= at_timestamp
        if not past_mask.any():
            logger.debug(
                f"extract_ttt_skill: '{strategy_name}' no tiene puntos "
                f"<= {at_timestamp}. Usando primer punto disponible."
            )
            row = curve.iloc[0]
        else:
            row = curve.loc[past_mask].iloc[-1]

    return float(row["ttt_mu"]), float(row["ttt_sigma"])


def compute_conservative_skill(
    mu: float,
    sigma: float,
    k: float = 3.0,
) -> float:
    """
    Habilidad conservadora = mu - k * sigma.

    Equivalente al "skill score" conservador de TrueSkill clásico (Herbrich
    et al., 2007). Penaliza la incertidumbre: estrategias con alta sigma
    (pocas observaciones OOS) son degradadas en el ranking.

    Recomendaciones:
    · k = 3.0: límite inferior del 99.7% con distribución normal (muy conservador).
    · k = 2.0: 95% (moderado; recomendado para rankings intermedios).
    · k = 1.0: 68% (agresivo; útil si todos los jugadores tienen sigma similar).

    Args:
        mu:    Media de habilidad de la estrategia.
        sigma: Desviación estándar de habilidad.
        k:     Factor de conservadurismo (default: 3.0).

    Returns:
        Skill score conservador (float). Puede ser negativo.
    """
    if sigma < 0:
        raise ValueError(f"sigma={sigma} debe ser >= 0.")
    if k < 0:
        raise ValueError(f"k={k} debe ser >= 0.")
    return float(mu - k * sigma)


def rank_strategies_by_ttt(
    learning_curves: Dict[str, pd.DataFrame],
    at_timestamp: Optional[pd.Timestamp] = None,
    k_conservative: float = 3.0,
    include_uncertain: bool = True,
) -> pd.DataFrame:
    """
    Genera un ranking de estrategias por habilidad TTT post-convergencia.

    CRITERIO PRIMARIO: conservative_skill = mu - k*sigma
    Criterio de desempate: mu (mayor es mejor)

    Este criterio es superior al ELO estático porque incorpora la
    incertidumbre epistémica sobre la habilidad (sigma), penalizando
    estrategias con pocas observaciones OOS incluso si tienen mu alto.

    Args:
        learning_curves:   Dict[strategy_name → DataFrame de curva TTT].
        at_timestamp:      Timestamp de evaluación. None → último punto.
        k_conservative:    Factor k para compute_conservative_skill().
        include_uncertain: Si False, excluye estrategias con sigma=inf
                           (no observadas en ningún fold OOS).

    Returns:
        DataFrame con columnas [rank, strategy, ttt_mu, ttt_sigma,
        conservative_skill] ordenado de mejor a peor.
    """
    records: List[dict] = []

    for name in learning_curves:
        mu, sigma = extract_ttt_skill(learning_curves, name, at_timestamp)

        if not include_uncertain and np.isinf(sigma):
            logger.debug(f"rank_strategies: '{name}' excluida (sigma=inf).")
            continue

        cs = compute_conservative_skill(mu, sigma if not np.isinf(sigma) else 1e6, k_conservative)
        records.append({
            "strategy":           name,
            "ttt_mu":             mu,
            "ttt_sigma":          sigma,
            "conservative_skill": cs,
        })

    if not records:
        logger.warning("rank_strategies_by_ttt: sin estrategias para rankear.")
        return pd.DataFrame(columns=["rank", "strategy", "ttt_mu", "ttt_sigma",
                                     "conservative_skill"])

    rank_df = (
        pd.DataFrame(records)
        .sort_values(["conservative_skill", "ttt_mu"], ascending=[False, False])
        .reset_index(drop=True)
    )
    rank_df.insert(0, "rank", rank_df.index + 1)

    return rank_df


def compute_ttt_geometric_mean(history_obj) -> float:
    """
    Calcula la media geométrica de la log-evidencia marginal del modelo TTT.

    h.geometric_mean() = exp( mean( log P(result_t | history_{<t}) ) )

    Un valor más alto indica mejor ajuste bayesiano (el modelo asigna mayor
    probabilidad a priori a los resultados observados). Usado en la
    calibración de (gamma, sigma) por grid search.

    Args:
        history_obj: Objeto History de trueskillthroughtime post-convergencia.

    Returns:
        float: Media geométrica. NaN si el objeto es None o inválido.
    """
    if history_obj is None:
        return float("nan")
    try:
        return float(history_obj.geometric_mean())
    except Exception as exc:
        logger.error(f"compute_ttt_geometric_mean: error — {exc}")
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRICAS FINANCIERAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(
    prices: pd.Series,
    fill_method: str = "ffill",
) -> pd.Series:
    """
    Calcula retornos logarítmicos vectorizados con manejo estricto de NaNs.

    r_t = ln(P_t / P_{t-1})

    Ventajas del retorno log sobre aritmético:
    · Aditividad temporal: suma de r_t = retorno total compuesto.
    · Simetría: ganancia/pérdida del mismo porcentaje son iguales en magnitud.
    · Aproximación gaussiana más robusta para modelado TTT.

    Args:
        prices:      Serie de precios con DatetimeIndex.
        fill_method: Método de relleno de gaps antes del cálculo:
                     'ffill', 'bfill', o 'none' (sin relleno).

    Returns:
        pd.Series de log-returns. El primer elemento es NaN (sin P_{t-1}).

    Raises:
        ValueError: Si la serie contiene precios <= 0 o es completamente NaN.
    """
    if prices.empty:
        raise ValueError("compute_log_returns: serie de precios vacía.")

    prices = prices.astype(float)

    # Relleno de gaps
    if fill_method == "ffill":
        prices = prices.ffill()
    elif fill_method == "bfill":
        prices = prices.bfill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError(
            f"fill_method='{fill_method}' inválido. Usar 'ffill', 'bfill' o 'none'."
        )

    # Precios no positivos → log indefinido
    nonpos = (prices <= 0).sum()
    if nonpos > 0:
        raise ValueError(
            f"compute_log_returns: {nonpos} precios <= 0 detectados. "
            "Verificar datos de origen."
        )

    if prices.isna().all():
        raise ValueError("compute_log_returns: todos los precios son NaN.")

    return np.log(prices / prices.shift(1))


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 30,
) -> float:
    """
    Sharpe Ratio anualizado.

    SR = (E[r] - rf) / std(r) * sqrt(T)

    Args:
        returns:          Serie de retornos (log o aritméticos).
        risk_free_rate:   Tasa libre de riesgo diaria (default 0).
        periods_per_year: Factor de anualización (252 para diario).
        min_periods:      Mínimo de observaciones; devuelve NaN si no se cumple.

    Returns:
        Sharpe Ratio anualizado, o NaN si datos insuficientes.
    """
    r = returns.dropna()
    if len(r) < min_periods:
        return float("nan")

    excess = r - risk_free_rate
    sigma  = excess.std(ddof=1)
    if sigma < 1e-12:
        return float("nan")

    return float((excess.mean() / sigma) * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Drawdown máximo desde pico sobre una serie de retornos.

    MDD = max( (P_peak - P_trough) / P_peak )

    Calcula el wealth index normalizado a 1.0 y mide la mayor caída
    porcentual desde cualquier máximo histórico previo.

    Args:
        returns: Serie de retornos diarios (log o aritméticos).

    Returns:
        MDD como número positivo en [0, 1]. 0.0 si la serie es plana.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")

    # Wealth index compuesto (1 = capital inicial)
    wealth = (1.0 + r).cumprod()
    rolling_peak = wealth.cummax()
    drawdown = (wealth - rolling_peak) / rolling_peak
    return float(abs(drawdown.min()))


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    min_periods: int = 60,
) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|.

    Métrica de riesgo/retorno que penaliza explícitamente el tail risk.
    Usada como fitness function en el motor TTT (fitness='calmar').

    CAGR = (wealth_final / wealth_initial)^(periods_per_year / n) - 1

    Args:
        returns:          Serie de retornos diarios.
        periods_per_year: Factor de anualización.
        min_periods:      Mínimo de observaciones para cálculo válido.

    Returns:
        Calmar Ratio. NaN si datos insuficientes o MDD=0.
    """
    r = returns.dropna()
    if len(r) < min_periods:
        return float("nan")

    # CAGR desde wealth index
    wealth = (1.0 + r).cumprod()
    n = len(r)
    cagr = float((wealth.iloc[-1] / wealth.iloc[0]) ** (periods_per_year / n) - 1.0)

    mdd = compute_max_drawdown(r)
    if mdd < 1e-12:
        return float("nan")   # CAGR/0 → undefined

    return float(cagr / mdd)


def compute_cvar(
    returns: pd.Series,
    alpha: float = 0.05,
    min_periods: int = 30,
) -> float:
    """
    Conditional Value at Risk (CVaR / Expected Shortfall) al nivel alpha.

    CVaR_alpha = E[r | r <= VaR_alpha]

    Mide el retorno esperado en el peor alpha% de los casos, capturando
    el riesgo de cola (tail risk) que el Sharpe no penaliza.

    Ref: Rockafellar & Uryasev (2000). Journal of Risk, 2(3), 21–41.

    Args:
        returns:     Serie de retornos diarios.
        alpha:       Nivel de confianza (default 0.05 → CVaR al 5%).
        min_periods: Mínimo de observaciones para cálculo válido.

    Returns:
        CVaR como número negativo (pérdida esperada). NaN si insuficiente.
    """
    r = returns.dropna()
    if len(r) < min_periods:
        return float("nan")

    var_threshold = r.quantile(alpha)
    tail = r[r <= var_threshold]
    if tail.empty:
        return float(var_threshold)

    return float(tail.mean())


# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDACIÓN ANTI-LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════

def check_for_leakage(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    embargo_days: int = 0,
    raise_on_overlap: bool = True,
) -> dict:
    """
    Verifica que no hay solapamiento temporal entre IS y OOS.

    LEAKAGE DIRECTO: Un timestamp IS aparece también en OOS — data leakage
    que invalidaría completamente el backtest.

    LEAKAGE POR EMBARGO: Si embargo_days > 0, verifica adicionalmente que
    el final del IS está al menos embargo_days antes del inicio del OOS.
    Esto detecta correlación serial residual incluso sin solapamiento directo.

    Es la función de auditoría final antes de ejecutar cualquier backtest.
    Úsala sistemáticamente en cada split del motor Walk-Forward.

    Args:
        df_train:        DataFrame del período IS con DatetimeIndex.
        df_test:         DataFrame del período OOS con DatetimeIndex.
        embargo_days:    Si > 0, verifica el buffer temporal embargo.
        raise_on_overlap: Si True, lanza ValueError ante cualquier leakage.

    Returns:
        Dict con:
        · 'ok'              : bool — True si no hay problemas.
        · 'n_overlap'       : int  — timestamps en común IS ∩ OOS.
        · 'overlap_dates'   : List[str] — las fechas solapadas (max 5 para brevedad).
        · 'embargo_ok'      : bool — cumple el buffer de embargo.
        · 'train_end'       : str  — último timestamp IS.
        · 'test_start'      : str  — primer timestamp OOS.
        · 'gap_days'        : int  — días calendario entre IS-end y OOS-start.

    Raises:
        ValueError: Si raise_on_overlap=True y se detecta leakage.
    """
    report: dict = {
        "ok": True,
        "n_overlap": 0,
        "overlap_dates": [],
        "embargo_ok": True,
        "train_end": None,
        "test_start": None,
        "gap_days": None,
        "errors": [],
        "warnings": [],
    }

    if df_train.empty:
        report["errors"].append("df_train está vacío.")
        report["ok"] = False
        if raise_on_overlap:
            raise ValueError("check_for_leakage: df_train está vacío.")
        return report

    if df_test.empty:
        report["errors"].append("df_test está vacío.")
        report["ok"] = False
        if raise_on_overlap:
            raise ValueError("check_for_leakage: df_test está vacío.")
        return report

    # ── Solapamiento directo ───────────────────────────────────────────────
    train_idx = df_train.index
    test_idx  = df_test.index

    overlap = train_idx.intersection(test_idx)
    report["n_overlap"]     = int(len(overlap))
    report["overlap_dates"] = [str(d.date()) for d in overlap[:5]]

    if len(overlap) > 0:
        msg = (
            f"DATA LEAKAGE DETECTADO: {len(overlap)} timestamps en IS ∩ OOS. "
            f"Primeras fechas: {report['overlap_dates']}. "
            "El backtest está completamente invalidado."
        )
        report["errors"].append(msg)
        report["ok"] = False
        logger.error(f"check_for_leakage: {msg}")
        if raise_on_overlap:
            raise ValueError(msg)

    # ── Temporalidad: IS debe preceder a OOS ─────────────────────────────
    train_end  = train_idx.max()
    test_start = test_idx.min()

    report["train_end"]  = str(train_end.date())
    report["test_start"] = str(test_start.date())

    if train_end >= test_start and len(overlap) == 0:
        # IS termina después de que OOS comienza (sin solapamiento directo)
        # Esto ocurre cuando IS y OOS son no-contiguos pero OOS comienza antes.
        msg = (
            f"Orden temporal violado: train_end={train_end.date()} >= "
            f"test_start={test_start.date()}. IS y OOS no son cronológicos."
        )
        report["errors"].append(msg)
        report["ok"] = False
        if raise_on_overlap:
            raise ValueError(f"check_for_leakage: {msg}")

    # ── Gap de días calendario ─────────────────────────────────────────────
    if train_end < test_start:
        gap_calendar = (test_start - train_end).days
        report["gap_days"] = int(gap_calendar)

        # Verificar embargo
        if embargo_days > 0 and gap_calendar < embargo_days:
            msg = (
                f"Embargo insuficiente: gap={gap_calendar} días calendario < "
                f"embargo_days={embargo_days}. Riesgo de correlación serial."
            )
            report["warnings"].append(msg)
            report["embargo_ok"] = False
            logger.warning(f"check_for_leakage: {msg}")
    else:
        report["gap_days"] = 0

    status = "OK ✓" if report["ok"] else "LEAKAGE DETECTADO ✗"
    logger.debug(
        f"check_for_leakage: {status} | "
        f"IS [{df_train.index.min().date()} → {train_end.date()}] | "
        f"OOS [{test_start.date()} → {df_test.index.max().date()}] | "
        f"overlap={len(overlap)} | gap={report['gap_days']}d"
    )

    return report


def sanity_check_market_data(
    df: pd.DataFrame,
    require_cols: Optional[List[str]] = None,
    nan_threshold: float = 0.05,
    fail_fast: bool = True,
) -> dict:
    """
    Validaciones de integridad básicas para DataFrames de datos de mercado.
    Retrocompatibilidad con walk_forward_engine_con_TTT.py.

    Checks:
      · Índice es DatetimeIndex y monótono creciente.
      · Sin timestamps duplicados.
      · Columnas requeridas presentes.
      · Sin precios close <= 0.
      · Fracción de NaNs dentro del umbral.

    Args:
        df:            DataFrame de datos de mercado.
        require_cols:  Lista de columnas que deben existir.
        nan_threshold: Fracción máxima de NaNs tolerada (aviso, no error).
        fail_fast:     Si True, lanza ValueError ante errores críticos.

    Returns:
        Dict {'ok': bool, 'errors': [...], 'warnings': [...]}.
    """
    report: dict = {"ok": True, "errors": [], "warnings": []}

    if df is None:
        report["errors"].append("DataFrame es None.")
        report["ok"] = False
        if fail_fast:
            raise ValueError("sanity_check_market_data: DataFrame es None.")
        return report

    # Índice
    if not isinstance(df.index, pd.DatetimeIndex):
        report["errors"].append("Índice no es DatetimeIndex.")
    else:
        if not df.index.is_monotonic_increasing:
            report["errors"].append("Índice no es monótono creciente.")
        if df.index.has_duplicates:
            n_dup = df.index.duplicated().sum()
            report["errors"].append(f"Índice tiene {n_dup} timestamps duplicados.")

    # Columnas requeridas
    if require_cols:
        missing = [c for c in require_cols if c not in df.columns]
        if missing:
            report["errors"].append(f"Columnas requeridas ausentes: {missing}.")

    # Precios
    if "close" in df.columns:
        nonpos = int((df["close"] <= 0).sum())
        if nonpos > 0:
            report["errors"].append(
                f"{nonpos} precios 'close' <= 0 detectados."
            )

    # NaNs
    total_vals = df.shape[0] * df.shape[1]
    nan_vals   = int(df.isna().sum().sum())
    nan_frac   = nan_vals / max(1, total_vals)
    if nan_frac > nan_threshold:
        report["warnings"].append(
            f"Fracción de NaNs elevada: {nan_frac:.3f} > umbral {nan_threshold}."
        )

    if report["errors"]:
        report["ok"] = False
        if fail_fast:
            raise ValueError(
                "sanity_check_market_data failed: " + " | ".join(report["errors"])
            )

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 5. VALIDACIÓN DE INPUTS TTT
# ══════════════════════════════════════════════════════════════════════════════

def validate_ttt_inputs(
    composition: List[List[List[str]]],
    results: List[List[float]],
    times: List[float],
    obs: Optional[List[str]] = None,
    raise_on_error: bool = True,
) -> dict:
    """
    Valida que las listas de partidos TTT son consistentes antes del fit.

    Verifica las invariantes que la librería trueskillthroughtime requiere:
    · Todas las listas tienen la misma longitud (n_matches).
    · composition[i] tiene exactamente 2 equipos de al menos 1 jugador c/u.
    · results[i] tiene exactamente 2 valores float.
    · times[i] son escalares float positivos y monótonos (recomendado).
    · obs[i] es 'Continuous' o 'Ordinal' si se provee.

    Args:
        composition: Lista de partidos [[equipo_A, equipo_B], ...].
        results:     Lista de resultados [[score_A, score_B], ...].
        times:       Lista de timestamps en días (float).
        obs:         Lista de tipos de observación por partido.
        raise_on_error: Si True, lanza ValueError ante errores.

    Returns:
        Dict {'ok': bool, 'n_matches': int, 'errors': [...], 'warnings': [...]}.
    """
    report: dict = {"ok": True, "n_matches": 0, "errors": [], "warnings": []}

    n = len(composition)
    report["n_matches"] = n

    if n == 0:
        report["errors"].append("composition está vacía — ningún partido acumulado.")
        report["ok"] = False
        if raise_on_error:
            raise ValueError("validate_ttt_inputs: composition vacía.")
        return report

    # Consistencia de longitudes
    if len(results) != n:
        report["errors"].append(
            f"len(results)={len(results)} != len(composition)={n}."
        )
    if len(times) != n:
        report["errors"].append(
            f"len(times)={len(times)} != len(composition)={n}."
        )
    if obs is not None and len(obs) != n:
        report["errors"].append(
            f"len(obs)={len(obs)} != len(composition)={n}."
        )

    # Validar cada partido
    invalid_matches = []
    for i, (comp, res) in enumerate(zip(composition, results)):
        # Exactamente 2 equipos
        if len(comp) != 2:
            invalid_matches.append(
                f"partido {i}: composition debe tener 2 equipos, tiene {len(comp)}."
            )
            continue

        # Cada equipo al menos 1 jugador
        if len(comp[0]) < 1 or len(comp[1]) < 1:
            invalid_matches.append(
                f"partido {i}: cada equipo debe tener >= 1 jugador."
            )

        # Exactamente 2 resultados
        if len(res) != 2:
            invalid_matches.append(
                f"partido {i}: results debe tener 2 valores, tiene {len(res)}."
            )
            continue

        # Resultados deben ser finitos
        if not all(np.isfinite(v) for v in res):
            invalid_matches.append(
                f"partido {i}: results contiene inf/NaN: {res}."
            )

    if invalid_matches:
        # Reportar los primeros 5 para no saturar el log
        report["errors"].extend(invalid_matches[:5])
        if len(invalid_matches) > 5:
            report["errors"].append(
                f"... y {len(invalid_matches) - 5} partidos adicionales inválidos."
            )

    # Times deben ser positivos
    times_arr = np.array(times, dtype=float)
    if (times_arr <= 0).any():
        report["warnings"].append(
            f"{(times_arr <= 0).sum()} valores de times <= 0. "
            "Verificar conversión de timestamps a días."
        )

    # Times recomendados monótonos no-decrecientes
    if not np.all(np.diff(times_arr) >= 0):
        report["warnings"].append(
            "times no es monótono no-decreciente. "
            "TTT puede funcionar, pero la suavización backward podría ser subóptima."
        )

    # Tipos de obs
    if obs is not None:
        valid_obs = {"Continuous", "Ordinal"}
        invalid_obs = [o for o in obs if o not in valid_obs]
        if invalid_obs:
            report["errors"].append(
                f"Valores de obs inválidos: {set(invalid_obs)}. "
                f"Válidos: {valid_obs}."
            )

    if report["errors"]:
        report["ok"] = False
        if raise_on_error:
            raise ValueError(
                "validate_ttt_inputs failed:\n  "
                + "\n  ".join(report["errors"])
            )

    logger.debug(
        f"validate_ttt_inputs: {'OK' if report['ok'] else 'ERRORES'} | "
        f"{n} partidos | errores={len(report['errors'])} | "
        f"advertencias={len(report['warnings'])}"
    )
    return report
