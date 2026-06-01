#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/xgboost_strategy.py
# XGBoost Trend Strategy — primera estrategia ML del Zoo
# =============================================================================
"""
Estrategia de machine learning basada en XGBoost Regressor que predice la
dirección del mercado a partir de features técnicos causales (sin look-ahead).

Integración en quant_arena:
    - Hereda de AbstractStrategy (core/abstracciones.py).
    - Se auto-registra en el Zoo con @RegistroZoo.registrar('xgboost_trend').
    - Compatible con BacktestEngine (backtesting/motor.py) y ZooManager.
    - El BacktestEngine llama a generar_señales() en cada fecha de rebalanceo;
      el resultado es un pd.Series [ticker -> peso] con valores in {-1.0, 0.0, 1.0}.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Permite ejecutar este archivo directamente (python xgboost_strategy.py) ──
_PROYECTO_RAIZ = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROYECTO_RAIZ) not in sys.path:
    sys.path.insert(0, str(_PROYECTO_RAIZ))

try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo


# =============================================================================
# Hiperparámetros por defecto del XGBRegressor
# =============================================================================

_XGB_PARAMS_DEFAULT: Dict = {
    "n_estimators":     200,
    "max_depth":          4,
    "learning_rate":   0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       1.0,
    "reg_alpha":        0.1,
    "min_child_weight":   5,    # penaliza overfitting en conjuntos pequeños
    "objective": "reg:squarederror",
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":          0,
}


# =============================================================================
# Estrategia
# =============================================================================

@RegistroZoo.registrar("xgboost_trend")
class XGBoostStrategy(AbstractStrategy):
    """
    Estrategia ML basada en XGBoost Regressor con ventana expansiva y
    features técnicos estrictamente causales.

    Pipeline por cada llamada a generar_señales(datos, fecha_corte):
        1. Corte causal: datos[: fecha_corte].
        2. Construye matriz de features F — todo rezagado, sin look-ahead.
        3. Target: log-retorno del día siguiente  (shift(-1)).
           La última fila de F (fecha_corte) tiene target=NaN -> excluida de
           entrenamiento por dropna(). NUNCA se filtra información futura.
        4. Entrena XGBRegressor con {F[:-1 filas válidas], target[:-1]}.
        5. Predice el retorno esperado con F[-1] (= features de fecha_corte).
        6. Discretiza con umbral simétrico:
               pred > +umbral  ->  1.0  (Largo)
               pred < −umbral  -> −1.0  (Corto)
               else            ->  0.0  (Neutral / Cash)

    Features construidas (todas causales en t):
        ret_1d, ret_5d, ret_21d : Log-retornos multi-escala (shift > 0 = rezago).
        vol_21d                 : Std rodante 21 días de log-retornos.
        rsi_14                  : RSI de Wilder vía EWM causal, normalizado [0,1].
        price_sma50             : Close / SMA50 − 1.
        sma50_sma200            : SMA50 / SMA200 − 1 (tendencia estructural).
        pos_range_21            : Posición en el rango Hi−Lo de 21 días (reversión).
        atr_norm_14             : ATR normalizado por precio (si High/Low disponibles).
        vol_ratio_21            : Volumen / MA-vol-21 − 1 (si Volume disponible).
    """

    def __init__(
        self,
        universo: List[str],
        min_train_days: int = 252,
        ventana_vol: int = 21,
        rsi_period: int = 14,
        umbral_señal: float = 0.003,
        close_col: str = "Close",
        high_col: str = "High",
        low_col: str = "Low",
        volume_col: str = "Volume",
        xgb_params: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            universo:      Tickers del universo. Para activo único usar  ['TICKER'].
                           El primer elemento es el nombre bajo el cual se emite la señal.
            min_train_days: Mínimo de filas de entrenamiento (post warm-up y post dropna)
                            requeridas antes de emitir una señal no neutral.
                            Default: 252 (≈ 1 año hábil).
            ventana_vol:   Días para volatilidad rodante. Default: 21.
            rsi_period:    Período del RSI de Wilder. Default: 14.
            umbral_señal:  Umbral simétrico sobre el retorno predicho (escala log).
                           |pred| > umbral -> señal +-1.0; else -> 0.0. Default: 0.003.
            close_col:     Nombre de la columna de cierre en datos. Default: "Close".
            high_col:      Columna High (usado para ATR). Default: "High".
            low_col:       Columna Low (usado para ATR). Default: "Low".
            volume_col:    Columna Volume. Default: "Volume".
            xgb_params:    Override de hiperparámetros del XGBRegressor.
        """
        if not _XGB_OK:
            raise ImportError(
                "xgboost no está instalado. Ejecuta: pip install xgboost"
            )
        if min_train_days < 60:
            raise ValueError("min_train_days debe ser >= 60.")
        if not (0.0 < umbral_señal < 0.5):
            raise ValueError("umbral_señal debe pertenecer a (0.0, 0.5).")

        super().__init__(nombre="xgboost_trend", universo=universo)

        self._min_train    = min_train_days
        self._ventana_vol  = ventana_vol
        self._rsi_period   = rsi_period
        self._umbral       = umbral_señal
        self._close_col    = close_col
        self._high_col     = high_col
        self._low_col      = low_col
        self._volume_col   = volume_col

        params = {**_XGB_PARAMS_DEFAULT, **(xgb_params or {})}
        self._modelo = xgb.XGBRegressor(**params)

    # ──────────────────────────────────────────────────────────────────────────
    # Interfaz pública — AbstractStrategy
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def descripcion(self) -> str:
        return (
            f"XGBoost Trend Regressor | "
            f"min_train={self._min_train}d | "
            f"rsi={self._rsi_period} | "
            f"umbral=+-{self._umbral:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Ingeniería de características — 100% causal (privado)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
        """
        RSI de Wilder via EWM con alpha = 1/period.

        Causal por construcción: EWM de Pandas es un filtro unidireccional;
        cada valor en t usa solo datos t, t-1, t-2, ... sin look-ahead.
        Normalizado al rango [0, 1] para uniformidad con el resto de features.
        """
        delta   = close.diff()
        gain    = delta.clip(lower=0.0)
        loss    = (-delta).clip(lower=0.0)
        avg_g   = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_l   = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs      = avg_g / avg_l.replace(0.0, np.nan)
        return (100.0 - 100.0 / (1.0 + rs)) / 100.0   # -> [0, 1]

    def _build_features(self, datos: pd.DataFrame) -> pd.DataFrame:
        """
        Construye la matriz de features estrictamente causales.

        Garantía de causalidad:
            - shift(k) con k > 0 desplaza hacia el PASADO (rezago k días).
            - rolling(w).mean()/std() usa solo datos en [t-w+1, t] inclusive.
            - ewm(adjust=False) aplica el filtro recursivo hacia adelante en el tiempo.
            - La SMA(200) es el feature de mayor warm-up: las primeras 199 filas
              tendrán NaN y serán excluidas automáticamente en _train_predict().
            - NINGUNA operación accede a datos de t+1 o posterior.

        Args:
            datos: DataFrame con DatetimeIndex ascendente. Requiere close_col.

        Returns:
            DataFrame con las mismas fechas que datos; las primeras ~199 filas
            contendrán NaN por el warm-up de SMA200.
        """
        close   = datos[self._close_col].astype(float)
        log_ret = np.log(close / close.shift(1))

        f = pd.DataFrame(index=datos.index)

        # ── Retornos multi-escala ──────────────────────────────────────────────
        f["ret_1d"]  = log_ret
        f["ret_5d"]  = np.log(close / close.shift(5))
        f["ret_21d"] = np.log(close / close.shift(21))

        # ── Volatilidad rodante ───────────────────────────────────────────────
        f["vol_21d"] = log_ret.rolling(self._ventana_vol,
                                        min_periods=self._ventana_vol).std()

        # ── RSI de Wilder ─────────────────────────────────────────────────────
        f["rsi_14"]  = self._rsi_wilder(close, self._rsi_period)

        # ── Posición relativa a medias móviles (capturas de tendencia) ─────────
        sma50  = close.rolling(50,  min_periods=50).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        f["price_sma50"]  = close / sma50  - 1.0      # > 0 -> por encima de SMA50
        f["sma50_sma200"] = sma50  / sma200 - 1.0     # > 0 -> golden-cross

        # ── Posición en el rango Hi-Lo de 21 días (reversión a media) ─────────
        hi21  = close.rolling(21, min_periods=21).max()
        lo21  = close.rolling(21, min_periods=21).min()
        rango = (hi21 - lo21).replace(0.0, np.nan)
        f["pos_range_21"] = (close - lo21) / rango - 0.5   # in [-0.5, 0.5]

        # ── ATR normalizado (requiere High y Low) ─────────────────────────────
        if self._high_col in datos.columns and self._low_col in datos.columns:
            high = datos[self._high_col].astype(float)
            low  = datos[self._low_col].astype(float)
            true_range = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = true_range.ewm(span=14, adjust=False).mean()
            f["atr_norm_14"] = atr / close.replace(0.0, np.nan)

        # ── Ratio de volumen relativo (requiere Volume) ────────────────────────
        if self._volume_col in datos.columns:
            vol    = datos[self._volume_col].astype(float)
            vol_ma = vol.rolling(self._ventana_vol,
                                 min_periods=self._ventana_vol).mean()
            f["vol_ratio_21"] = vol / vol_ma.replace(0.0, np.nan) - 1.0

        return f

    # ──────────────────────────────────────────────────────────────────────────
    # Entrenamiento + predicción (ventana expansiva, causal)
    # ──────────────────────────────────────────────────────────────────────────

    def _train_predict(
        self,
        features: pd.DataFrame,
        close: pd.Series,
    ) -> float:
        """
        Entrena el modelo sobre el historial completo disponible y predice
        el retorno esperado para la ÚLTIMA fila (= fecha_corte).

        Partición de datos (garantía de causalidad):

            Índice de fila  │  features (X)  │  target (y)
            ────────────────┼────────────────┼──────────────────────
            0 … N-202       │  algunos NaN   │  log_ret[1..N-201]
            N-201           │  completo      │  log_ret[N-200]
            …               │  …             │  …
            N-2             │  completo      │  log_ret[N-1]  ← último par válido
            N-1 (fecha_corte│  completo      │  NaN (futuro desconocido)

        dropna() elimina automáticamente las filas con NaN en features O en target.
        La fila N-1 (fecha_corte) se excluye del entrenamiento porque target = NaN.
        Se usa SOLO para la predicción -> cero look-ahead.

        Returns:
            float: retorno log diario predicho para el día siguiente a fecha_corte.
                   Retorna 0.0 si no hay suficientes datos o la fila de pred tiene NaN.
        """
        log_ret = np.log(close / close.shift(1))
        target  = log_ret.shift(-1)       # target[t] = retorno de t+1

        # ── Conjunto de entrenamiento: X y y ambos no-NaN ─────────────────────
        df_tr = features.copy()
        df_tr["__y__"] = target
        df_tr = df_tr.dropna()            # excluye warm-up Y la última fila (target NaN)

        if len(df_tr) < self._min_train:
            return 0.0                    # datos históricos insuficientes

        X_train = df_tr.drop(columns="__y__").values
        y_train = df_tr["__y__"].values

        # ── Fila de predicción (fecha_corte) ──────────────────────────────────
        x_pred_row = features.iloc[[-1]]
        if x_pred_row.isnull().any(axis=1).values[0]:
            return 0.0                    # warm-up incompleto para la fecha actual

        X_pred = x_pred_row.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._modelo.fit(X_train, y_train)

        return float(self._modelo.predict(X_pred)[0])

    # ──────────────────────────────────────────────────────────────────────────
    # AbstractStrategy: generar_señales
    # ──────────────────────────────────────────────────────────────────────────

    def generar_señales(
        self,
        datos: pd.DataFrame,
        fecha_corte: pd.Timestamp,
    ) -> pd.Series:
        """
        Genera la señal de posición para el período siguiente a fecha_corte.

        El BacktestEngine (ZooManager) garantiza que datos ya está filtrado
        a datos <= fecha_corte; aquí se aplica un segundo filtro defensivo.

        Flujo:
            datos[:fecha_corte]
                -> _build_features()   [features causales]
                -> _train_predict()    [XGBoost, ventana expansiva]
                -> discretización      [pred > umbral -> 1.0; pred < -umbral -> -1.0]
                -> pd.Series [ticker -> peso]

        Returns:
            pd.Series indexada por self.universo.
            Valores por activo: 1.0 (largo), 0.0 (neutral), -1.0 (corto).
        """
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        # Corte causal (defensivo — ZooManager ya lo aplica)
        hist = datos.loc[datos.index <= fecha_corte]

        # Verificaciones rápidas antes de construir features
        if self._close_col not in hist.columns:
            return neutral

        # Warm-up mínimo: 200 (SMA200) + min_train_days filas completas + 1 (pred)
        if len(hist) < self._min_train + 200:
            return neutral

        features    = self._build_features(hist)
        close       = hist[self._close_col].astype(float)
        pred_return = self._train_predict(features, close)

        if pred_return > self._umbral:
            señal = 1.0
        elif pred_return < -self._umbral:
            señal = -1.0
        else:
            señal = 0.0

        pesos = pd.Series(0.0, index=indice, dtype=float)
        if self._universo:
            pesos.iloc[0] = señal      # primer (y único) ticker recibe la señal

        return pesos

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: serie temporal de señales (testing / visualización)
    # ──────────────────────────────────────────────────────────────────────────

    def generar_serie_temporal(
        self,
        datos: pd.DataFrame,
        fechas: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        Genera la serie temporal de señales para un conjunto de fechas.

        Llama a generar_señales() en cada fecha de forma walk-forward; cada
        iteración usa exclusivamente datos <= fecha (causalidad garantizada).

        Args:
            datos:  DataFrame OHLCV con DatetimeIndex.
            fechas: Fechas objetivo. Si None, usa datos.index completo.

        Returns:
            pd.Series [fecha -> señal] con valores in {-1.0, 0.0, 1.0}.
            El índice es exactamente igual a `fechas` (o datos.index si None).
        """
        if fechas is None:
            fechas = datos.index

        señales: Dict[pd.Timestamp, float] = {}
        for fecha in fechas:
            s              = self.generar_señales(datos, fecha)
            señales[fecha] = float(s.iloc[0]) if not s.empty else 0.0

        return pd.Series(señales, dtype=float, name="señal_xgboost")

    # ──────────────────────────────────────────────────────────────────────────
    # AbstractStrategy: calcular_retornos
    # ──────────────────────────────────────────────────────────────────────────

    def calcular_retornos(
        self,
        datos: pd.DataFrame,
        pesos_historicos: pd.DataFrame,
    ) -> pd.Series:
        """
        Aplica las señales históricas de posición a los retornos del activo.

        Fórmula: ret_portafolio[t] = señal[t-1] · ret_activo[t]

        Los pesos de rebalanceo se extienden a frecuencia diaria mediante
        forward-fill antes de llegar aquí (responsabilidad del BacktestEngine).

        Args:
            datos:            DataFrame con al menos close_col.
            pesos_historicos: DataFrame [fecha_rebalanceo × ticker] con señales.

        Returns:
            pd.Series de retornos diarios netos de la estrategia.
        """
        if self._close_col not in datos.columns:
            return pd.Series(dtype=float)

        if not self._universo:
            return pd.Series(dtype=float)

        ticker = self._universo[0]
        if ticker not in pesos_historicos.columns:
            return pd.Series(dtype=float)

        ret_activo           = datos[self._close_col].pct_change()
        pesos                = pesos_historicos[ticker]
        pesos_al, ret_al     = pesos.align(ret_activo, join="inner")
        return (pesos_al * ret_al).dropna()


# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER        = "SPXTEST"
    VALID_SIGNALS = {-1.0, 0.0, 1.0}

    # ── Helper: genera OHLCV sintético via random walk log-normal ────────────

    def _make_ohlcv(
        n: int,
        mu: float    = 3e-4,
        sigma: float = 0.012,
        start: str   = "2010-01-01",
        seed: int    = 42,
    ) -> pd.DataFrame:
        rng_l  = np.random.default_rng(seed)
        dates  = pd.date_range(start, periods=n, freq="B")
        lrets  = rng_l.normal(mu, sigma, n)
        close  = 1_000.0 * np.exp(np.cumsum(lrets))
        noise  = rng_l.uniform(0.001, 0.008, n)
        return pd.DataFrame({
            "Close":  close,
            "Open":   close * (1 + rng_l.normal(0, 0.002, n)),
            "High":   close * (1 + np.abs(rng_l.normal(0, noise))),
            "Low":    close * (1 - np.abs(rng_l.normal(0, noise))),
            "Volume": rng_l.integers(1_000_000, 20_000_000, n).astype(float),
        }, index=dates)

    # ── Helper: verificar contrato de señal ──────────────────────────────────

    def _check_señal(s: pd.Series, label: str, expected_ticker: str) -> None:
        assert isinstance(s, pd.Series), \
            f"[{label}] Output no es pd.Series."
        assert not s.isnull().any(), \
            f"[{label}] Output contiene NaN: {s[s.isnull()]}."
        bad = set(s.values) - VALID_SIGNALS
        assert not bad, \
            f"[{label}] Valores fuera de {{-1,0,1}}: {bad}."
        assert expected_ticker in s.index, \
            f"[{label}] Ticker '{expected_ticker}' no está en el índice de la señal."

    # ─────────────────────────────────────────────────────────────────────────
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("  XGBoostStrategy — Suite de Pruebas Unitarias  [7 tests]")
    print(SEP)

    strat = XGBoostStrategy(
        universo      = [TICKER],
        min_train_days = 252,
        umbral_señal   = 0.003,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1 — Propiedades básicas de la instancia
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre      == "xgboost_trend",      "nombre incorrecto."
    assert strat.universo    == [TICKER],              "universo incorrecto."
    assert isinstance(strat.descripcion, str),         "descripcion debe ser str."
    assert len(strat.descripcion) > 0,                 "descripcion vacía."
    print("  [OK] nombre='xgboost_trend' | universo correcto | descripcion no vacía.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2 — Período de warm-up: señal debe ser neutral (datos insuficientes)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 2] Warm-up (n=300 < 452) -> señal neutral esperada...")
    df_short  = _make_ohlcv(n=300, seed=1)
    s_short   = strat.generar_señales(df_short, df_short.index[-1])
    _check_señal(s_short, "TEST2", TICKER)
    assert s_short[TICKER] == 0.0, \
        f"Con datos insuficientes esperaba 0.0, obtuvo {s_short[TICKER]}."
    print(f"  [OK] Señal={s_short[TICKER]} (neutral) — warm-up respetado.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3 — Mercado random walk (~4 años de historia)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 3] Random walk (n=1 000 días)...")
    df_rw = _make_ohlcv(n=1_000, seed=42)
    s_rw  = strat.generar_señales(df_rw, df_rw.index[-1])
    _check_señal(s_rw, "TEST3", TICKER)
    print(f"  [OK] Señal={s_rw[TICKER]} in {{-1.0, 0.0, 1.0}}. Sin excepción.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4 — Mercado alcista fuerte (mu = +0.2 %/día)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 4] Mercado alcista fuerte (mu=+0.002/día)...")
    df_bull = _make_ohlcv(n=1_000, mu=0.002, sigma=0.008, seed=7)
    s_bull  = strat.generar_señales(df_bull, df_bull.index[-1])
    _check_señal(s_bull, "TEST4", TICKER)
    print(f"  [OK] Señal={s_bull[TICKER]}. Modelo ejecutado sin errores en tendencia.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5 — Volatilidad extrema (sigma = 5 %/día, crash-like)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 5] Volatilidad extrema (sigma=0.05/día, mu=-0.001)...")
    df_crash = _make_ohlcv(n=1_000, mu=-0.001, sigma=0.05, seed=99)
    s_crash  = strat.generar_señales(df_crash, df_crash.index[-1])
    _check_señal(s_crash, "TEST5", TICKER)
    print(f"  [OK] Señal={s_crash[TICKER]}. Sin crash de código con vol extrema.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6 — Serie temporal completa: índices, NaN, discreción
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 6] Serie temporal (n=600, últimas 30 fechas)...")
    df_full     = _make_ohlcv(n=600, seed=42)
    fechas_test = df_full.index[-30:]                # 30 fechas post warm-up
    serie       = strat.generar_serie_temporal(df_full, fechas=fechas_test)

    assert isinstance(serie, pd.Series), \
        "generar_serie_temporal debe retornar pd.Series."
    assert len(serie) == len(fechas_test), \
        f"Longitud mismatch: {len(serie)} vs {len(fechas_test)}."
    assert serie.index.equals(pd.DatetimeIndex(fechas_test)), \
        "El índice de la serie NO coincide exactamente con las fechas de entrada."
    assert not serie.isnull().any(), \
        f"NaN encontrados: {serie[serie.isnull()]}."
    bad_vals = set(serie.values) - VALID_SIGNALS
    assert not bad_vals, \
        f"Valores fuera de {{-1,0,1}}: {bad_vals}."

    n_long    = int((serie ==  1.0).sum())
    n_short   = int((serie == -1.0).sum())
    n_neutral = int((serie ==  0.0).sum())
    print(f"  [OK] 30 fechas | Largo={n_long} | Corto={n_short} | Neutral={n_neutral}")
    print(f"       Índice == fechas de entrada  OK")
    print(f"       Sin NaN                      OK")
    print(f"       Valores subset of {{-1.0, 0.0, 1.0}} OK")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 7 — DataFrame sin columna 'Close': neutral sin excepción
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[TEST 7] DataFrame sin columna 'Close' -> neutral sin crash...")
    df_bad = df_full.rename(columns={"Close": "precio_cierre"})
    s_bad  = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s_bad.isnull().any(),          "NaN inesperado en señal con bad data."
    assert (s_bad == 0.0).all(),              "Esperaba neutral cuando falta close_col."
    print(f"  [OK] Señal neutral retornada sin lanzar excepción.")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON EXITOSAMENTE  [7 / 7]  OK")
    print(f"{SEP}\n")
