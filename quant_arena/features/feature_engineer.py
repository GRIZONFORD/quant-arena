# feature_engineer.py
"""
FeatureEngineer — Batería de indicadores técnicos vectorizados (DIARIO / 1D)
============================================================================
Pipeline de Feature Engineering para datos OHLCV de FRECUENCIA DIARIA (1 barra =
1 día de mercado), diseñado para mitigar el sobreajuste de modelos ML (XGBoost,
etc.) que solo disponen de las cinco columnas crudas open/high/low/close/volume.

FRECUENCIA: DIARIA (1D) — EXCLUSIVAMENTE
----------------------------------------
Este módulo NO soporta intradía. Todas las ventanas se expresan en DÍAS de
mercado con su significado físico natural (RSI 14 días, SMA 50/200 días, etc.)
y la volatilidad se anualiza con 252 días hábiles. No existe ninguna conversión
"días → barras"; una fila es siempre un día.

PRINCIPIOS DE DISEÑO
--------------------
1. 100% VECTORIZADO. Ni un solo bucle `for` sobre filas. Toda la matemática se
   apoya en operaciones de Pandas/NumPy (rolling, ewm, shift, cumsum, where).
2. SIN LOOK-AHEAD BIAS. Todos los indicadores usan exclusivamente información
   pasada o presente:
     · `rolling(w)` y `ewm(...)` están alineados a la derecha (trailing).
     · `shift(k)` con k > 0 trae valores estrictamente pasados.
   En consecuencia, los únicos NaN aparecen en el periodo de calentamiento
   (warm-up) al inicio de la serie, nunca en el interior. Recortar esas filas
   iniciales es seguro y NO introduce sesgo de anticipación.

Familias de indicadores generadas (ver README → "Arquitectura de Datos"):
  · Retornos rezagados (lagged returns)
  · Momentum (RSI, ROC, MACD, Estocástico)
  · Tendencia (SMA/EMA ratios, pendiente)
  · Volatilidad (rolling std, ATR, Bollinger, Parkinson)
  · Volumen (z-score, OBV, desviación de VWAP)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    """
    Parámetros de las ventanas de los indicadores, en DÍAS de mercado (1D).

    Todos los valores son periodos diarios con significado físico estándar:
    RSI 14 días, MACD 12/26/9 días, SMA 50/200 días (cruce dorado/de la muerte),
    volatilidad de 21 días (~1 mes bursátil) anualizada con `trading_days`.
    """
    trading_days:     int = 252          # días hábiles/año → anualización de vol
    return_lags:      List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    sma_windows:      List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_windows:      List[int] = field(default_factory=lambda: [12, 26])
    rsi_window:       int = 14           # RSI de 14 días (Wilder)
    roc_window:       int = 10           # Rate of Change de 10 días
    momentum_window:  int = 10           # Momentum de 10 días
    macd_fast:        int = 12           # MACD diario estándar
    macd_slow:        int = 26
    macd_signal:      int = 9
    stoch_window:     int = 14           # Estocástico de 14 días
    stoch_smooth:     int = 3
    vol_window:       int = 20           # Vol. de ~1 mes (20 días)
    atr_window:       int = 14           # ATR de 14 días
    bb_window:        int = 20           # Bollinger de 20 días
    bb_n_std:         float = 2.0
    parkinson_window: int = 20           # Parkinson de 20 días
    realized_window:  int = 21           # Vol. realizada de ~1 mes (21 días)
    volume_window:    int = 20           # Estadísticos de volumen de 20 días

    @property
    def max_lookback(self) -> int:
        """Lookback más largo (en días): define el warm-up a recortar (~200)."""
        return max(
            max(self.return_lags),
            max(self.sma_windows),
            max(self.ema_windows),
            self.rsi_window,
            self.roc_window,
            self.momentum_window,
            self.macd_slow + self.macd_signal,
            self.stoch_window + self.stoch_smooth,
            self.vol_window,
            self.atr_window,
            self.bb_window,
            self.parkinson_window,
            self.realized_window,
            self.volume_window,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Calcula una batería de indicadores técnicos de forma 100% vectorizada.

    Uso típico:
        fe = FeatureEngineer()
        df_feat = fe.transform(df_ohlcv)            # añade ~28 columnas
        df_clean = fe.transform(df_ohlcv, trim_warmup=True)  # + recorta warm-up

    El método `transform` es idempotente respecto al sesgo temporal: nunca usa
    información futura. La única limpieza que aplica es recortar el periodo de
    calentamiento inicial (filas con NaN por lookback insuficiente).

    Args:
        config: FeatureConfig con las ventanas. Si None, usa los valores default.
    """

    _REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        trim_warmup: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Genera todos los features técnicos sobre un DataFrame OHLCV.

        Args:
            df:          DataFrame con columnas open/high/low/close/volume y
                         DatetimeIndex monótono creciente.
            trim_warmup: Si True, recorta de forma segura las filas iniciales
                         del periodo de calentamiento (NaN por lookback). NO
                         introduce look-ahead bias (solo descarta el inicio).
            inplace:     Si False (default), opera sobre una copia.

        Returns:
            DataFrame con las columnas OHLCV originales + los indicadores.
        """
        self._validate(df)
        out = df if inplace else df.copy()

        # Cada familia añade columnas in-place sobre `out`. Todas las llamadas
        # son vectorizadas (rolling / ewm / shift / cumsum).
        self._add_returns(out)
        self._add_momentum(out)
        self._add_trend(out)
        self._add_volatility(out)
        self._add_volume(out)

        new_cols = [c for c in out.columns if c not in self._REQUIRED_COLS]
        logger.info(
            "FeatureEngineer.transform: %d features añadidos | filas=%d",
            len(new_cols), len(out),
        )

        if trim_warmup:
            out = self.trim_warmup(out)

        return out

    def trim_warmup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recorta SOLO el periodo de calentamiento inicial de forma segura.

        Como todos los indicadores son trailing, los NaN se concentran en las
        primeras `max_lookback` filas. Eliminar esas filas iniciales es la única
        forma de limpieza que respeta la causalidad temporal. Se hace por
        posición (head) y no por `dropna` global para evitar borrar días
        interiores legítimos con NaN puntual de origen.
        """
        warmup = self.config.max_lookback
        n_before = len(df)
        trimmed = df.iloc[warmup:].copy()

        # Salvaguarda: si quedaran NaN interiores residuales (gaps de datos),
        # se reportan pero NO se interpolan hacia atrás (eso sí sería look-ahead).
        residual_nans = int(trimmed.isna().sum().sum())
        logger.info(
            "trim_warmup: recortadas %d filas de warm-up (lookback=%d). "
            "Filas restantes=%d | NaN residuales=%d",
            n_before - len(trimmed), warmup, len(trimmed), residual_nans,
        )
        return trimmed

    def feature_names(self) -> List[str]:
        """Devuelve la lista de nombres de features que produce `transform`."""
        cfg = self.config
        names: List[str] = []
        names += [f"ret_lag_{k}" for k in cfg.return_lags]
        names += ["log_return", "rsi", "roc", "momentum",
                  "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d"]
        names += [f"sma_ratio_{w}" for w in cfg.sma_windows]
        names += [f"ema_ratio_{w}" for w in cfg.ema_windows]
        names += ["sma_slope", "volatility", "atr", "bb_width", "bb_pctb",
                  "parkinson_vol", "realized_vol",
                  "volume_zscore", "volume_roc", "obv", "vwap_dev"]
        return names

    # ──────────────────────────────────────────────────────────────────────────
    # FAMILIA 1 · RETORNOS REZAGADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame) -> None:
        close = df["close"].astype(float)
        # Retorno logarítmico base (presente, no es feature predictivo de fuga).
        log_ret = np.log(close / close.shift(1))
        df["log_return"] = log_ret
        # Retornos rezagados: valores ESTRICTAMENTE pasados (shift k>0).
        for k in self.config.return_lags:
            df[f"ret_lag_{k}"] = log_ret.shift(k)

    # ──────────────────────────────────────────────────────────────────────────
    # FAMILIA 2 · MOMENTUM
    # ──────────────────────────────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> None:
        cfg = self.config
        close = df["close"].astype(float)

        # ── RSI (Wilder) vectorizado vía EWM ──────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / cfg.rsi_window, min_periods=cfg.rsi_window).mean()
        avg_loss = loss.ewm(alpha=1.0 / cfg.rsi_window, min_periods=cfg.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # ── Rate of Change y Momentum ─────────────────────────────────────
        df["roc"] = close.pct_change(cfg.roc_window)
        df["momentum"] = close - close.shift(cfg.momentum_window)

        # ── MACD (EMA rápida - EMA lenta) ─────────────────────────────────
        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = macd - signal

        # ── Oscilador estocástico %K / %D ─────────────────────────────────
        low_min = df["low"].rolling(cfg.stoch_window).min()
        high_max = df["high"].rolling(cfg.stoch_window).max()
        denom = (high_max - low_min).replace(0.0, np.nan)
        stoch_k = 100.0 * (close - low_min) / denom
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_k.rolling(cfg.stoch_smooth).mean()

    # ──────────────────────────────────────────────────────────────────────────
    # FAMILIA 3 · TENDENCIA
    # ──────────────────────────────────────────────────────────────────────────

    def _add_trend(self, df: pd.DataFrame) -> None:
        cfg = self.config
        close = df["close"].astype(float)

        # Distancia relativa del precio a sus medias (estacionaria, sin escala).
        for w in cfg.sma_windows:
            sma = close.rolling(w).mean()
            df[f"sma_ratio_{w}"] = close / sma - 1.0
        for w in cfg.ema_windows:
            ema = close.ewm(span=w, adjust=False).mean()
            df[f"ema_ratio_{w}"] = close / ema - 1.0

        # Pendiente de la SMA media (normalizada) como proxy de fuerza de tendencia.
        sma_ref = close.rolling(cfg.sma_windows[1]).mean()
        df["sma_slope"] = sma_ref.pct_change()

    # ──────────────────────────────────────────────────────────────────────────
    # FAMILIA 4 · VOLATILIDAD
    # ──────────────────────────────────────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame) -> None:
        cfg = self.config
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        log_ret = df["log_return"]

        # Volatilidad realizada (rolling std de retornos diarios).
        df["volatility"] = log_ret.rolling(cfg.vol_window).std()
        # Anualización DIARIA correcta: σ_diaria · √(252 días hábiles).
        df["realized_vol"] = (
            log_ret.rolling(cfg.realized_window).std() * np.sqrt(cfg.trading_days)
        )

        # ── ATR (Average True Range) vectorizado ──────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / cfg.atr_window, min_periods=cfg.atr_window).mean()

        # ── Bandas de Bollinger: ancho y %B ───────────────────────────────
        mid = close.rolling(cfg.bb_window).mean()
        std = close.rolling(cfg.bb_window).std()
        upper = mid + cfg.bb_n_std * std
        lower = mid - cfg.bb_n_std * std
        width = (upper - lower)
        df["bb_width"] = width / mid                      # ancho normalizado
        df["bb_pctb"] = (close - lower) / width.replace(0.0, np.nan)

        # ── Volatilidad de Parkinson (basada en rango high/low) ───────────
        hl_ratio = np.log(high / low) ** 2
        df["parkinson_vol"] = np.sqrt(
            hl_ratio.rolling(cfg.parkinson_window).mean() / (4.0 * np.log(2.0))
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FAMILIA 5 · VOLUMEN
    # ──────────────────────────────────────────────────────────────────────────

    def _add_volume(self, df: pd.DataFrame) -> None:
        cfg = self.config
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Z-score del volumen (anomalías de actividad).
        vol_mean = volume.rolling(cfg.volume_window).mean()
        vol_std = volume.rolling(cfg.volume_window).std()
        df["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)

        # Rate of change del volumen.
        df["volume_roc"] = volume.pct_change(cfg.volume_window)

        # On-Balance Volume (OBV) vectorizado: signo del retorno * volumen, acumulado.
        direction = np.sign(close.diff()).fillna(0.0)
        df["obv"] = (direction * volume).cumsum()

        # Desviación del precio respecto al VWAP rolling.
        pv = (close * volume).rolling(cfg.volume_window).sum()
        vv = volume.rolling(cfg.volume_window).sum().replace(0.0, np.nan)
        vwap = pv / vv
        df["vwap_dev"] = close / vwap - 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN
    # ──────────────────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self._REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"FeatureEngineer: columnas OHLCV ausentes: {missing}. "
                f"Disponibles: {list(df.columns)}"
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("FeatureEngineer: el índice debe ser DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "FeatureEngineer: el índice no es monótono creciente. "
                "Ordenar por fecha antes de calcular features (evita look-ahead)."
            )
