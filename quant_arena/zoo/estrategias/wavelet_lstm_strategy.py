#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/wavelet_lstm_strategy.py
# Wavelet-LSTM Strategy — Reversion a la media en datos denoised
# =============================================================================
"""
Estrategia que combina Transformada Wavelet Discreta (DWT) con LSTM para
capturar alfa de reversion a la media en series de precios filtradas.

Pipeline:
    precios -> DWT multi-nivel (db4) -> soft thresholding -> IDWT
    -> serie denoised -> features causales -> LSTM -> señal

La fase DWT actua como filtro paso-bajo causal sobre la serie historica,
eliminando el ruido de microestructura (bid-ask bounce, outliers intradiarios)
y manteniendo los componentes de frecuencia baja y media que representan
las tendencias y ciclos de inversion relevantes.

Registro: 'wavelet_lstm'
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import pywt
    _PWT_OK = True
except ImportError:
    _PWT_OK = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo


# =============================================================================
# Denoiser Wavelet
# =============================================================================

class WaveletDenoiser:
    """
    Denoiser causal mediante DWT multi-nivel con umbral VisuShrink.

    Aplica la Transformada Wavelet Discreta (DWT) a la serie completa hasta
    el punto de corte, umbrea los coeficientes de detalle con soft-thresholding
    y reconstruye mediante IDWT.

    El umbral de VisuShrink es: T = sigma * sqrt(2 * log(n))
    donde sigma se estima del nivel de detalle mas fino con el estimador robusto
    de la mediana absoluta: sigma = MAD(d1) / 0.6745.

    Args:
        wavelet:     Familia wavelet. 'db4' (Daubechies 4) es estandar en finanzas.
        nivel:       Numero de niveles de descomposicion.
        modo:        Modo de extension de bordes. 'periodization' evita artefactos.
        threshold_mode: 'soft' | 'hard'. Soft produce series mas suaves.
    """

    def __init__(
        self,
        wavelet:        str = "db4",
        nivel:          int = 3,
        modo:           str = "periodization",
        threshold_mode: str = "soft",
    ) -> None:
        if not _PWT_OK:
            raise ImportError("pip install PyWavelets")
        self._wavelet        = wavelet
        self._nivel          = nivel
        self._modo           = modo
        self._threshold_mode = threshold_mode

    def denoise(self, serie: np.ndarray) -> np.ndarray:
        """
        Aplica DWT -> thresholding -> IDWT a la serie de entrada.

        Args:
            serie: array 1D float, serie de precios o log-retornos.

        Returns:
            serie_denoised: array 1D float del mismo largo que serie.
        """
        n     = len(serie)
        nivel = min(self._nivel, pywt.dwt_max_level(n, self._wavelet))
        if nivel == 0:
            return serie.copy()

        # Descomposicion multi-nivel
        coefs = pywt.wavedec(serie, self._wavelet, mode=self._modo, level=nivel)

        # Estimar sigma del nivel de detalle mas fino (coefs[1])
        detail_finest = coefs[1]
        sigma = np.median(np.abs(detail_finest)) / 0.6745
        if sigma < 1e-10:
            return serie.copy()

        # Umbral VisuShrink universal
        threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

        # Umbralado de todos los coeficientes de detalle (no la aproximacion)
        coefs_thresh = [coefs[0]] + [
            pywt.threshold(c, threshold, mode=self._threshold_mode)
            for c in coefs[1:]
        ]

        # Reconstruccion
        denoised = pywt.waverec(coefs_thresh, self._wavelet, mode=self._modo)

        # Asegurar mismo largo (waverec puede agregar un punto por relleno)
        return denoised[:n]


# =============================================================================
# Modelo LSTM
# =============================================================================

class LSTMRegressor(nn.Module):
    """
    LSTM de regresion para predecir retornos a partir de features denoised.

    Args:
        n_features:  Dimension de entrada.
        hidden_size: Dimension estado oculto LSTM.
        n_layers:    Capas LSTM apiladas.
        dropout:     Dropout inter-capas (solo si n_layers > 1).
    """
    def __init__(
        self,
        n_features:  int,
        hidden_size: int   = 64,
        n_layers:    int   = 2,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        out, _ = self.lstm(x)          # (B, T, H)
        last   = self.norm(out[:, -1, :])
        return self.head(last).squeeze(-1)


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("wavelet_lstm")
class WaveletLSTMStrategy(AbstractStrategy):
    """
    Estrategia Wavelet-LSTM para quant_arena.

    Aplica un filtro wavelet causal a la serie de precios para eliminar
    ruido de microestructura antes de construir features y entrenar el LSTM.
    Esta arquitectura es especialmente efectiva en periodos de baja volatilidad
    donde la señal de reversion esta enmascarada por ruido transaccional.

    Args:
        universo:             Tickers (un elemento).
        min_train_days:       Minimo de muestras de entrenamiento.
        lookback:             Longitud de la secuencia LSTM.
        wavelet:              Familia wavelet para el denoiser.
        wavelet_nivel:        Niveles de descomposicion DWT.
        hidden_size:          Dimension del estado LSTM.
        n_lstm_layers:        Capas LSTM apiladas.
        umbral_señal:         Umbral de discretizacion.
        max_epochs:           Epocas de entrenamiento.
        batch_size:           Mini-batch.
        lr:                   Learning rate.
        retrain_every_n_days: Dias entre re-entrenamientos.
        close_col:            Columna de cierre.
        volume_col:           Columna de volumen.
        device:               Dispositivo de computo.
        seed:                 Semilla.
    """

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        lookback:             int   = 42,
        wavelet:              str   = "db4",
        wavelet_nivel:        int   = 3,
        hidden_size:          int   = 64,
        n_lstm_layers:        int   = 2,
        umbral_señal:         float = 0.003,
        max_epochs:           int   = 20,
        batch_size:           int   = 32,
        lr:                   float = 1e-3,
        retrain_every_n_days: int   = 63,
        close_col:            str   = "Close",
        volume_col:           str   = "Volume",
        high_col:             str   = "High",
        low_col:              str   = "Low",
        device:               Optional[str] = None,
        seed:                 int   = 42,
    ) -> None:
        if not _PWT_OK:
            raise ImportError("pip install PyWavelets")
        if not _TORCH_OK:
            raise ImportError("pip install torch")

        super().__init__(nombre="wavelet_lstm", universo=universo)

        self._min_train    = min_train_days
        self._lookback     = lookback
        self._hidden_size  = hidden_size
        self._n_lstm       = n_lstm_layers
        self._umbral       = umbral_señal
        self._max_epochs   = max_epochs
        self._batch_size   = batch_size
        self._lr           = lr
        self._retrain_days = retrain_every_n_days
        self._close_col    = close_col
        self._volume_col   = volume_col
        self._high_col     = high_col
        self._low_col      = low_col
        self._seed         = seed
        self._device: str  = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._denoiser = WaveletDenoiser(wavelet=wavelet, nivel=wavelet_nivel)
        self._net: Optional[LSTMRegressor] = None
        self._n_features: Optional[int]    = None
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

    @property
    def descripcion(self) -> str:
        return (
            f"Wavelet-LSTM | wavelet=db4 | lookback={self._lookback}d "
            f"| hidden={self._hidden_size} | umbral=+-{self._umbral:.4f}"
        )

    # ------------------------------------------------------------------
    def _denoise_close(self, close: pd.Series) -> pd.Series:
        """
        Aplica denoising a la serie de precios de cierre.

        El denoising se realiza sobre la serie completa disponible hasta
        fecha_corte. Es causal porque no usa datos futuros.

        Returns:
            pd.Series con los precios denoised, mismo indice que close.
        """
        close_arr     = close.values.astype(np.float64)
        denoised_arr  = self._denoiser.denoise(close_arr)
        denoised_arr  = np.clip(denoised_arr, 1e-8, None)   # precios > 0
        return pd.Series(denoised_arr, index=close.index, name=close.name)

    def _build_features(self, close_raw: pd.Series, close_den: pd.Series,
                         datos: pd.DataFrame) -> pd.DataFrame:
        """
        Construye features usando tanto el precio raw como el denoised.

        Los retornos denoised capturan la componente de baja frecuencia
        (tendencias) mientras que la diferencia raw-denoised mide el ruido
        (revertion proxy).
        """
        log_ret_raw = np.log(close_raw / close_raw.shift(1))
        log_ret_den = np.log(close_den / close_den.shift(1))

        f = pd.DataFrame(index=datos.index)
        f["ret_1d_raw"]  = log_ret_raw
        f["ret_5d_raw"]  = np.log(close_raw / close_raw.shift(5))
        f["ret_21d_raw"] = np.log(close_raw / close_raw.shift(21))
        f["ret_1d_den"]  = log_ret_den
        f["ret_5d_den"]  = np.log(close_den / close_den.shift(5))

        # Ruido = diferencia entre serie raw y denoised (proxy de microestructura)
        noise = (close_raw - close_den) / close_den.replace(0.0, np.nan)
        f["noise_ratio"] = noise
        f["noise_5d_std"] = noise.rolling(5, min_periods=5).std()

        f["vol_21d"] = log_ret_raw.rolling(21, min_periods=21).std()

        d  = close_raw.diff()
        ag = d.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
        al = (-d).clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
        f["rsi_14"] = (100.0 - 100.0 / (1.0 + ag / al.replace(0.0, np.nan))) / 100.0

        sma50  = close_den.rolling(50, min_periods=50).mean()
        sma200 = close_den.rolling(200, min_periods=200).mean()
        f["price_sma50_den"]  = close_den / sma50  - 1.0
        f["sma50_sma200_den"] = sma50 / sma200 - 1.0

        if self._volume_col in datos.columns:
            vol    = datos[self._volume_col].astype(float)
            vol_ma = vol.rolling(21, min_periods=21).mean()
            f["vol_ratio"] = vol / vol_ma.replace(0.0, np.nan) - 1.0

        return f

    def _construir_dataset(
        self, features: pd.DataFrame, close: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_ret = np.log(close / close.shift(1))
        df      = features.copy()
        df["__y__"] = log_ret.shift(-1)
        df_clean    = df.dropna()

        feat_np = df_clean.drop(columns="__y__").values.astype(np.float32)
        tgt_np  = df_clean["__y__"].values.astype(np.float32)

        X_train, y_train = [], []
        for i in range(self._lookback, len(feat_np)):
            X_train.append(feat_np[i - self._lookback : i])
            y_train.append(tgt_np[i])

        feat_pred = features.dropna().values[-self._lookback :].astype(np.float32)
        return (
            np.array(X_train, dtype=np.float32),
            np.array(y_train, dtype=np.float32),
            feat_pred[np.newaxis, ...],
        )

    def _entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        n_features = X.shape[-1]

        if self._net is None or self._n_features != n_features:
            self._net = LSTMRegressor(
                n_features  = n_features,
                hidden_size = self._hidden_size,
                n_layers    = self._n_lstm,
            ).to(self._device)
            self._n_features = n_features

        X_t      = torch.from_numpy(X).to(self._device)
        y_t      = torch.from_numpy(y).to(self._device)
        loader   = DataLoader(TensorDataset(X_t, y_t), batch_size=self._batch_size, shuffle=True)
        opt      = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        crit     = nn.MSELoss()

        self._net.train()
        for _ in range(self._max_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(self._net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()

    def _predecir(self, X_pred: np.ndarray) -> float:
        if self._net is None:
            return 0.0
        self._net.eval()
        with torch.no_grad():
            pred = self._net(torch.from_numpy(X_pred).to(self._device))
        return float(pred.cpu().numpy()[0])

    # ------------------------------------------------------------------
    def generar_señales(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return neutral
        if len(hist) < self._min_train + self._lookback + 200:
            return neutral

        close_raw = hist[self._close_col].astype(float)
        try:
            close_den = self._denoise_close(close_raw)
            features  = self._build_features(close_raw, close_den, hist)
            X_train, y_train, X_pred = self._construir_dataset(features, close_raw)
        except Exception:
            return neutral

        if len(X_train) < self._min_train or X_pred.shape[1] < self._lookback:
            return neutral

        necesita = (
            self._ultimo_entrenamiento is None
            or (fecha_corte - self._ultimo_entrenamiento).days >= self._retrain_days
        )
        if necesita:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._entrenar(X_train, y_train)
            self._ultimo_entrenamiento = fecha_corte

        pred_ret = self._predecir(X_pred)
        señal    = 1.0 if pred_ret > self._umbral else (-1.0 if pred_ret < -self._umbral else 0.0)

        pesos = pd.Series(0.0, index=indice, dtype=float)
        if self._universo:
            pesos.iloc[0] = señal
        return pesos

    def calcular_retornos(
        self, datos: pd.DataFrame, pesos_historicos: pd.DataFrame
    ) -> pd.Series:
        if self._close_col not in datos.columns or not self._universo:
            return pd.Series(dtype=float)
        ticker = self._universo[0]
        if ticker not in pesos_historicos.columns:
            return pd.Series(dtype=float)
        ret_al, pesos_al = datos[self._close_col].pct_change().align(
            pesos_historicos[ticker], join="inner"
        )
        return (pesos_al * ret_al).dropna()


# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "WLSTMTEST"

    def _make_ohlcv(n, mu=3e-4, sigma=0.012, start="2010-01-01", seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start, periods=n, freq="B")
        close = 1_000.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
        noise = rng.uniform(0.001, 0.008, n)
        return pd.DataFrame({
            "Close": close,
            "Open":  close * (1 + rng.normal(0, 0.002, n)),
            "High":  close * (1 + np.abs(rng.normal(0, noise))),
            "Low":   close * (1 - np.abs(rng.normal(0, noise))),
            "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
        }, index=dates)

    def _check(s, label, ticker, valid={-1.0, 0.0, 1.0}):
        assert isinstance(s, pd.Series),    f"[{label}] No es pd.Series"
        assert not s.isnull().any(),        f"[{label}] NaN en senal"
        bad = set(s.values) - valid
        assert not bad,                     f"[{label}] Valores fuera de {{-1,0,1}}: {bad}"
        assert ticker in s.index,           f"[{label}] Ticker no en indice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  WaveletLSTMStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = WaveletLSTMStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=42,
        max_epochs=3,
        hidden_size=32,
        retrain_every_n_days=9999,
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "wavelet_lstm"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='wavelet_lstm' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up (datos insuficientes)
    print("\n[TEST 2] Warm-up (n=300 < umbral efectivo)...")
    df_short = _make_ohlcv(n=300)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] senal={s[TICKER]} neutral -- warm-up respetado")

    # TEST 3 -- Operacion normal
    print("\n[TEST 3] Operacion normal (n=700)...")
    df_normal = _make_ohlcv(n=700, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] senal={s[TICKER]} in {{-1.0, 0.0, 1.0}}")

    # TEST 4 -- Regimen extremo (bull market fuerte)
    print("\n[TEST 4] Regimen bull extremo (mu=0.005, sigma=0.008)...")
    df_bull = _make_ohlcv(n=700, mu=0.005, sigma=0.008, seed=7)
    s = strat.generar_señales(df_bull, df_bull.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] senal={s[TICKER]} sin crash en bull extremo")

    # TEST 5 -- Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba senal neutral con datos malformados"
    print("  [OK] neutral retornado sin excepcion")

    # TEST 6 -- Boundary condition
    print("\n[TEST 6] Boundary condition (n=494, min_train+lookback+buffer)...")
    df_boundary = _make_ohlcv(n=494)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary respetado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
