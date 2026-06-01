#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/tft_strategy.py
# Temporal Fusion Transformer — macro-regimenes y atencion temporal multi-horizonte
# =============================================================================
"""
Implementacion del Temporal Fusion Transformer (Lim et al., 2021) adaptada
al pipeline de quant_arena para prediccion de retornos sobre series OHLCV.

Arquitectura:
    features(T, F) -> VSN -> LSTM encoder -> multi-head attention
                   -> gated residual -> output head -> señal ∈ {-1, 0, 1}

Registro: 'tft_trend'
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo


# =============================================================================
# Bloques neurales
# =============================================================================

class GatedResidualNetwork(nn.Module):
    """
    GRN = LayerNorm(skip(x) + GLU(ELU(Linear(x)))).

    Bloque base del TFT: combina gating (GLU) con residual para estabilidad
    de gradientes y capacidad de seleccion de informacion.

    Args:
        input_dim:  Dimensionalidad de entrada.
        hidden_dim: Dimension oculta intermedia.
        output_dim: Dimension de salida.
        dropout:    Tasa de dropout aplicada post-ELU.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1    = nn.Linear(input_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, output_dim * 2)  # factor-2 para GLU
        self.skip   = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm   = nn.LayerNorm(output_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h        = self.drop(F.elu(self.fc1(x)))
        eta1, eta2 = self.fc2(h).chunk(2, dim=-1)         # GLU split
        gated    = eta1 * torch.sigmoid(eta2)
        return self.norm(self.skip(x) + gated)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: aprende en cada timestep que features son relevantes.

    Aplica un GRN independiente por feature (proyeccion a d_model) y
    pondera con softmax sobre un GRN global que recibe todas las features
    concatenadas.

    Args:
        n_features: Numero de features de entrada.
        d_model:    Dimension del espacio de embedding.
        dropout:    Dropout compartido.
    """
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model
        self.grns       = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        # GRN para calcular los pesos de seleccion (softmax sobre n_features)
        self.selector   = GatedResidualNetwork(n_features, d_model, n_features, dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, T, n_features)
        processed = torch.stack(
            [self.grns[i](x[..., i : i + 1]) for i in range(self.n_features)],
            dim=-2,
        )  # (batch, T, n_features, d_model)

        weights = F.softmax(self.selector(x), dim=-1).unsqueeze(-1)
        # weights: (batch, T, n_features, 1)

        return (processed * weights).sum(dim=-2)  # (batch, T, d_model)


class TFTCore(nn.Module):
    """
    Nucleo del Temporal Fusion Transformer.

    Pipeline:
        VSN -> LSTM -> gated residual -> multi-head attention
             -> gated residual -> output projection.

    Args:
        n_features:    Numero de features tecnicas de entrada.
        d_model:       Dimension del espacio latente.
        n_heads:       Cabezas de atencion multi-head.
        n_lstm_layers: Capas apiladas del encoder LSTM.
        dropout:       Dropout global.
    """
    def __init__(
        self,
        n_features:    int,
        d_model:       int   = 32,
        n_heads:       int   = 2,
        n_lstm_layers: int   = 2,
        dropout:       float = 0.1,
    ) -> None:
        super().__init__()
        self.vsn       = VariableSelectionNetwork(n_features, d_model, dropout)
        self.encoder   = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.gate_lstm = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_attn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)
        self.head      = nn.Linear(d_model, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, T, n_features)
        vsn_out         = self.vsn(x)                              # (B, T, d)
        lstm_out, _     = self.encoder(vsn_out)                    # (B, T, d)
        post_lstm       = self.norm1(vsn_out + self.gate_lstm(lstm_out))
        attn_out, _     = self.attention(post_lstm, post_lstm, post_lstm)
        post_attn       = self.norm2(post_lstm + self.gate_attn(self.drop(attn_out)))
        return self.head(post_attn[:, -1, :]).squeeze(-1)           # (B,)


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("tft_trend")
class TFTStrategy(AbstractStrategy):
    """
    Estrategia TFT para quant_arena.

    Genera señales de posicion (-1 / 0 / 1) sobre un activo unico a partir
    de features OHLCV causales procesados por un Temporal Fusion Transformer
    entrenado con ventana expansiva.

    Ejemplo de uso::

        strat = TFTStrategy(universo=["^GSPC"], min_train_days=252)
        señal = strat.generar_señales(df_ohlcv, fecha_corte)

    Args:
        universo:             Lista de tickers (un elemento para activo unico).
        min_train_days:       Minimo de pares (X, y) para entrenar.
        lookback:             Longitud de la secuencia de entrada (timesteps).
        d_model:              Dimension del espacio latente TFT.
        n_heads:              Cabezas de atencion.
        n_lstm_layers:        Capas LSTM.
        dropout:              Tasa de dropout.
        umbral_señal:         Umbral simetrico sobre retorno predicho.
        max_epochs:           Epocas maximas por sesion de entrenamiento.
        batch_size:           Tamano de mini-batch.
        lr:                   Tasa de aprendizaje Adam.
        retrain_every_n_days: Dias minimos entre re-entrenamientos.
        close_col:            Columna de precio de cierre en datos.
        volume_col:           Columna de volumen.
        device:               'cpu' | 'cuda' | None (auto-detectar).
        seed:                 Semilla de reproducibilidad.
    """

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        lookback:             int   = 42,
        d_model:              int   = 32,
        n_heads:              int   = 2,
        n_lstm_layers:        int   = 2,
        dropout:              float = 0.1,
        umbral_señal:         float = 0.003,
        max_epochs:           int   = 25,
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
        if not _TORCH_OK:
            raise ImportError("PyTorch no instalado. Ejecuta: pip install torch")
        if min_train_days < 60:
            raise ValueError("min_train_days >= 60.")
        if not (0.0 < umbral_señal < 0.5):
            raise ValueError("umbral_señal en (0, 0.5).")

        super().__init__(nombre="tft_trend", universo=universo)

        self._min_train   = min_train_days
        self._lookback    = lookback
        self._d_model     = d_model
        self._n_heads     = n_heads
        self._n_lstm      = n_lstm_layers
        self._dropout     = dropout
        self._umbral      = umbral_señal
        self._max_epochs  = max_epochs
        self._batch_size  = batch_size
        self._lr          = lr
        self._retrain_days = retrain_every_n_days
        self._close_col   = close_col
        self._volume_col  = volume_col
        self._high_col    = high_col
        self._low_col     = low_col
        self._seed        = seed

        self._device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net: Optional[TFTCore] = None
        self._n_features: Optional[int] = None
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    @property
    def descripcion(self) -> str:
        return (
            f"TFT Trend | lookback={self._lookback}d | d_model={self._d_model} "
            f"| heads={self._n_heads} | umbral=+-{self._umbral:.4f}"
        )

    # ------------------------------------------------------------------
    # Ingenieria de features (causal)
    # ------------------------------------------------------------------

    def _build_features(self, datos: pd.DataFrame) -> pd.DataFrame:
        """
        Computa features OHLCV estrictamente causales.

        Todos los calculos usan shift() > 0 (rezago) o ventanas rolling
        unidireccionales. La SMA(200) es el indicador con mayor warm-up,
        generando NaN en las primeras 199 filas.

        Args:
            datos: DataFrame con DatetimeIndex y columnas OHLCV.

        Returns:
            DataFrame de features con el mismo indice que datos.
        """
        close   = datos[self._close_col].astype(float)
        log_ret = np.log(close / close.shift(1))

        f = pd.DataFrame(index=datos.index)
        f["ret_1d"]  = log_ret
        f["ret_5d"]  = np.log(close / close.shift(5))
        f["ret_21d"] = np.log(close / close.shift(21))
        f["vol_21d"] = log_ret.rolling(21, min_periods=21).std()

        # RSI Wilder (EWM causal)
        d = close.diff()
        ag = d.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
        al = (-d).clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
        f["rsi_14"] = (100.0 - 100.0 / (1.0 + ag / al.replace(0.0, np.nan))) / 100.0

        sma50  = close.rolling(50,  min_periods=50).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        f["price_sma50"]  = close / sma50  - 1.0
        f["sma50_sma200"] = sma50  / sma200 - 1.0

        hi21  = close.rolling(21, min_periods=21).max()
        lo21  = close.rolling(21, min_periods=21).min()
        rango = (hi21 - lo21).replace(0.0, np.nan)
        f["pos_range_21"] = (close - lo21) / rango - 0.5

        if self._high_col in datos.columns and self._low_col in datos.columns:
            high = datos[self._high_col].astype(float)
            low  = datos[self._low_col].astype(float)
            tr   = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            f["atr_norm"] = tr.ewm(span=14, adjust=False).mean() / close.replace(0.0, np.nan)

        if self._volume_col in datos.columns:
            vol    = datos[self._volume_col].astype(float)
            vol_ma = vol.rolling(21, min_periods=21).mean()
            f["vol_ratio"] = vol / vol_ma.replace(0.0, np.nan) - 1.0

        return f

    # ------------------------------------------------------------------
    # Construccion de dataset (X, y) para entrenamiento
    # ------------------------------------------------------------------

    def _construir_dataset(
        self, features: pd.DataFrame, close: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye pares (X_seq, y) para entrenamiento y X_pred para inferencia.

        Garantia de causalidad:
            X[i] = features[i-lookback : i]  (informacion en t)
            y[i] = log_ret[i+1]              (retorno del dia siguiente)
            El target del ultimo dia es NaN -> excluido de entrenamiento.
            X_pred = features[-lookback:]    (incluye fecha_corte, sin target futuro)

        Returns:
            X_train: (n_samples, lookback, n_features)
            y_train: (n_samples,)
            X_pred:  (1, lookback, n_features)
        """
        log_ret = np.log(close / close.shift(1))

        # Dataset de entrenamiento (elimina fila con target NaN)
        df = features.copy()
        df["__y__"] = log_ret.shift(-1)
        df_clean = df.dropna()

        feat_np = df_clean.drop(columns="__y__").values.astype(np.float32)
        tgt_np  = df_clean["__y__"].values.astype(np.float32)

        X_train, y_train = [], []
        for i in range(self._lookback, len(feat_np)):
            X_train.append(feat_np[i - self._lookback : i])
            y_train.append(tgt_np[i])

        # Prediccion: ventana que termina en fecha_corte (ultimo dato disponible)
        feat_for_pred = features.dropna()                           # retiene ultima fila
        X_pred_seq    = feat_for_pred.values[-self._lookback :]     # (lookback, F)

        return (
            np.array(X_train, dtype=np.float32),
            np.array(y_train, dtype=np.float32),
            X_pred_seq[np.newaxis, ...].astype(np.float32),        # (1, lookback, F)
        )

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def _fijar_semilla(self) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

    def _entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena TFTCore sobre (X, y) con Adam y early stopping implicito
        por numero fijo de epocas.

        Args:
            X: (n_samples, lookback, n_features)
            y: (n_samples,)
        """
        self._fijar_semilla()
        n_features = X.shape[-1]

        # (Re)crear red si cambia el numero de features o es la primera vez
        if self._net is None or self._n_features != n_features:
            self._net = TFTCore(
                n_features    = n_features,
                d_model       = self._d_model,
                n_heads       = self._n_heads,
                n_lstm_layers = self._n_lstm,
                dropout       = self._dropout,
            ).to(self._device)
            self._n_features = n_features

        X_t = torch.from_numpy(X).to(self._device)
        y_t = torch.from_numpy(y).to(self._device)

        loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=self._batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self._net.train()
        for _ in range(self._max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

    def _predecir(self, X_pred: np.ndarray) -> float:
        """Inferencia sobre un unico batch (1, lookback, n_features)."""
        if self._net is None:
            return 0.0
        self._net.eval()
        with torch.no_grad():
            x_t  = torch.from_numpy(X_pred).to(self._device)
            pred = self._net(x_t)
        return float(pred.cpu().numpy()[0])

    # ------------------------------------------------------------------
    # AbstractStrategy: generar_señales
    # ------------------------------------------------------------------

    def generar_señales(
        self,
        datos: pd.DataFrame,
        fecha_corte: pd.Timestamp,
    ) -> pd.Series:
        """
        Genera la señal de posicion para el periodo siguiente a fecha_corte.

        Reentrenamiento condicional: solo ocurre si han pasado al menos
        retrain_every_n_days dias desde el ultimo entrenamiento.

        Returns:
            pd.Series [ticker -> peso] con valores en {-1.0, 0.0, 1.0}.
        """
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return neutral

        # Warm-up minimo: 200 (SMA200) + lookback + min_train_days
        if len(hist) < self._min_train + self._lookback + 200:
            return neutral

        features = self._build_features(hist)
        close    = hist[self._close_col].astype(float)

        try:
            X_train, y_train, X_pred = self._construir_dataset(features, close)
        except Exception:
            return neutral

        if len(X_train) < self._min_train or X_pred.shape[1] < self._lookback:
            return neutral

        # Reentrenar solo si es necesario
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

        señal = 1.0 if pred_ret > self._umbral else (-1.0 if pred_ret < -self._umbral else 0.0)

        pesos = pd.Series(0.0, index=indice, dtype=float)
        if self._universo:
            pesos.iloc[0] = señal
        return pesos

    # ------------------------------------------------------------------
    # AbstractStrategy: calcular_retornos
    # ------------------------------------------------------------------

    def calcular_retornos(
        self,
        datos: pd.DataFrame,
        pesos_historicos: pd.DataFrame,
    ) -> pd.Series:
        """Aplica señales historicas a retornos del activo."""
        if self._close_col not in datos.columns or not self._universo:
            return pd.Series(dtype=float)
        ticker = self._universo[0]
        if ticker not in pesos_historicos.columns:
            return pd.Series(dtype=float)
        ret_activo       = datos[self._close_col].pct_change()
        pesos            = pesos_historicos[ticker]
        pesos_al, ret_al = pesos.align(ret_activo, join="inner")
        return (pesos_al * ret_al).dropna()


# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "TFTTEST"

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
    print("  TFTStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = TFTStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=42,
        max_epochs=1,
        d_model=8,
        n_heads=2,
        retrain_every_n_days=9999,
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "tft_trend"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='tft_trend' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up (datos insuficientes)
    print("\n[TEST 2] Warm-up (n=300 < umbral efectivo 494)...")
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

    # TEST 4 -- Regimen crash extremo
    print("\n[TEST 4] Regimen crash extremo (mu=-0.002, sigma=0.035)...")
    df_crash = _make_ohlcv(n=700, mu=-0.002, sigma=0.035, seed=99)
    s = strat.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] senal={s[TICKER]} sin crash en regimen extremo")

    # TEST 5 -- Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba senal neutral con datos malformados"
    print("  [OK] neutral retornado sin excepcion")

    # TEST 6 -- Boundary condition
    print("\n[TEST 6] Boundary condition (n=494 = min_train + lookback + 200)...")
    df_boundary = _make_ohlcv(n=494)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary respetado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
