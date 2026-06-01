#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/mamba_ssm_strategy.py
# Mamba SSM Strategy — State Space Models para secuencias largas y baja latencia
# =============================================================================
"""
Implementacion del bloque Mamba (Gu & Dao, 2023) en PyTorch puro (CPU-compatible).

Mecanismo clave — Selective State Space Model (S6):
    dt[t]  = softplus(W_dt @ x[t])          # paso de discretizacion input-dependiente
    B[t]   = W_B @ x[t]                     # matriz de entrada selectiva
    C[t]   = W_C @ x[t]                     # matriz de salida selectiva
    A_bar  = exp(A * dt[t])                 # A discretizada (diagonal, log-space)
    h[t]   = A_bar * h[t-1] + B[t] * x[t]  # estado oculto
    y[t]   = C[t] * h[t]                   # salida

Registro: 'mamba_ssm'
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
# Arquitectura Mamba
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    S6: Selective Scan SSM con parametros B, C, dt input-dependientes.

    El mecanismo selectivo permite al modelo decidir, para cada input,
    cuanta informacion retener en el estado oculto (similar a una LSTM
    pero con complejidad O(L) en tiempo y memoria).

    Args:
        d_inner: Dimension interna del SSM.
        d_state: Dimension del estado oculto h. Tipicamente 16.
        dt_rank: Rango de la proyeccion de dt. Tipicamente d_inner//16.
    """
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 4) -> None:
        super().__init__()
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.dt_rank  = dt_rank

        # Proyecciones selectivas
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A en log-space (diagonal para eficiencia): A < 0 garantiza estabilidad
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))         # (d_inner, d_state)
        self.D     = nn.Parameter(torch.ones(d_inner))  # skip connection escalar

        # Inicializacion de dt_proj siguiendo Mamba paper
        with torch.no_grad():
            self.dt_proj.bias.data = torch.exp(
                torch.rand(d_inner) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
                + torch.log(torch.tensor(0.001))
            ).log()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Scan selectivo sobre la secuencia.

        Args:
            x: (batch, T, d_inner)

        Returns:
            y: (batch, T, d_inner)
        """
        B_sz, T, d = x.shape
        A           = -torch.exp(self.A_log.float())   # (d_inner, d_state) negativo

        # Proyecciones selectivas
        x_dbl       = self.x_proj(x)                   # (B, T, dt_rank + 2*d_state)
        dt_raw, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt: (B, T, d_inner)  B,C: (B, T, d_state)
        dt = F.softplus(self.dt_proj(dt_raw))

        # Discretizacion ZOH: A_bar[t] = exp(A * dt[t])
        # Shape expansion: A (d,N) x dt (B,T,d) -> (B,T,d,N)
        dt_exp   = dt.unsqueeze(-1)                                # (B,T,d,1)
        A_exp    = A.unsqueeze(0).unsqueeze(0)                     # (1,1,d,N)
        A_bar    = torch.exp(dt_exp * A_exp)                       # (B,T,d,N)

        # B_bar = dt * B (simplificacion ZOH para B)
        B_bar    = dt_exp * B.unsqueeze(2)                         # (B,T,d,N)

        # Scan recurrente O(T) — no CUDA kernel, valido para T<=128
        h = torch.zeros(B_sz, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h    = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :].unsqueeze(-1)
            yt   = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)       # (B, d)
            ys.append(yt)

        y = torch.stack(ys, dim=1)                                  # (B,T,d)
        return y + x * self.D                                       # skip D


class MambaBlock(nn.Module):
    """
    Bloque Mamba completo: proyeccion -> gating -> SSM -> salida.

    Arquitectura:
        x -> [proj_in -> split -> (z, x')] donde
          x' -> conv1d (4 frames) -> silu -> SSM(x') -> * sigmoid(z) -> proj_out

    Args:
        d_model: Dimension de embedding.
        d_inner: Dimension interna expandida (tipicamente 2*d_model).
        d_state: Dimension del estado SSM.
        dt_rank: Rango del paso temporal.
        dropout: Dropout post-bloque.
    """
    def __init__(
        self,
        d_model: int,
        d_inner: int   = 0,
        d_state: int   = 16,
        dt_rank: int   = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_inner     = d_inner or d_model * 2
        self.proj_in  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.ssm      = SelectiveSSM(d_inner, d_state, dt_rank)
        self.proj_out = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, d_model)
        skip = x
        xz         = self.proj_in(x)                       # (B,T,2*d_inner)
        x_in, z    = xz.chunk(2, dim=-1)                   # (B,T,d_inner) each

        # Conv1d causal (padding elimina el look-ahead)
        x_conv     = self.conv1d(x_in.transpose(1, 2))[:, :, :x_in.size(1)].transpose(1, 2)
        x_conv     = F.silu(x_conv)

        y          = self.ssm(x_conv) * F.silu(z)          # gate multiplicativo
        out        = self.proj_out(y)
        return self.norm(skip + self.drop(out))


class MambaPredictor(nn.Module):
    """
    Stack de N MambaBlocks con head de regresion escalar.

    Args:
        n_features: Features de entrada.
        d_model:    Dimension de embedding.
        n_layers:   Bloques Mamba apilados.
        d_state:    Dimension estado SSM.
        dropout:    Dropout.
    """
    def __init__(
        self,
        n_features: int,
        d_model:    int   = 32,
        n_layers:   int   = 2,
        d_state:    int   = 16,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.embed  = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head   = nn.Linear(d_model, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1, :]).squeeze(-1)          # (B,)


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("mamba_ssm")
class MambaStrategy(AbstractStrategy):
    """
    Estrategia Mamba SSM para quant_arena.

    Captura dependencias de largo plazo en series temporales de features OHLCV
    mediante el mecanismo selectivo del bloque Mamba, con complejidad lineal
    en la longitud de la secuencia.

    Args:
        universo:             Tickers (un elemento para activo unico).
        min_train_days:       Minimo de muestras de entrenamiento.
        lookback:             Longitud de la secuencia de entrada.
        d_model:              Dimension de embedding Mamba.
        n_layers:             Bloques Mamba apilados.
        d_state:              Dimension del estado SSM.
        umbral_señal:         Umbral simetrico para discretizacion.
        max_epochs:           Epocas maximas de entrenamiento.
        batch_size:           Tamano de mini-batch.
        lr:                   Learning rate Adam.
        retrain_every_n_days: Dias entre re-entrenamientos.
        close_col:            Columna de cierre en datos.
        volume_col:           Columna de volumen.
        device:               Dispositivo de computo.
        seed:                 Semilla reproducibilidad.
    """

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        lookback:             int   = 63,
        d_model:              int   = 32,
        n_layers:             int   = 2,
        d_state:              int   = 16,
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
        if not _TORCH_OK:
            raise ImportError("pip install torch")
        if min_train_days < 60:
            raise ValueError("min_train_days >= 60.")

        super().__init__(nombre="mamba_ssm", universo=universo)

        self._min_train    = min_train_days
        self._lookback     = lookback
        self._d_model      = d_model
        self._n_layers     = n_layers
        self._d_state      = d_state
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

        self._net: Optional[MambaPredictor] = None
        self._n_features: Optional[int]     = None
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

    @property
    def descripcion(self) -> str:
        return (
            f"Mamba SSM | lookback={self._lookback}d | d_model={self._d_model} "
            f"| layers={self._n_layers} | d_state={self._d_state} | umbral=+-{self._umbral:.4f}"
        )

    # ------------------------------------------------------------------
    def _build_features(self, datos: pd.DataFrame) -> pd.DataFrame:
        close   = datos[self._close_col].astype(float)
        log_ret = np.log(close / close.shift(1))
        f       = pd.DataFrame(index=datos.index)

        f["ret_1d"]  = log_ret
        f["ret_5d"]  = np.log(close / close.shift(5))
        f["ret_21d"] = np.log(close / close.shift(21))
        f["vol_21d"] = log_ret.rolling(21, min_periods=21).std()

        d  = close.diff()
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
            h, l = datos[self._high_col].astype(float), datos[self._low_col].astype(float)
            tr   = pd.concat([h - l, (h - close.shift(1)).abs(), (l - close.shift(1)).abs()], axis=1).max(axis=1)
            f["atr_norm"] = tr.ewm(span=14, adjust=False).mean() / close.replace(0.0, np.nan)

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
            self._net = MambaPredictor(
                n_features = n_features,
                d_model    = self._d_model,
                n_layers   = self._n_layers,
                d_state    = self._d_state,
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

        features = self._build_features(hist)
        close    = hist[self._close_col].astype(float)

        try:
            X_train, y_train, X_pred = self._construir_dataset(features, close)
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

    TICKER = "MBTEST"

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
    print("  MambaStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = MambaStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=63,
        max_epochs=1,
        d_model=8,
        n_layers=1,
        d_state=4,
        retrain_every_n_days=9999,
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "mamba_ssm"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='mamba_ssm' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up
    print("\n[TEST 2] Warm-up (n=300 < umbral efectivo 515)...")
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

    # TEST 4 -- Crash extremo
    print("\n[TEST 4] Regimen crash extremo (mu=-0.002, sigma=0.035)...")
    df_crash = _make_ohlcv(n=700, mu=-0.002, sigma=0.035, seed=99)
    s = strat.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] senal={s[TICKER]} sin crash en regimen extremo")

    # TEST 5 -- Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "Cierre"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba senal neutral con datos malformados"
    print("  [OK] neutral retornado sin excepcion")

    # TEST 6 -- Boundary condition
    _THRESHOLD = strat._min_train + strat._lookback + 200  # = 515
    print(f"\n[TEST 6] Boundary condition (n={_THRESHOLD} = min_train + lookback + 200)...")
    df_boundary = _make_ohlcv(_THRESHOLD, seed=7)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary respetado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
