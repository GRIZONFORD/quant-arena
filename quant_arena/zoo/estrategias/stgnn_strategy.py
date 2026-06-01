#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/stgnn_strategy.py
# ST-GNN Strategy — Alfa de correlacion cruzada en grafo de canales features
# =============================================================================
"""
Spatio-Temporal Graph Neural Network aplicado a un grafo de grupos de features
derivados de datos OHLCV de un activo unico.

Cada nodo del grafo representa un grupo de features economico:
    Nodo 0 — Momentum:      ret_1d, ret_5d, ret_21d
    Nodo 1 — Volatilidad:   vol_21d, atr_norm
    Nodo 2 — Tendencia:     price_sma50, sma50_sma200
    Nodo 3 — Mean-Rev.:     pos_range_21, rsi_14
    Nodo 4 — Volumen:       vol_ratio

Las aristas son las correlaciones rolling entre grupos (grafo adaptativo).

Pipeline:
    x(B,T,N,F_node) -> GCN -> GRU sobre T -> head -> señal

Registro: 'stgnn_alpha'
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
# Bloques del grafo
# =============================================================================

class GraphConvLayer(nn.Module):
    """
    Capa GCN espectral: H' = sigma(D^{-1/2} A D^{-1/2} H W).

    Acepta la matriz de adyacencia normalizada precalculada para evitar
    recalcularla en el forward pass cuando A es fija por periodo de reentrenamiento.

    Args:
        in_features:  Features de entrada por nodo.
        out_features: Features de salida por nodo.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.W    = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self, H: "torch.Tensor", A_norm: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Args:
            H:      (batch, N, in_features) — representacion de nodos.
            A_norm: (N, N)  — adyacencia normalizada D^{-1/2} A D^{-1/2}.
        Returns:
            H': (batch, N, out_features)
        """
        # Proyeccion lineal
        HW = self.W(H)                           # (B, N, out)
        # Propagacion espectral: A_norm @ HW
        out = torch.einsum("nm,bmd->bnd", A_norm, HW) + self.bias
        return F.relu(out)


class STGNNCore(nn.Module):
    """
    Pipeline espatiotemporal:
        1. GCN en cada timestep: agrega informacion entre nodos.
        2. GRU sobre la dimension temporal: captura dependencias temporales.
        3. Head escalar sobre el ultimo estado del GRU.

    Args:
        n_nodes:      Numero de nodos en el grafo de features.
        node_feat_in: Features por nodo de entrada.
        d_gcn:        Dimension de salida de la capa GCN.
        d_gru:        Dimension del estado oculto GRU.
        n_gru_layers: Capas GRU apiladas.
        dropout:      Dropout.
    """
    def __init__(
        self,
        n_nodes:      int,
        node_feat_in: int,
        d_gcn:        int   = 16,
        d_gru:        int   = 32,
        n_gru_layers: int   = 2,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.gcn1    = GraphConvLayer(node_feat_in, d_gcn)
        self.gcn2    = GraphConvLayer(d_gcn, d_gcn)
        self.gru     = nn.GRU(
            n_nodes * d_gcn, d_gru, n_gru_layers,
            batch_first=True, dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(d_gru)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(d_gru, 1)

    def forward(self, x: "torch.Tensor", A_norm: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x:      (batch, T, N, F_node)
            A_norm: (N, N)
        Returns:
            output: (batch,)
        """
        B, T, N, F = x.shape

        # Aplicar GCN en cada timestep (reshape para procesar en batch)
        x_flat = x.view(B * T, N, F)                        # (B*T, N, F)
        h      = self.gcn1(x_flat, A_norm)                   # (B*T, N, d_gcn)
        h      = self.gcn2(h, A_norm)                        # (B*T, N, d_gcn)
        h      = h.reshape(B, T, N * h.shape[-1])               # (B, T, N*d_gcn)

        # GRU temporal
        gru_out, _ = self.gru(h)                             # (B, T, d_gru)
        last       = self.norm(self.drop(gru_out[:, -1, :])) # (B, d_gru)
        return self.head(last).squeeze(-1)                    # (B,)


# =============================================================================
# Grupos de features y grafo adaptativo
# =============================================================================

# Definicion de grupos de features: cada grupo es un nodo del grafo
_GRUPOS: List[List[str]] = [
    ["ret_1d", "ret_5d", "ret_21d"],     # Nodo 0: Momentum
    ["vol_21d", "atr_norm"],             # Nodo 1: Volatilidad
    ["price_sma50", "sma50_sma200"],     # Nodo 2: Tendencia
    ["pos_range_21", "rsi_14"],          # Nodo 3: Mean-Reversion
    ["vol_ratio"],                        # Nodo 4: Volumen
]
_N_NODES = len(_GRUPOS)
_MAX_NODE_FEATS = max(len(g) for g in _GRUPOS)   # padding a esta dimension


def _build_adjacency(feature_df: pd.DataFrame, grupos: List[List[str]]) -> np.ndarray:
    """
    Construye la matriz de adyacencia normalizada D^{-1/2}(A+I)D^{-1/2}.

    Los pesos de aristas son las correlaciones absolutas de Pearson entre
    los promedios de cada grupo de features (sobre la ventana disponible).
    Se agrega auto-conexion (+I) para estabilidad numerica.

    Args:
        feature_df: DataFrame con features ya limpias (sin NaN).
        grupos:     Lista de grupos de features por nodo.

    Returns:
        A_norm: (N, N) como ndarray float32.
    """
    N = len(grupos)
    # Media de cada grupo por dia -> representacion del nodo
    nodo_series = []
    for g in grupos:
        cols_ok = [c for c in g if c in feature_df.columns]
        if cols_ok:
            nodo_series.append(feature_df[cols_ok].mean(axis=1).values)
        else:
            nodo_series.append(np.zeros(len(feature_df)))

    # Correlacion entre nodos
    data_mat = np.stack(nodo_series, axis=0)           # (N, T)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(data_mat)                   # (N, N)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)       # correlacion absoluta

    # A = corr + I (auto-conexion)
    A = corr + np.eye(N, dtype=np.float32)

    # Normalizacion simetrica D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)


def _features_a_tensor_nodo(
    feature_df: pd.DataFrame,
    grupos: List[List[str]],
    max_feat: int,
) -> np.ndarray:
    """
    Convierte el DataFrame de features en un tensor de nodos.

    Returns:
        tensor: (T, N, max_feat) con padding de ceros para nodos con menos features.
    """
    T = len(feature_df)
    arr = np.zeros((T, len(grupos), max_feat), dtype=np.float32)
    for n, g in enumerate(grupos):
        cols_ok = [c for c in g if c in feature_df.columns]
        for fi, col in enumerate(cols_ok[:max_feat]):
            arr[:, n, fi] = feature_df[col].values.astype(np.float32)
    return arr


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("stgnn_alpha")
class STGNNStrategy(AbstractStrategy):
    """
    Estrategia ST-GNN para quant_arena.

    Modela las dependencias cruzadas entre grupos de features economicos
    como un grafo de correlaciones, procesando su evolucion temporal
    con un GRU sobre las representaciones GCN en cada timestep.

    Args:
        universo:             Tickers (un elemento para activo unico).
        min_train_days:       Minimo de muestras de entrenamiento.
        lookback:             Pasos temporales de la secuencia de entrada.
        d_gcn:                Dimension GCN por nodo.
        d_gru:                Dimension estado GRU.
        n_gru_layers:         Capas GRU.
        umbral_señal:         Umbral de discretizacion.
        max_epochs:           Epocas de entrenamiento.
        batch_size:           Tamano de mini-batch.
        lr:                   Learning rate Adam.
        retrain_every_n_days: Dias entre re-entrenamientos.
        close_col:            Columna de cierre.
        device:               Dispositivo.
        seed:                 Semilla.
    """

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        lookback:             int   = 42,
        d_gcn:                int   = 16,
        d_gru:                int   = 32,
        n_gru_layers:         int   = 2,
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
        super().__init__(nombre="stgnn_alpha", universo=universo)

        self._min_train    = min_train_days
        self._lookback     = lookback
        self._d_gcn        = d_gcn
        self._d_gru        = d_gru
        self._n_gru_layers = n_gru_layers
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

        self._net: Optional[STGNNCore]          = None
        self._A_norm_cache: Optional[np.ndarray] = None
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

    @property
    def descripcion(self) -> str:
        return (
            f"ST-GNN | nodes={_N_NODES} | lookback={self._lookback}d "
            f"| d_gcn={self._d_gcn} | d_gru={self._d_gru} | umbral=+-{self._umbral:.4f}"
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
            tr   = pd.concat([h - l, (h-close.shift(1)).abs(), (l-close.shift(1)).abs()], axis=1).max(axis=1)
            f["atr_norm"] = tr.ewm(span=14, adjust=False).mean() / close.replace(0.0, np.nan)

        if self._volume_col in datos.columns:
            vol    = datos[self._volume_col].astype(float)
            vol_ma = vol.rolling(21, min_periods=21).mean()
            f["vol_ratio"] = vol / vol_ma.replace(0.0, np.nan) - 1.0

        return f

    def _construir_dataset_grafo(
        self, features: pd.DataFrame, close: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye X en formato grafo (B, lookback, N, F_node) y y escalar.

        Returns:
            X_train: (n_samples, lookback, N, F_node)
            y_train: (n_samples,)
            X_pred:  (1, lookback, N, F_node)
            A_norm:  (N, N)
        """
        log_ret = np.log(close / close.shift(1))
        df      = features.copy()
        df["__y__"] = log_ret.shift(-1)
        df_clean    = df.dropna()

        feat_df = df_clean.drop(columns="__y__")
        tgt_np  = df_clean["__y__"].values.astype(np.float32)

        A_norm = _build_adjacency(feat_df, _GRUPOS)

        # Tensor de nodos: (T, N, F_node)
        node_tensor = _features_a_tensor_nodo(feat_df, _GRUPOS, _MAX_NODE_FEATS)

        X_train, y_train = [], []
        for i in range(self._lookback, len(node_tensor)):
            X_train.append(node_tensor[i - self._lookback : i])
            y_train.append(tgt_np[i])

        # Prediccion: ultima ventana de features_limpias (incluye fecha_corte)
        feat_pred  = features.dropna()
        nt_pred    = _features_a_tensor_nodo(feat_pred, _GRUPOS, _MAX_NODE_FEATS)
        X_pred_seq = nt_pred[-self._lookback :]                    # (lookback, N, F)

        return (
            np.array(X_train, dtype=np.float32),
            np.array(y_train, dtype=np.float32),
            X_pred_seq[np.newaxis, ...],
            A_norm,
        )

    def _entrenar(self, X: np.ndarray, y: np.ndarray, A_norm: np.ndarray) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        node_feat_in = X.shape[-1]
        if self._net is None:
            self._net = STGNNCore(
                n_nodes      = _N_NODES,
                node_feat_in = node_feat_in,
                d_gcn        = self._d_gcn,
                d_gru        = self._d_gru,
                n_gru_layers = self._n_gru_layers,
            ).to(self._device)

        A_t      = torch.from_numpy(A_norm).to(self._device)
        X_t      = torch.from_numpy(X).to(self._device)
        y_t      = torch.from_numpy(y).to(self._device)
        loader   = DataLoader(TensorDataset(X_t, y_t), batch_size=self._batch_size, shuffle=True)
        opt      = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        crit     = nn.MSELoss()

        self._A_norm_cache = A_norm
        self._net.train()
        for _ in range(self._max_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self._net(xb, A_t)
                loss = crit(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()

    def _predecir(self, X_pred: np.ndarray) -> float:
        if self._net is None or self._A_norm_cache is None:
            return 0.0
        A_t = torch.from_numpy(self._A_norm_cache).to(self._device)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(torch.from_numpy(X_pred).to(self._device), A_t)
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
            X_train, y_train, X_pred, A_norm = self._construir_dataset_grafo(features, close)
        except Exception:
            return neutral

        if len(X_train) < self._min_train:
            return neutral

        necesita = (
            self._ultimo_entrenamiento is None
            or (fecha_corte - self._ultimo_entrenamiento).days >= self._retrain_days
        )
        if necesita:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._entrenar(X_train, y_train, A_norm)
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

    TICKER = "STGNNTEST"

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
    print("  STGNNStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = STGNNStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=42,
        max_epochs=1,
        d_gcn=4,
        d_gru=8,
        n_gru_layers=1,
        retrain_every_n_days=9999,
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "stgnn_alpha"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='stgnn_alpha' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up
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
    _THRESHOLD = strat._min_train + strat._lookback + 200  # = 494
    print(f"\n[TEST 6] Boundary condition (n={_THRESHOLD} = min_train + lookback + 200)...")
    df_boundary = _make_ohlcv(_THRESHOLD, seed=7)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary respetado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
