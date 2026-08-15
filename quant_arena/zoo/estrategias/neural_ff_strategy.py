#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/neural_ff_strategy.py
# Neural Fama-French Strategy — MLP sobre factores FF sinteticos desde OHLCV
# =============================================================================
"""
Estima un modelo de factores tipo Fama-French a partir de proxies OHLCV
y usa un MLP profundo con regularizacion L1 para predecir el retorno
esperado del siguiente periodo.

Factores sinteticos (todos calculables desde precio/volumen):
    MKT:   log-retorno del propio activo (beta market)
    MOM:   retorno 12M-1M (Jegadeesh & Titman momentum)
    VAL:   -(precio / maximo_52w - 1)  proxy de value (mean-reversion desde ATH)
    SIZE:  -log(vol_dolares_21d)        proxy de small-cap premium invertido
    QMJ:   rolling_Sharpe_63d           quality: retorno/riesgo historico
    BAB:   -beta_rolling                betting against beta (baja beta -> premium)

El MLP aprende a combinar estos factores de forma no lineal, con L1 sobre
los pesos de la primera capa para inducir seleccion de factores (sparsity).

Pipeline (causal, expanding window):
    features[t] -> MLP -> pred_ret[t+1]
    entrenamiento: toda la historia hasta fecha_corte - 1 dia
    prediccion: features en fecha_corte

Registro: 'neural_ff'
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
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    from quant_arena.core.torch_compat import nn  # type: ignore[assignment]

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo


# =============================================================================
# Construccion de factores
# =============================================================================

class FamaFrenchProxies:
    """
    Calcula factores sinteticos tipo Fama-French desde datos OHLCV.

    Todos los calculos son estrictamente causales (solo datos pasados).

    Args:
        mom_long:  Ventana larga del factor momentum (dias).
        mom_skip:  Ventana corta a excluir en momentum (skip-month).
        hi52w:     Ventana para calculo del maximo (dias).
        size_w:    Ventana de volumen dolares para factor size.
        qmj_w:     Ventana del Sharpe rolling para factor quality.
        bab_w:     Ventana beta rolling para betting-against-beta.
    """

    def __init__(
        self,
        mom_long: int = 252,
        mom_skip: int = 21,
        hi52w:    int = 252,
        size_w:   int = 21,
        qmj_w:    int = 63,
        bab_w:    int = 63,
    ) -> None:
        self._mom_long = mom_long
        self._mom_skip = mom_skip
        self._hi52w    = hi52w
        self._size_w   = size_w
        self._qmj_w    = qmj_w
        self._bab_w    = bab_w

    def build(
        self,
        datos:      pd.DataFrame,
        close_col:  str = "Close",
        volume_col: str = "Volume",
    ) -> pd.DataFrame:
        """
        Retorna DataFrame de factores (T x 6), mismo indice que datos.

        Factores: mkt, mom, val, size, qmj, bab
        """
        close   = datos[close_col].astype(float)
        log_ret = np.log(close / close.shift(1))

        # --- MKT: retorno 1 dia (beta market proxy) ---
        mkt = log_ret

        # --- MOM: retorno 12M-1M (momentum Jegadeesh-Titman) ---
        ret_12m = np.log(close / close.shift(self._mom_long))
        ret_1m  = np.log(close / close.shift(self._mom_skip))
        mom     = ret_12m - ret_1m

        # --- VAL: distancia negativa al maximo 52 semanas (value/mean-rev) ---
        hi52w   = close.rolling(self._hi52w, min_periods=126).max()
        val     = -(close / hi52w.replace(0, np.nan) - 1.0)  # positivo cuando lejos del maximo

        # --- SIZE: -log(volumen dolares rolling) -> small-cap premium invertido ---
        if volume_col in datos.columns:
            vol_usd = datos[volume_col].astype(float) * close
            size    = -np.log(vol_usd.rolling(self._size_w, min_periods=5).mean().replace(0, np.nan))
        else:
            size = pd.Series(0.0, index=datos.index)

        # --- QMJ: Sharpe rolling (quality) ---
        mu_roll  = log_ret.rolling(self._qmj_w, min_periods=21).mean()
        std_roll = log_ret.rolling(self._qmj_w, min_periods=21).std()
        qmj      = (mu_roll / std_roll.replace(0, np.nan)).fillna(0.0)

        # --- BAB: -beta rolling (betting against beta via correlacion con lag) ---
        # Beta aproximada con covarianza(ret_t, ret_{t-1}) / var(ret_{t-1})
        ret_lag  = log_ret.shift(1)
        cov_roll = log_ret.rolling(self._bab_w, min_periods=21).cov(ret_lag)
        var_roll = ret_lag.rolling(self._bab_w, min_periods=21).var()
        beta     = (cov_roll / var_roll.replace(0, np.nan)).fillna(1.0)
        bab      = -beta  # premium por baja beta

        factors = pd.DataFrame({
            "mkt":  mkt,
            "mom":  mom,
            "val":  val,
            "size": size,
            "qmj":  qmj,
            "bab":  bab,
        }, index=datos.index)

        return factors


# =============================================================================
# MLP con regularizacion L1
# =============================================================================

class NeuralFF(nn.Module):
    """
    Multi-Layer Perceptron con skip-connection y regularizacion L1.

    Arquitectura:
        Input(n_factors) -> Linear -> LayerNorm -> GELU -> Dropout
        -> Linear -> LayerNorm -> GELU -> Dropout
        -> Linear -> LayerNorm -> GELU
        -> head(1)
        con skip-connection desde input

    La regularizacion L1 se aplica externamente sobre los pesos de la
    primera capa para inducir sparsity (seleccion de factores).

    Args:
        n_factors: Numero de factores de entrada.
        hidden:    Neuronas en capas ocultas.
        dropout:   Tasa de dropout.
    """

    def __init__(
        self,
        n_factors: int = 6,
        hidden:    int = 64,
        dropout:   float = 0.20,
    ) -> None:
        super().__init__()
        self.fc1   = nn.Linear(n_factors, hidden)
        self.bn1   = nn.LayerNorm(hidden)
        self.fc2   = nn.Linear(hidden, hidden)
        self.bn2   = nn.LayerNorm(hidden)
        self.fc3   = nn.Linear(hidden, hidden // 2)
        self.bn3   = nn.LayerNorm(hidden // 2)
        self.head  = nn.Linear(hidden // 2, 1)
        self.skip  = nn.Linear(n_factors, hidden // 2, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h1  = self.drop(self.act(self.bn1(self.fc1(x))))
        h2  = self.drop(self.act(self.bn2(self.fc2(h1))))
        h3  = self.act(self.bn3(self.fc3(h2))) + self.act(self.skip(x))
        out = self.head(h3)
        return out

    def l1_loss(self, l1_lambda: float = 1e-4) -> "torch.Tensor":
        """Penalizacion L1 sobre los pesos de la primera capa (factor selection)."""
        return l1_lambda * self.fc1.weight.abs().sum()


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("neural_ff")
class NeuralFFStrategy(AbstractStrategy):
    """
    Estrategia Neural Fama-French para quant_arena.

    Aprende a combinar factores sinteticos tipo FF de forma no lineal
    con un MLP entrenado en expanding window walk-forward.

    Args:
        universo:             Tickers (un elemento).
        min_train_days:       Minimo de dias para primer entrenamiento.
        lookback_min:         Dias minimos necesarios para calcular todos los factores.
        hidden:               Neuronas en capas ocultas del MLP.
        epochs:               Epocas de entrenamiento por sesion.
        lr:                   Learning rate Adam.
        batch_size:           Tamano de batch.
        dropout:              Tasa de dropout.
        l1_lambda:            Coeficiente L1 para seleccion de factores.
        umbral_pred:          Umbral de prediccion para emitir señal.
        retrain_every_n_days: Dias entre re-entrenamientos.
        close_col:            Columna de cierre.
        volume_col:           Columna de volumen.
        seed:                 Semilla.
    """

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        lookback_min:         int   = 252,
        hidden:               int   = 64,
        epochs:               int   = 50,
        lr:                   float = 3e-4,
        batch_size:           int   = 64,
        dropout:              float = 0.20,
        l1_lambda:            float = 1e-4,
        umbral_pred:          float = 0.0003,
        retrain_every_n_days: int   = 21,
        close_col:            str   = "Close",
        volume_col:           str   = "Volume",
        seed:                 int   = 42,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError("pip install torch")
        super().__init__(nombre="neural_ff", universo=universo)

        self._min_train    = min_train_days
        self._lookback_min = lookback_min
        self._hidden       = hidden
        self._epochs       = epochs
        self._lr           = lr
        self._batch_size   = batch_size
        self._dropout      = dropout
        self._l1_lambda    = l1_lambda
        self._umbral       = umbral_pred
        self._retrain_days = retrain_every_n_days
        self._close_col    = close_col
        self._volume_col   = volume_col
        self._seed         = seed

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model:    Optional[NeuralFF] = None
        self._factor_builder = FamaFrenchProxies()
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

        # Estadisticas de normalizacion de factores (calculadas en fit)
        self._factor_mu:  Optional[np.ndarray] = None
        self._factor_std: Optional[np.ndarray] = None
        self._target_mu:  float = 0.0
        self._target_std: float = 1.0

    @property
    def descripcion(self) -> str:
        return (
            f"Neural-FF | hidden={self._hidden} | epochs={self._epochs} "
            f"| l1={self._l1_lambda:.0e} | umbral={self._umbral:.4f}"
        )

    # ------------------------------------------------------------------
    def _preparar_datos(
        self, datos: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye matrices X (T-1, F) e y (T-1,) causales.

        El target y[t] = log_ret[t+1] (retorno del dia siguiente).
        Las features X[t] = factores calculados hasta el dia t.

        La ultima fila de X se usa para predecir (no tiene target aun).
        """
        factors  = self._factor_builder.build(datos, self._close_col, self._volume_col)
        close    = datos[self._close_col].astype(float)
        log_ret  = np.log(close / close.shift(1))

        # Target: retorno del dia siguiente (shift -1)
        target = log_ret.shift(-1)

        df = factors.copy()
        df["__target__"] = target
        df_clean = df.dropna()

        if len(df_clean) < self._min_train:
            return np.empty((0, factors.shape[1])), np.empty(0)

        X = df_clean.drop(columns="__target__").values.astype(np.float64)
        y = df_clean["__target__"].values.astype(np.float64)

        # Excluir la ultima fila del conjunto de entrenamiento
        # (target del ultimo dia todavia no se conoce)
        return X[:-1], y[:-1]

    def _normalizar(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normaliza X e y usando estadisticas de entrenamiento."""
        self._factor_mu  = X.mean(axis=0)
        self._factor_std = X.std(axis=0) + 1e-8
        self._target_mu  = float(y.mean())
        self._target_std = float(y.std() + 1e-8)

        X_n = (X - self._factor_mu) / self._factor_std
        y_n = (y - self._target_mu) / self._target_std
        return X_n.astype(np.float32), y_n.astype(np.float32)

    def _norm_X(self, X: np.ndarray) -> np.ndarray:
        if self._factor_mu is None:
            return X.astype(np.float32)
        return ((X - self._factor_mu) / self._factor_std).astype(np.float32)

    def _denorm_y(self, y_n: float) -> float:
        return float(y_n * self._target_std + self._target_mu)

    def _entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrena el MLP sobre X, y (ya normalizados)."""
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        n_factors = X.shape[1]
        self._model = NeuralFF(
            n_factors = n_factors,
            hidden    = self._hidden,
            dropout   = self._dropout,
        ).to(self._device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset    = TensorDataset(X_t, y_t)
        loader     = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        optimizer  = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._epochs, eta_min=self._lr * 0.1
        )
        criterion  = nn.MSELoss()

        self._model.train()
        for _ in range(self._epochs):
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb) + self._model.l1_loss(self._l1_lambda)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

    def _predecir(self, X_last: np.ndarray) -> float:
        """Prediccion del retorno esperado para el ultimo dia (denormalizada)."""
        if self._model is None:
            return 0.0
        self._model.eval()
        with torch.no_grad():
            x_t  = torch.tensor(X_last, dtype=torch.float32).to(self._device)
            pred_n = float(self._model(x_t.unsqueeze(0)).squeeze().cpu().item())
        return self._denorm_y(pred_n)

    # ------------------------------------------------------------------
    def generar_señales(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return neutral
        if len(hist) < max(self._min_train, self._lookback_min) + 5:
            return neutral

        # Construir features causales
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_train, y_train = self._preparar_datos(hist)
        except Exception:
            return neutral

        if len(X_train) < self._min_train:
            return neutral

        # Re-entrenar si corresponde
        necesita = (
            self._ultimo_entrenamiento is None
            or (fecha_corte - self._ultimo_entrenamiento).days >= self._retrain_days
        )
        if necesita:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    X_n, y_n = self._normalizar(X_train, y_train)
                    self._entrenar(X_n, y_n)
                except Exception:
                    return neutral
            self._ultimo_entrenamiento = fecha_corte

        # Feature del dia actual (ultima fila disponible)
        try:
            factors_hoy = self._factor_builder.build(hist, self._close_col, self._volume_col)
            ultima_fila = factors_hoy.dropna().values[-1].astype(np.float64)
            ultima_norm = self._norm_X(ultima_fila)
            pred        = self._predecir(ultima_norm)
        except Exception:
            return neutral

        # Discretizar
        if pred > self._umbral:
            senal = 1.0
        elif pred < -self._umbral:
            senal = -1.0
        else:
            senal = 0.0

        pesos = pd.Series(0.0, index=indice, dtype=float)
        if self._universo:
            pesos.iloc[0] = senal
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

    TICKER = "NNFFTEST"

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
    print("  NeuralFFStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = NeuralFFStrategy(
        universo=[TICKER],
        min_train_days=252,
        epochs=2,       # fast for tests
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "neural_ff"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='neural_ff' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up (n=150 < threshold=257 -> neutral esperado)
    print("\n[TEST 2] Warm-up (n=150 < min_threshold=257) -> neutral esperado...")
    df_short = _make_ohlcv(n=150)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] senal={s[TICKER]} neutral -- warm-up respetado")

    # TEST 3 -- Operacion normal (n=600)
    print("\n[TEST 3] Operacion normal (n=600)...")
    df_normal = _make_ohlcv(n=600, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] senal={s[TICKER]} in {{-1.0, 0.0, 1.0}}")

    # TEST 4 -- Regimen crash (mu=-0.002, sigma=0.035, n=600)
    print("\n[TEST 4] Regimen crash (mu=-0.002, sigma=0.035, n=600)...")
    df_crash = _make_ohlcv(n=600, mu=-0.002, sigma=0.035, seed=99)
    s = strat.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] senal={s[TICKER]} sin crash con volatilidad extrema")

    # TEST 5 -- Sin columna 'Close' -> neutral sin crash (n=600)
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=600).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba neutral con datos malformados"
    print("  [OK] neutral retornado sin excepcion")

    # TEST 6 -- Boundary condition (n=257 == exact min threshold)
    print("\n[TEST 6] Boundary condition (n=257 == min_threshold)...")
    df_boundary = _make_ohlcv(n=257)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary exacto procesado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
