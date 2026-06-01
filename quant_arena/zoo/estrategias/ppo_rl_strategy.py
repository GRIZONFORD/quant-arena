#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/ppo_rl_strategy.py
# PPO RL Strategy — Asignacion end-to-end por Aprendizaje por Refuerzo
# =============================================================================
"""
Estrategia de trading basada en Proximal Policy Optimization (PPO) con un
entorno Gymnasium custom que simula el mercado de forma episodica.

Entorno (TradingEnv):
    Observacion: features tecnicas OHLCV + posicion actual
    Accion:      Discrete(3) -> {0: corto -1, 1: neutral 0, 2: largo +1}
    Reward:      log_ret[t+1] * accion[t]  (recompensa de trading directa)

Pipeline:
    1. Construir TradingEnv con datos hasta fecha_corte.
    2. Entrenar PPO durante max_steps total_timesteps.
    3. Predecir la accion en el ultimo estado del episodio.

Registro: 'ppo_rl'
"""
from __future__ import annotations

import sys, warnings, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:
    _GYM_OK = False

try:
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.env_util import make_vec_env
    _SB3_OK = True
except ImportError:
    _SB3_OK = False

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo

logger = logging.getLogger(__name__)


# =============================================================================
# Entorno Gymnasium
# =============================================================================

class TradingEnv(gym.Env):
    """
    Entorno de trading episodico para RL.

    Cada episodio recorre el historial de precios de inicio a fin. En cada
    step el agente observa el estado actual y elige una accion de posicion.

    Attributes:
        observation_space: Box(n_obs,) — features normalizados + posicion.
        action_space:      Discrete(3) — {0: corto, 1: neutral, 2: largo}.

    Args:
        features_arr: (T, F) array de features normalizados sin NaN.
        log_rets_arr: (T,) array de log-retornos alineados con features.
        window_size:  Numero de pasos de contexto en la observacion.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_arr: np.ndarray,
        log_rets_arr: np.ndarray,
        window_size:  int = 10,
    ) -> None:
        super().__init__()
        assert len(features_arr) == len(log_rets_arr), "features y log_rets deben tener el mismo largo."

        self._features   = features_arr.astype(np.float32)
        self._log_rets   = log_rets_arr.astype(np.float32)
        self._T          = len(features_arr)
        self._window     = window_size
        self._n_feat     = features_arr.shape[1]

        # Obs: ultimos window_size * n_features + posicion actual
        obs_dim = self._window * self._n_feat + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Acciones: 0=corto(-1), 1=neutral(0), 2=largo(+1)
        self.action_space = spaces.Discrete(3)
        self._action_map  = {0: -1.0, 1: 0.0, 2: 1.0}

        self._current_step: int  = 0
        self._position:     float = 0.0

    def _obs(self) -> np.ndarray:
        """
        Construye el vector de observacion actual.

        Incluye los ultimos window_size pasos de features (aplanados)
        mas la posicion actual, para dar contexto al agente sobre el
        regimen actual y su estado de inventario.
        """
        start = max(0, self._current_step - self._window)
        window_feats = self._features[start : self._current_step]  # (k, F) k <= window

        # Padding a la izquierda si k < window (inicio del episodio)
        if len(window_feats) < self._window:
            pad = np.zeros((self._window - len(window_feats), self._n_feat), dtype=np.float32)
            window_feats = np.concatenate([pad, window_feats], axis=0)

        return np.concatenate([window_feats.flatten(), [self._position]], axis=0)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._current_step = self._window
        self._position     = 0.0
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecuta un step de trading.

        La recompensa es: log_ret[t+1] * accion[t], normalizada para
        estabilidad numerica del gradiente PPO (clip a [-1, 1]).

        Args:
            action: int en {0, 1, 2}

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._position = self._action_map[int(action)]

        if self._current_step >= self._T - 1:
            return self._obs(), 0.0, True, False, {}

        # Reward: retorno del activo escalado por la posicion tomada
        log_ret = float(self._log_rets[self._current_step])
        reward  = float(np.clip(self._position * log_ret * 100.0, -1.0, 1.0))

        self._current_step += 1
        terminated = self._current_step >= self._T - 1
        return self._obs(), reward, terminated, False, {}

    def render(self) -> None:
        pass


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("ppo_rl")
class PPOStrategy(AbstractStrategy):
    """
    Estrategia PPO-RL para quant_arena.

    Entrena un agente PPO (stable-baselines3) sobre un entorno de trading
    custom y utiliza su politica aprendida para generar señales de posicion.

    Args:
        universo:             Tickers (un elemento para activo unico).
        min_train_days:       Minimo de dias historicos para entrenar el agente.
        total_timesteps:      Pasos de entrenamiento PPO por sesion.
        window_size:          Pasos de contexto en la observacion.
        umbral_accion:        Umbral de confianza para emitir señal no-neutral.
                              Si la accion mas probable supera umbral -> emite señal.
        retrain_every_n_days: Dias entre re-entrenamientos del agente.
        close_col:            Columna de cierre en datos.
        volume_col:           Columna de volumen.
        high_col:             Columna High.
        low_col:              Columna Low.
        seed:                 Semilla de reproducibilidad.
        ppo_kwargs:           Override de hiperparametros de SB3 PPO.
    """

    _PPO_DEFAULTS: Dict = {
        "learning_rate":    3e-4,
        "n_steps":          512,
        "batch_size":       64,
        "n_epochs":         5,
        "gamma":            0.99,
        "gae_lambda":       0.95,
        "clip_range":       0.2,
        "ent_coef":         0.01,
        "vf_coef":          0.5,
        "max_grad_norm":    0.5,
        "policy_kwargs":    {"net_arch": [64, 64]},
        "verbose":          0,
    }

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        total_timesteps:      int   = 10_000,
        window_size:          int   = 10,
        umbral_accion:        float = 0.5,
        retrain_every_n_days: int   = 63,
        close_col:            str   = "Close",
        volume_col:           str   = "Volume",
        high_col:             str   = "High",
        low_col:              str   = "Low",
        seed:                 int   = 42,
        ppo_kwargs:           Optional[Dict] = None,
    ) -> None:
        if not _GYM_OK:
            raise ImportError("pip install gymnasium")
        if not _SB3_OK:
            raise ImportError("pip install stable-baselines3")

        super().__init__(nombre="ppo_rl", universo=universo)

        self._min_train       = min_train_days
        self._total_timesteps = total_timesteps
        self._window_size     = window_size
        self._umbral          = umbral_accion
        self._retrain_days    = retrain_every_n_days
        self._close_col       = close_col
        self._volume_col      = volume_col
        self._high_col        = high_col
        self._low_col         = low_col
        self._seed            = seed
        self._ppo_kwargs      = {**self._PPO_DEFAULTS, **(ppo_kwargs or {})}

        self._agente: Optional[SB3_PPO]         = None
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

    @property
    def descripcion(self) -> str:
        return (
            f"PPO-RL | timesteps={self._total_timesteps:,} "
            f"| window={self._window_size} | retrain={self._retrain_days}d"
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

        sma50  = close.rolling(50, min_periods=50).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        f["price_sma50"]  = close / sma50  - 1.0
        f["sma50_sma200"] = sma50 / sma200  - 1.0

        if self._high_col in datos.columns and self._low_col in datos.columns:
            h, l = datos[self._high_col].astype(float), datos[self._low_col].astype(float)
            tr   = pd.concat([h - l, (h-close.shift(1)).abs(), (l-close.shift(1)).abs()], axis=1).max(axis=1)
            f["atr_norm"] = tr.ewm(span=14, adjust=False).mean() / close.replace(0.0, np.nan)

        if self._volume_col in datos.columns:
            vol    = datos[self._volume_col].astype(float)
            vol_ma = vol.rolling(21, min_periods=21).mean()
            f["vol_ratio"] = vol / vol_ma.replace(0.0, np.nan) - 1.0

        return f

    def _preparar_arrays(
        self, features: pd.DataFrame, close: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alinea features y log-retornos eliminando NaN.

        El log-retorno en t se alinea con la observacion en t, de modo que
        el agente recibe el retorno resultante de su posicion en t.

        Returns:
            feat_arr:  (T, F) float32
            log_rets:  (T,)   float32
        """
        log_ret = np.log(close / close.shift(1))
        df      = features.copy()
        df["__lr__"] = log_ret
        df_clean     = df.dropna()

        feat_arr  = df_clean.drop(columns="__lr__").values.astype(np.float32)
        lr_arr    = df_clean["__lr__"].values.astype(np.float32)

        # Normalizar features para estabilidad PPO (clip +-5 sigma)
        mu    = feat_arr.mean(axis=0, keepdims=True)
        sigma = feat_arr.std(axis=0, keepdims=True) + 1e-8
        feat_arr = np.clip((feat_arr - mu) / sigma, -5.0, 5.0)

        return feat_arr.astype(np.float32), lr_arr

    def _entrenar(self, feat_arr: np.ndarray, lr_arr: np.ndarray) -> None:
        """Entrena el agente PPO sobre el historial completo."""
        np.random.seed(self._seed)

        env = TradingEnv(feat_arr, lr_arr, window_size=self._window_size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._agente = SB3_PPO(
                "MlpPolicy",
                env,
                seed = self._seed,
                **self._ppo_kwargs,
            )
            self._agente.learn(total_timesteps=self._total_timesteps, progress_bar=False)

    def _predecir_accion(self, feat_arr: np.ndarray, lr_arr: np.ndarray) -> float:
        """
        Genera la señal usando la politica aprendida sobre el estado actual.

        Resetea el entorno al inicio, avanza hasta el ultimo paso y toma la
        accion que dicta la politica.
        """
        if self._agente is None:
            return 0.0

        env = TradingEnv(feat_arr, lr_arr, window_size=self._window_size)
        obs, _ = env.reset()

        # Avanzar hasta el ultimo estado con la politica aprendida
        done = False
        last_action = 1  # neutral por defecto
        while not done:
            action, _ = self._agente.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            last_action = int(action)

        action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        return action_map.get(last_action, 0.0)

    # ------------------------------------------------------------------
    def generar_señales(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return neutral
        if len(hist) < self._min_train + 200:
            return neutral

        features = self._build_features(hist)
        close    = hist[self._close_col].astype(float)

        try:
            feat_arr, lr_arr = self._preparar_arrays(features, close)
        except Exception:
            return neutral

        if len(feat_arr) < self._min_train:
            return neutral

        necesita = (
            self._ultimo_entrenamiento is None
            or (fecha_corte - self._ultimo_entrenamiento).days >= self._retrain_days
        )
        if necesita:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suprimir logs de SB3
                logging.getLogger("stable_baselines3").setLevel(logging.ERROR)
                self._entrenar(feat_arr, lr_arr)
            self._ultimo_entrenamiento = fecha_corte

        señal = self._predecir_accion(feat_arr, lr_arr)

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

    TICKER = "PPOTEST"

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
    print("  PPOStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = PPOStrategy(
        universo=[TICKER],
        min_train_days=252,
        total_timesteps=500,
    )

    # ---------------------------------------------------------------
    # TEST 1 -- Propiedades basicas
    # ---------------------------------------------------------------
    print("\n  [T1] Propiedades basicas ...", end=" ")
    assert strat.nombre == "ppo_rl",         "[T1] nombre incorrecto"
    assert TICKER in strat._universo,        "[T1] ticker no en universo"
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0, \
        "[T1] descripcion vacia"
    print("OK")

    # ---------------------------------------------------------------
    # TEST 2 -- Warm-up (n=200 -> neutral)
    # ---------------------------------------------------------------
    print("  [T2] Warm-up n=200 -> neutral ...", end=" ")
    df200     = _make_ohlcv(200)
    fc200     = df200.index[-1]
    senal200  = strat.generar_señales(df200, fc200)
    _check(senal200, "T2", TICKER)
    assert senal200[TICKER] == 0.0, f"[T2] Esperaba 0.0, obtuvo {senal200[TICKER]}"
    print("OK")

    # ---------------------------------------------------------------
    # TEST 3 -- Operacion normal (n=600)
    # ---------------------------------------------------------------
    print("  [T3] Operacion normal n=600 ...", end=" ")
    df600   = _make_ohlcv(600)
    fc600   = df600.index[-1]
    senal3  = strat.generar_señales(df600, fc600)
    _check(senal3, "T3", TICKER)
    print(f"OK  (senal={senal3[TICKER]})")

    # ---------------------------------------------------------------
    # TEST 4 -- Regimen crash (mu=-0.002, sigma=0.035, n=600, seed=99)
    # ---------------------------------------------------------------
    print("  [T4] Regimen crash n=600 ...", end=" ")
    df_crash  = _make_ohlcv(600, mu=-0.002, sigma=0.035, seed=99)
    fc_crash  = df_crash.index[-1]
    senal4    = strat.generar_señales(df_crash, fc_crash)
    _check(senal4, "T4", TICKER)
    print(f"OK  (senal={senal4[TICKER]})")

    # ---------------------------------------------------------------
    # TEST 5 -- Sin columna 'Close' -> neutral sin crash
    # ---------------------------------------------------------------
    print("  [T5] Sin columna 'Close' -> neutral ...", end=" ")
    df_noclose = _make_ohlcv(600).drop(columns=["Close"])
    fc_noclose = df_noclose.index[-1]
    s          = strat.generar_señales(df_noclose, fc_noclose)
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba senal neutral con datos malformados"
    print("OK")

    # ---------------------------------------------------------------
    # TEST 6 -- Boundary condition (n = 452 = min_train_days + 200)
    # ---------------------------------------------------------------
    print("  [T6] Boundary n=452 (umbral exacto) ...", end=" ")
    # threshold: len(hist) < min_train_days + 200  =>  452 < 452 is False
    # so n=452 should pass the first guard and attempt training.
    df452  = _make_ohlcv(452)
    fc452  = df452.index[-1]
    senal6 = strat.generar_señales(df452, fc452)
    _check(senal6, "T6", TICKER)
    print(f"OK  (senal={senal6[TICKER]})")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
