#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/olps_rmr_strategy.py
# OLPS-RMR Strategy — Robust Median Reversion (Huang et al., 2013)
# =============================================================================
"""
Implementacion de Robust Median Reversion (RMR) del paper:
    Huang, D., Zhou, J., Li, B., Hoi, S. C., & Zhou, S. (2013).
    Robust Median Reversion Strategy for Online Portfolio Selection.
    IJCAI, 2135-2141.

El algoritmo estima el precio "justo" del activo como la mediana de los
ultimos L precios ajustados por el retorno de la ventana, y genera una
señal larga cuando el precio actual esta por debajo de esa estimacion
(el activo esta "barato" segun la hipotesis de reversion a la mediana).

Pipeline:
    1. Ventana de precios: P = [close_{t-L+1}, ..., close_t]
    2. Prediccion RMR: p_hat = median(P) / precio_actual  (precio relativo esperado)
    3. Exceso de rendimiento esperado: e_t = p_hat - 1
    4. Señal:
        e_t > +eps  -> largo  (+1)
        e_t < -eps  -> corto  (-1)
        |e_t| <= eps -> neutral (0)

Variante on-line (actualizable sin reentrenamiento):
    - No requiere ningun modelo ML
    - O(L) memoria, O(1) prediccion por paso
    - Puro NumPy: sin dependencias de ML/DL

Para que la señal sea robusta ante outliers, se usa la mediana de Huber
(mediana simple) en vez de la media aritmetica.

Registro: 'olps_rmr'
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo


# =============================================================================
# Utilidades RMR
# =============================================================================

def _rmr_predict(prices: np.ndarray) -> float:
    """
    Prediccion de precio relativo usando mediana robusta.

    Dados L precios historicos, estima el precio normalizado esperado
    como la mediana de los ratios precio_i / precio_actual.

    Args:
        prices: array (L,) de precios historicos, el ultimo es el actual.

    Returns:
        price_relative: estimacion del retorno esperado (>=0).
                        Valor >1 => espera subida (largo), <1 => bajada (corto).
    """
    if len(prices) < 2:
        return 1.0
    p_current = float(prices[-1])
    if p_current <= 0.0:
        return 1.0
    # Ratio de cada precio historico con respecto al actual
    ratios = prices / p_current
    return float(np.median(ratios))


def _linprog_simplex_projection(p_hat: float, eps: float = 1e-5) -> float:
    """
    Calcula la señal optima OLPS bajo restriccion de simplex unitario.

    En la version single-asset (simplex de dimension 1), la solucion
    es trivial: invertir todo el capital cuando p_hat > 1, nada cuando <1.

    Args:
        p_hat: precio relativo predicho (>= 0).
        eps:   tolerancia para clasificar como neutral.

    Returns:
        weight: peso del activo en [-1, 1] (permitimos posicion corta).
    """
    exceso = p_hat - 1.0
    if exceso > eps:
        return 1.0
    elif exceso < -eps:
        return -1.0
    return 0.0


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("olps_rmr")
class OLPSRMRStrategy(AbstractStrategy):
    """
    Estrategia OLPS-RMR para quant_arena.

    Implementa Robust Median Reversion en su variante on-line de activo unico.
    No requiere entrenamiento: la prediccion es O(L) en cada paso.

    La señal es la respuesta del algoritmo al exceso de rendimiento esperado
    calculado con la mediana robusta de la ventana de precios:

        e_t = median(P_{t-L:t}) / P_t - 1

    Con un umbral eps para filtrar ruido de mercado.

    Args:
        universo:        Tickers (un elemento).
        window_size:     L: longitud de la ventana de precios (dias).
        eps:             Umbral de exceso de retorno para emitir señal.
        min_train_days:  Dias minimos antes de emitir señales.
        close_col:       Columna de cierre.
        use_ema_smooth:  Si True, suaviza p_hat con EMA(span=3) para reducir whipsaw.
        ema_span:        Span del EMA de suavizado.
    """

    def __init__(
        self,
        universo:       List[str],
        window_size:    int   = 10,
        eps:            float = 0.003,
        min_train_days: int   = 30,
        close_col:      str   = "Close",
        use_ema_smooth: bool  = True,
        ema_span:       int   = 3,
    ) -> None:
        super().__init__(nombre="olps_rmr", universo=universo)

        self._window     = window_size
        self._eps        = eps
        self._min_train  = min_train_days
        self._close_col  = close_col
        self._smooth     = use_ema_smooth
        self._ema_span   = ema_span

    @property
    def descripcion(self) -> str:
        return (
            f"OLPS-RMR | window={self._window} | eps={self._eps:.4f} "
            f"| smooth={'EMA' if self._smooth else 'raw'}"
        )

    # ------------------------------------------------------------------
    def _compute_exceso_serie(self, close: pd.Series) -> pd.Series:
        """
        Calcula la serie de excesos de retorno esperado e_t = p_hat_t - 1
        para cada fecha usando ventana deslizante.

        Args:
            close: Serie de precios de cierre.

        Returns:
            pd.Series de excesos en [-inf, +inf] (antes de suavizar).
        """
        prices = close.values.astype(np.float64)
        T      = len(prices)
        excesos = np.full(T, np.nan)

        for t in range(self._window, T):
            ventana     = prices[t - self._window : t + 1]  # L+1 elementos, ultimo=actual
            p_hat       = _rmr_predict(ventana)
            excesos[t]  = p_hat - 1.0

        return pd.Series(excesos, index=close.index)

    def generar_señales(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return neutral

        close = hist[self._close_col].astype(float).dropna()
        if len(close) < max(self._min_train, self._window + 1):
            return neutral

        # Calcular serie de excesos
        excesos = self._compute_exceso_serie(close)

        if self._smooth:
            # Suavizar con EMA para reducir señales falsas
            excesos_suav = excesos.ewm(span=self._ema_span, adjust=False).mean()
        else:
            excesos_suav = excesos

        ultimo_exceso = excesos_suav.dropna()
        if ultimo_exceso.empty:
            return neutral

        e_t = float(ultimo_exceso.iloc[-1])

        # Discretizar
        senal = _linprog_simplex_projection(e_t + 1.0, eps=self._eps)

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

    # ------------------------------------------------------------------
    def generar_serie_temporal(
        self,
        datos:     pd.DataFrame,
        fecha_ini: pd.Timestamp,
        fecha_fin: pd.Timestamp,
        close_col: Optional[str] = None,
    ) -> pd.Series:
        """
        Helper de backtesting: genera la serie de señales sobre el rango.

        Util para pruebas y visualizacion sin necesidad del motor walk-forward.

        Args:
            datos:     DataFrame completo con cierre.
            fecha_ini: Inicio del periodo de señales.
            fecha_fin: Fin del periodo de señales.
            close_col: Columna de cierre (None -> usa self._close_col).

        Returns:
            pd.Series de señales {-1, 0, 1} indexada por fecha.
        """
        col    = close_col or self._close_col
        close  = datos[col].astype(float).dropna()
        fechas = close.loc[fecha_ini:fecha_fin].index

        señales = {}
        for fecha in fechas:
            hist_f = datos.loc[datos.index <= fecha]
            s      = self.generar_señales(hist_f, fecha)
            señales[fecha] = float(s.iloc[0]) if len(s) > 0 else 0.0

        return pd.Series(señales, name="señal_rmr")


# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "RMRTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  OLPSRMRStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = OLPSRMRStrategy(
        universo=[TICKER], window_size=10, eps=0.003, min_train_days=30
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "olps_rmr"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='olps_rmr' | universo | descripcion no vacía")

    # TEST 2 — Warm-up (datos insuficientes)
    print("\n[TEST 2] Warm-up (n=15 < min_train=30)...")
    df_short = _make_ohlcv(n=15)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral — warm-up respetado")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=100)...")
    df_normal = _make_ohlcv(n=100, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} in {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Mercado de baja volatilidad (mean-reversion fuerte)
    print("\n[TEST 4] Mercado baja volatilidad (sigma=0.001)...")
    df_lowvol = _make_ohlcv(n=100, mu=0.0, sigma=0.001, seed=7)
    s = strat.generar_señales(df_lowvol, df_lowvol.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} sin crash en mercado de baja vol")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=100).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba señal neutral con datos malformados"
    print("  [OK] neutral retornado sin excepción")

    # TEST 6 — Boundary condition
    print("\n[TEST 6] Boundary: exactamente min_train+window+1 días...")
    df_boundary = _make_ohlcv(n=42)  # window=10 + min_train=30 + 2
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} en boundary condition")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
