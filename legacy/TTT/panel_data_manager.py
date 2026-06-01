# panel_data_manager.py
"""
PanelDataManager — Contenedor de datos de panel (Tier 1, Cross-Sectional)
=========================================================================
Gestiona un panel OHLCV en formato MultiIndex `[date, ticker]` para un universo
de activos (p.ej. ETFs sectoriales: XLK, XLF, XLV, XLE, XLU, ...).

GARANTÍAS DE INTEGRIDAD TEMPORAL:
  · Sin LOOK-AHEAD: los forward returns (target) usan `shift(-h)` POR TICKER y se
    exponen explícitamente como objetivo, nunca como feature contemporáneo.
  · Sin SESGO DE SUPERVIVENCIA fabricado: los días en que un ticker aún no cotiza
    quedan como NaN y se EXCLUYEN de la sección cruzada de esa fecha (vía máscara
    de disponibilidad), en lugar de rellenarse hacia atrás (lo que inventaría
    historia inexistente).
  · Las transformaciones `wide` usan `.unstack('ticker')` → matrices Date×Ticker
    listas para operaciones vectorizadas en el `CrossSectionalEngine`.

Limitación honesta: yfinance solo entrega tickers SUPERVIVIENTES (no incluye
deslistados). La arquitectura admite un panel deslistado-inclusivo si se inyecta
externamente vía `from_panel()`; la fuente yahoo no lo resuelve por sí sola.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]


class PanelDataManager:
    """
    Gestor de datos de panel para investigación cross-sectional.

    Args:
        tickers:               Universo de tickers (lista de símbolos).
        trading_days_per_year: Convención de anualización (252 diario).

    Atributos:
        panel: DataFrame con MultiIndex `[date, ticker]` y columnas OHLCV.
        tickers: universo efectivamente cargado.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        trading_days_per_year: int = 252,
    ) -> None:
        self.tickers: List[str] = list(tickers)
        self.trading_days_per_year = trading_days_per_year
        self.panel: pd.DataFrame = pd.DataFrame()

    # ── CONSTRUCCIÓN DEL PANEL ──────────────────────────────────────────────────

    def load(self, start: str, end: str, source: str = "yahoo") -> "PanelDataManager":
        """Carga el panel OHLCV del universo y lo normaliza a `[date, ticker]`."""
        if source != "yahoo":
            raise ValueError(f"source='{source}' no soportado (usar 'yahoo').")
        self.panel = self._fetch_yahoo(self.tickers, start, end)
        logger.info(
            "PanelDataManager: %d tickers | %s → %s | %d filas de panel",
            self.panel.index.get_level_values("ticker").nunique(),
            self.panel.index.get_level_values("date").min().date(),
            self.panel.index.get_level_values("date").max().date(),
            len(self.panel),
        )
        return self

    @classmethod
    def from_panel(cls, panel: pd.DataFrame) -> "PanelDataManager":
        """Inyecta un panel preconstruido (p.ej. universo deslistado-inclusivo)."""
        tickers = list(panel.index.get_level_values("ticker").unique())
        obj = cls(tickers)
        obj.panel = panel.sort_index()
        return obj

    @staticmethod
    def _fetch_yahoo(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Descarga multi-ticker vía yfinance y lo apila a panel `[date, ticker]`."""
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("yfinance no instalado: pip install yfinance") from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, group_by="column",
            )
        if raw.empty:
            raise ValueError("yfinance devolvió un panel vacío.")

        # Columnas MultiIndex (field, ticker): detectar el nivel de los tickers.
        if isinstance(raw.columns, pd.MultiIndex):
            tickset = set(tickers)
            lvl_ticker = 1 if tickset & set(raw.columns.get_level_values(1)) else 0
            panel = raw.stack(level=lvl_ticker, future_stack=True)
        else:  # un solo ticker
            panel = raw.copy()
            panel["ticker"] = tickers[0]
            panel = panel.set_index("ticker", append=True)

        panel.index.names = ["date", "ticker"]
        panel.columns = [str(c).lower() for c in panel.columns]
        panel = panel[[c for c in _OHLCV if c in panel.columns]]
        panel = panel.sort_index()
        # Eliminar filas totalmente vacías (ticker inexistente esa fecha).
        panel = panel.dropna(how="all")
        return panel

    # ── MATRICES WIDE (Date × Ticker) ───────────────────────────────────────────

    def wide(self, field: str = "close") -> pd.DataFrame:
        """Devuelve el campo solicitado como matriz Date×Ticker (`.unstack`)."""
        self._require_panel()
        if field not in self.panel.columns:
            raise KeyError(f"Campo '{field}' ausente. Disponibles: {list(self.panel.columns)}")
        return self.panel[field].unstack("ticker").sort_index()

    def availability_mask(self) -> pd.DataFrame:
        """
        Máscara booleana Date×Ticker: True si el activo cotiza (close válido) ese día.
        Permite al motor excluir activos no disponibles SIN rellenar look-ahead.
        """
        return self.wide("close").notna()

    # ── TARGETS Y RETORNOS (anti look-ahead) ────────────────────────────────────

    def simple_returns(self, price_col: str = "close") -> pd.DataFrame:
        """Retornos simples CONTEMPORÁNEOS por ticker (matriz Date×Ticker)."""
        return self.wide(price_col).pct_change()

    def log_returns(self, price_col: str = "close") -> pd.DataFrame:
        """Retornos logarítmicos contemporáneos por ticker."""
        px = self.wide(price_col)
        return np.log(px / px.shift(1))

    def forward_returns(self, horizon: int = 1, price_col: str = "close") -> pd.DataFrame:
        """
        TARGET: retorno hacia adelante a `horizon` días por ticker.
            fwd_ret(t) = close(t+h)/close(t) − 1
        Calculado con `shift(-h)` POR TICKER (la matriz wide ya está por ticker en
        columnas, así que el shift temporal no mezcla activos). Las últimas `h`
        filas quedan NaN (sin futuro) — es un OBJETIVO, jamás un feature.
        """
        px = self.wide(price_col)
        return px.shift(-horizon) / px - 1.0

    def realized_vol(self, window: int = 21, price_col: str = "close") -> pd.DataFrame:
        """Vol realizada anualizada por ticker (para ponderación inverse-vol)."""
        ret = self.log_returns(price_col)
        return ret.rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(
            self.trading_days_per_year
        )

    # ── UTILIDADES ──────────────────────────────────────────────────────────────

    @property
    def n_assets(self) -> int:
        if self.panel.empty:
            return 0
        return self.panel.index.get_level_values("ticker").nunique()

    def _require_panel(self) -> None:
        if self.panel.empty:
            raise RuntimeError("Panel vacío: invocar load() o from_panel() primero.")

    def __repr__(self) -> str:
        return f"PanelDataManager(tickers={self.tickers}, n_assets={self.n_assets})"
