# data_manager.py
"""
DataManager — Proyecto Paraguay V2 · Motor Walk-Forward TTT/CPCV (DIARIO / 1D)
==============================================================================
FRECUENCIA: DIARIA (1D) — EXCLUSIVAMENTE. Una fila = un día de mercado.
Tras descartar la vía intradía (imposibilidad de datos 5-min reales; prohibido
sintetizarlos), toda la arquitectura opera en temporalidad diaria con su
significado físico natural: `trading_days_per_year=252`, ventanas walk-forward
y de purga/embargo en DÍAS, y features diarios (RSI 14d, SMA 50/200d, etc.).
La fuente de datos diaria de referencia es yfinance (SPY / ^GSPC, 1997→hoy).

Gestión de datos con integridad temporal estricta para motores de validación
basados en TrueSkill Through Time + Purged Cross-Validation (López de Prado, 2018).

DISEÑO TEMPORAL:
  · Purging  (Cap. 7): elimina observaciones IS cuyo horizonte de etiqueta se
    solapa con el período OOS, cortando la correlación serial contaminante.
  · Embargo  (Cap. 7): buffer adicional post-purging para absorber correlación
    serial residual de microestructura (momentum de muy corto plazo).
  · CPCV     (Cap. 12): generación de C(N,k) splits combinatoriales para
    distribuir el riesgo de overfitting en lugar de una única realización.

Compatibilidad garantizada con walk_forward_engine_con_TTT.py:
  · Interface: load_data(symbol, start, end) → DataFrame con log-returns
  · Interface: get_train_test_splits(df, n_splits, ...) → List[Tuple[...]]
  · Interface: get_cpcv_splits(df, n_folds, n_test_folds, ...) → List[Tuple[...]]

Referencias:
  López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
  Bailey, D. & López de Prado, M. (2014). JPM 40(5). SSRN: 2460551.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from feature_engineer import FeatureConfig, FeatureEngineer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_MIN_ROWS_FOR_SPLIT   = 252          # 1 año mínimo para calcular estadísticas
_EMBARGO_MIN_DAYS     = 5            # Mínimo práctico; regla: >= 0.01 * train_size


# ══════════════════════════════════════════════════════════════════════════════
# DATA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """
    Gestor de datos con integridad temporal para motores TTT/CPCV.

    Responsabilidades principales:
    1. Carga y normalización de datos OHLCV (Yahoo Finance o CSV local).
    2. Cálculo de retornos logarítmicos como feature base para backtests.
    3. Segmentación temporal con purging + embargo estrictos.
    4. Generación de splits CPCV combinatoriales.

    Args:
        cache_dir: Directorio para cachear datos descargados. None = sin caché.
        trading_days_per_year: Convención de días hábiles (252 estándar USA).
        add_technical_features: Si True, load_data() añade la batería de
            indicadores técnicos (FeatureEngineer) tras calcular log-returns.
        feature_config: FeatureConfig opcional con las ventanas de los features.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        trading_days_per_year: int = 252,
        add_technical_features: bool = False,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.trading_days_per_year = trading_days_per_year
        self.add_technical_features = add_technical_features
        # La anualización de la volatilidad diaria hereda la convención 252.
        if feature_config is None:
            feature_config = FeatureConfig(trading_days=trading_days_per_year)
        self.feature_engineer = FeatureEngineer(feature_config)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"DataManager: caché en '{self.cache_dir}'")

    # ──────────────────────────────────────────────────────────────────────────
    # CARGA DE DATOS
    # ──────────────────────────────────────────────────────────────────────────

    def load_data(
        self,
        symbol: str,
        start: str,
        end: str,
        source: str = "yahoo",
        csv_path: Optional[Union[str, Path]] = None,
        adjust_prices: bool = True,
    ) -> pd.DataFrame:
        """
        Carga datos OHLCV y añade retornos logarítmicos.

        La columna `log_return` es el feature principal que consumen los
        backtests del motor TTT. Se calcula sobre `adj_close` si está
        disponible (corrige splits/dividendos), con fallback a `close`.

        Args:
            symbol:        Ticker (e.g. 'SPY', 'COLCAP').
            start:         Fecha inicio ISO-8601 (e.g. '1997-01-01').
            end:           Fecha fin ISO-8601 (e.g. '2024-01-01').
            source:        'yahoo' (yfinance) | 'csv' (requiere csv_path).
            csv_path:      Ruta al CSV local; requerido si source='csv'.
            adjust_prices: Si True, usa adj_close para log-returns.

        Returns:
            DataFrame con DatetimeIndex (tz-naive, frecuencia diaria),
            columnas OHLCV + adj_close + log_return. Index name = 'date'.

        Raises:
            ValueError:    Datos vacíos, símbolo inválido, o rango sin datos.
            FileNotFoundError: Si source='csv' y csv_path no existe.
        """
        cache_key = f"{symbol}_{start}_{end}.parquet"
        cache_file = self.cache_dir / cache_key if self.cache_dir else None

        # ── Intentar leer desde caché ──────────────────────────────────────
        if cache_file and cache_file.exists():
            df = pd.read_parquet(cache_file)
            logger.info(f"Cache hit: {symbol} [{start} → {end}]")
            df = self._filter_date_range(df, start, end)
            df = self._ensure_log_returns(df, adjust_prices)
            return self._maybe_add_features(df)

        # ── Descarga / lectura ─────────────────────────────────────────────
        if source == "yahoo":
            df = self._fetch_yahoo(symbol, start, end)
        elif source == "csv":
            df = self._load_csv(csv_path, symbol)
            df = self._filter_date_range(df, start, end)
        elif source == "parquet":
            df = self._load_parquet(csv_path, symbol)
            df = self._filter_date_range(df, start, end)
        else:
            raise ValueError(
                f"source='{source}' desconocido. Usar 'yahoo', 'csv' o 'parquet'."
            )

        # ── Validaciones de integridad ─────────────────────────────────────
        self._validate_ohlcv(df, symbol)

        # ── Log-returns ────────────────────────────────────────────────────
        df = self._ensure_log_returns(df, adjust_prices)

        # ── Indicadores técnicos (opcional) ────────────────────────────────
        df = self._maybe_add_features(df)

        # ── Guardar en caché ───────────────────────────────────────────────
        if cache_file:
            df.to_parquet(cache_file, index=True)
            logger.debug(f"Cache guardado: {cache_file}")

        logger.info(
            f"DataManager.load_data: {symbol} | {df.index[0].date()} → "
            f"{df.index[-1].date()} | {len(df)} filas | "
            f"NaN log_return: {df['log_return'].isna().sum()}"
        )
        return df

    def _fetch_yahoo(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Descarga datos via yfinance con normalización estricta de columnas."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance no instalado. Ejecutar: pip install yfinance"
            ) from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )

        if raw.empty:
            raise ValueError(
                f"yfinance devolvió DataFrame vacío para '{symbol}' "
                f"en [{start}, {end}]. Verificar ticker y rango."
            )

        # yfinance puede devolver MultiIndex si se descarga un ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

        # Mapeo de nombres legacy de yfinance
        rename_map = {
            "adj_close": "adj_close",
            "adj close": "adj_close",
            "stock_splits": "splits",
        }
        raw = raw.rename(columns=rename_map)

        # Normalizar índice
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "date"
        raw = raw.sort_index()

        # Eliminar filas completamente vacías (fines de semana residuales)
        raw = raw.dropna(how="all")

        return raw

    def _maybe_add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade la batería de indicadores técnicos si add_technical_features=True.

        Idempotente: si los features ya están presentes (e.g. dataset cacheado
        o pre-enriquecido), no recalcula. Recorta el warm-up de forma segura
        (sin look-ahead), delegando en FeatureEngineer.trim_warmup().
        """
        if not self.add_technical_features:
            return df
        if "rsi" in df.columns:          # sentinela: features ya presentes
            return df
        return self.feature_engineer.transform(df, trim_warmup=True)

    def engineer_features(
        self, df: pd.DataFrame, trim_warmup: bool = True
    ) -> pd.DataFrame:
        """
        API pública para inyectar los indicadores técnicos a un DataFrame OHLCV
        ya cargado (independiente del flag de instancia). Delega en FeatureEngineer.
        """
        return self.feature_engineer.transform(df, trim_warmup=trim_warmup)

    def _load_parquet(
        self,
        parquet_path: Optional[Union[str, Path]],
        symbol: str,
    ) -> pd.DataFrame:
        """Carga un parquet OHLCV diario (1D) local, normalizando columnas."""
        if parquet_path is None:
            raise ValueError("csv_path (ruta al parquet) requerido cuando source='parquet'.")
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet no encontrado: {path}")
        df = pd.read_parquet(path)
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{path.name}: el índice no es DatetimeIndex.")
        df.index.name = "date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["symbol"] = symbol
        return df

    def _load_csv(
        self,
        csv_path: Optional[Union[str, Path]],
        symbol: str,
    ) -> pd.DataFrame:
        """Carga CSV local con inferencia de columnas y fechas."""
        if csv_path is None:
            raise ValueError("csv_path requerido cuando source='csv'.")

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {path}")

        df = pd.read_csv(path, parse_dates=True, index_col=0)
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.sort_index()
        df["symbol"] = symbol
        return df

    def _filter_date_range(
        self, df: pd.DataFrame, start: str, end: str
    ) -> pd.DataFrame:
        """Filtra el DataFrame al rango [start, end] inclusive."""
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        df = df.loc[(df.index >= s) & (df.index <= e)]
        if df.empty:
            raise ValueError(
                f"Sin datos en el rango [{start}, {end}] tras filtrado."
            )
        return df

    def _validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Validaciones de integridad mínimas para un DataFrame OHLCV.

        NO lanza excepciones por NaNs aislados (se manejan en _ensure_log_returns),
        pero SÍ falla ante problemas estructurales que invalidan el dataset.
        """
        missing_cols = [c for c in _REQUIRED_OHLCV_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"'{symbol}': columnas requeridas ausentes: {missing_cols}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"'{symbol}': índice no es DatetimeIndex.")

        if not df.index.is_monotonic_increasing:
            raise ValueError(
                f"'{symbol}': índice no es monótono creciente. "
                "Ordenar por fecha antes de usar DataManager."
            )

        if df.index.has_duplicates:
            n_dup = df.index.duplicated().sum()
            logger.warning(
                f"'{symbol}': {n_dup} timestamps duplicados detectados — "
                "se conservará la última entrada por fecha."
            )
            # El motor TTT no tolera duplicados — deduplicar aquí
            df = df[~df.index.duplicated(keep="last")]

        # Precios negativos o cero: error crítico
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            nonpos = (df[col] <= 0).sum()
            if nonpos > 0:
                raise ValueError(
                    f"'{symbol}': {nonpos} valores <= 0 en columna '{col}'. "
                    "Datos corruptos o mal ajustados."
                )

        # Consistencia H >= L
        if "high" in df.columns and "low" in df.columns:
            broken = (df["high"] < df["low"]).sum()
            if broken > 0:
                logger.warning(
                    f"'{symbol}': {broken} filas con high < low. "
                    "Posibles errores en los datos de origen."
                )

        # Fracción de NaNs
        total = df.shape[0] * df.shape[1]
        nan_frac = df.isna().sum().sum() / max(1, total)
        if nan_frac > 0.10:
            logger.warning(
                f"'{symbol}': fracción NaN elevada = {nan_frac:.2%}. "
                "Verificar calidad de datos."
            )

    def _ensure_log_returns(
        self, df: pd.DataFrame, adjust_prices: bool
    ) -> pd.DataFrame:
        """
        Añade columna `log_return` al DataFrame si no existe.

        Usa `adj_close` si disponible y adjust_prices=True (corrige
        splits y dividendos, crítico para períodos >1 año). Fallback a `close`.

        Los NaNs se rellenan con forward-fill + backward-fill para no romper
        la continuidad temporal, pero se loguea si hay más del 1%.
        """
        if "log_return" in df.columns:
            return df

        price_col = (
            "adj_close"
            if (adjust_prices and "adj_close" in df.columns)
            else "close"
        )

        # Asegurar dtype float antes de log
        prices = df[price_col].astype(float)

        # Forward + back fill para gaps (e.g. datos de Colombia con huecos)
        nan_before = prices.isna().sum()
        if nan_before > 0:
            prices = prices.ffill().bfill()
            nan_after = prices.isna().sum()
            if nan_after > 0:
                raise ValueError(
                    f"Columna '{price_col}' tiene {nan_after} NaNs "
                    "no rellenables (inicio/fin del dataset)."
                )
            if nan_before / len(prices) > 0.01:
                logger.warning(
                    f"Se rellenaron {nan_before} NaNs en '{price_col}' "
                    f"({nan_before/len(prices):.2%} del total)."
                )

        # Retorno logarítmico: ln(P_t / P_{t-1})
        log_ret = np.log(prices / prices.shift(1))

        # El primer valor siempre es NaN — dropping aquí violaría continuidad
        # temporal; el motor lo gestiona por split. Se loguea.
        df = df.copy()
        df["log_return"] = log_ret
        df["price_col_used"] = price_col   # trazabilidad

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # SPLITS WALK-FORWARD (Purging + Embargo)
    # ──────────────────────────────────────────────────────────────────────────

    def get_train_test_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 10,
        min_train_days: int = 252,
        test_days: int = 21,
        purge_days: int = 5,
        embargo_days: int = 10,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
        """
        Genera splits walk-forward con Purging + Embargo estrictos.

        ALGORITMO (López de Prado, 2018, Cap. 7):
        ─────────────────────────────────────────────────────────────────────
        Sea D el conjunto de datos de longitud T días.

        Para cada split i (i = 0..n_splits-1):
          1. test_end   = D.end - (n_splits - 1 - i) * test_days
          2. test_start = test_end - test_days + 1
          3. embargo_end   = test_start - 1
          4. embargo_start = embargo_end - embargo_days + 1
             (el motor NO entrena sobre este buffer — absorbe autocorrelación)
          5. purge_end  = embargo_start - 1
             (el motor purga observaciones IS con horizonte solapado)
          6. train_start = max(D.start, test_end - min_train_days - test_days
                              - embargo_days - purge_days)
          7. train_end  = purge_end - purge_days
             (se eliminan las últimas purge_days del entrenamiento)

        El resultado garantiza separación temporal estricta:
          train  ──► [purge_days] ──► [embargo_days] ──► test

        Args:
            df:             DataFrame con DatetimeIndex y columna log_return.
            n_splits:       Número de folds walk-forward.
            min_train_days: Mínimo de días IS para primer split.
            test_days:      Días por período OOS.
            purge_days:     Días de purging (horizonte de etiqueta).
            embargo_days:   Días de embargo post-purge.

        Returns:
            Lista de (df_train, df_test, meta) donde meta es un dict con
            las fechas de cada región para trazabilidad y debugging.

        Raises:
            ValueError: Si el DataFrame es demasiado corto para generar splits.
        """
        self._validate_split_inputs(df, n_splits, min_train_days, test_days,
                                    purge_days, embargo_days)

        dates = df.index
        total_required = min_train_days + n_splits * test_days + embargo_days + purge_days
        if len(dates) < total_required:
            raise ValueError(
                f"DataFrame insuficiente: {len(dates)} días disponibles, "
                f"se necesitan ~{total_required} para {n_splits} splits con "
                f"min_train={min_train_days}, test={test_days}, "
                f"purge={purge_days}, embargo={embargo_days}."
            )

        splits: List[Tuple[pd.DataFrame, pd.DataFrame, dict]] = []
        buffer = purge_days + embargo_days  # separación mínima IS→OOS

        for i in range(n_splits):
            # ── Período OOS ───────────────────────────────────────────────
            # Anclar desde el final del dataset hacia atrás
            oos_end_idx   = len(dates) - 1 - (n_splits - 1 - i) * test_days
            oos_start_idx = oos_end_idx - test_days + 1

            if oos_start_idx < 0 or oos_end_idx >= len(dates):
                logger.warning(f"Split {i}: índices OOS fuera de rango — saltando.")
                continue

            test_start = dates[oos_start_idx]
            test_end   = dates[oos_end_idx]

            # ── Embargo: buffer entre IS y OOS ────────────────────────────
            # Las últimas embargo_days antes del test son zona excluida
            embargo_end_idx   = oos_start_idx - 1
            embargo_start_idx = max(0, embargo_end_idx - embargo_days + 1)
            embargo_start     = dates[embargo_start_idx]

            # ── Purging: cortar el IS antes del buffer ────────────────────
            # Eliminar las últimas purge_days del IS (etiquetas solapadas)
            train_end_idx   = embargo_start_idx - 1 - purge_days
            if train_end_idx < 0:
                logger.warning(f"Split {i}: train_end_idx < 0 tras purging — saltando.")
                continue

            # ── Inicio del IS: asegurar min_train_days ────────────────────
            train_start_idx = max(0, train_end_idx - min_train_days + 1)
            train_start     = dates[train_start_idx]
            train_end       = dates[train_end_idx]

            # ── Verificación mínima de longitud IS ────────────────────────
            n_train = train_end_idx - train_start_idx + 1
            if n_train < min_train_days:
                logger.warning(
                    f"Split {i}: IS demasiado corto ({n_train} < {min_train_days}) "
                    "— saltando. Aumentar el dataset o reducir min_train_days."
                )
                continue

            # ── Extraer sub-DataFrames ────────────────────────────────────
            df_train = df.loc[train_start:train_end].copy()
            df_test  = df.loc[test_start:test_end].copy()

            # Eliminar primera fila de log_return si es NaN (inicio IS)
            df_train = df_train.dropna(subset=["log_return"])
            df_test  = df_test.dropna(subset=["log_return"])

            if df_train.empty or df_test.empty:
                logger.warning(f"Split {i}: train o test vacío tras dropna — saltando.")
                continue

            meta = {
                "split_idx":       i,
                "train_start":     train_start,
                "train_end":       train_end,
                "embargo_start":   embargo_start,
                "test_start":      test_start,
                "test_end":        test_end,
                "n_train_days":    len(df_train),
                "n_test_days":     len(df_test),
                "purge_days":      purge_days,
                "embargo_days":    embargo_days,
            }

            splits.append((df_train, df_test, meta))
            logger.debug(
                f"Split {i}: IS [{train_start.date()}→{train_end.date()}] "
                f"({len(df_train)}d) | "
                f"Embargo {embargo_days}d | "
                f"OOS [{test_start.date()}→{test_end.date()}] "
                f"({len(df_test)}d)"
            )

        if not splits:
            raise ValueError(
                "No se generó ningún split válido. Revisar parámetros de "
                "longitud de datos, purge_days y embargo_days."
            )

        logger.info(
            f"get_train_test_splits: {len(splits)}/{n_splits} splits válidos | "
            f"purge={purge_days}d | embargo={embargo_days}d"
        )
        return splits

    # ──────────────────────────────────────────────────────────────────────────
    # CPCV — Combinatorial Purged Cross-Validation
    # ──────────────────────────────────────────────────────────────────────────

    def get_cpcv_splits(
        self,
        df: pd.DataFrame,
        n_folds: int = 6,
        n_test_folds: int = 2,
        min_train_days: int = 252,
        purge_days: int = 5,
        embargo_days: int = 10,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
        """
        Genera C(N, k) splits combinatoriales purged (CPCV).

        DISEÑO (López de Prado, 2018, Cap. 12):
        ─────────────────────────────────────────────────────────────────────
        · El dataset se divide en N folds contiguos de igual longitud.
        · Para cada combinación de k folds como test (C(N,k) total):
            - El IS es la unión de los N-k folds restantes.
            - Se aplica purging en los bordes IS/OOS y embargo entre regiones.
        · Produce una DISTRIBUCIÓN de métricas OOS (no una realización única),
          reduciendo el Probability of Backtest Overfitting (PBO).

        El número de combinaciones C(N,k) puede ser grande. Con N=6, k=2:
        C(6,2) = 15 combinaciones. Con N=10, k=3: C(10,3) = 120.
        El motor TTT debe estar preparado para acumular todos los folds.

        Args:
            df:             DataFrame con DatetimeIndex y log_return.
            n_folds:        N — número de folds totales.
            n_test_folds:   k — folds usados como test en cada combinación.
            min_train_days: Mínimo de días IS por combinación.
            purge_days:     Días de purging en bordes IS/OOS.
            embargo_days:   Días de embargo post-purge.

        Returns:
            Lista de (df_train, df_test, meta) para cada combinación CPCV.
            meta incluye combo_idx, test_fold_indices, y fechas de IS/OOS.

        Raises:
            ValueError: Si k >= N o datos insuficientes.
        """
        from math import comb as math_comb

        if n_test_folds >= n_folds:
            raise ValueError(
                f"n_test_folds={n_test_folds} debe ser < n_folds={n_folds}."
            )

        n_combos = math_comb(n_folds, n_test_folds)
        logger.info(
            f"CPCV: C({n_folds},{n_test_folds}) = {n_combos} combinaciones | "
            f"purge={purge_days}d | embargo={embargo_days}d"
        )

        # ── Dividir el dataset en N folds contiguos ───────────────────────
        dates = df.index
        fold_size = len(dates) // n_folds
        if fold_size < 20:
            raise ValueError(
                f"Fold size = {fold_size} demasiado pequeño. "
                f"Aumentar datos o reducir n_folds."
            )

        fold_boundaries: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for f in range(n_folds):
            start_idx = f * fold_size
            end_idx   = (f + 1) * fold_size - 1 if f < n_folds - 1 else len(dates) - 1
            fold_boundaries.append((dates[start_idx], dates[end_idx]))

        logger.debug(f"CPCV fold boundaries: {fold_boundaries}")

        cpcv_splits: List[Tuple[pd.DataFrame, pd.DataFrame, dict]] = []
        buffer = purge_days + embargo_days

        # ── Generar todas las combinaciones ───────────────────────────────
        all_fold_indices = list(range(n_folds))
        for combo_idx, test_fold_indices in enumerate(
            combinations(all_fold_indices, n_test_folds)
        ):
            train_fold_indices = [f for f in all_fold_indices
                                  if f not in test_fold_indices]

            # ── OOS: unión de folds test ──────────────────────────────────
            test_start = fold_boundaries[test_fold_indices[0]][0]
            test_end   = fold_boundaries[test_fold_indices[-1]][1]

            # Manejar folds test no-contiguos: usar el rango completo del primero al último
            # (conservador — simplificación práctica para CPCV con k=2)
            df_test = df.loc[test_start:test_end].copy()

            # ── IS: union de folds train, con purging en bordes ───────────
            train_frames: List[pd.DataFrame] = []
            for f_idx in train_fold_indices:
                f_start, f_end = fold_boundaries[f_idx]

                # Purging: si el fold de entrenamiento está adyacente a un fold de test,
                # recortar las purge_days en el borde adyacente.
                f_start_adj, f_end_adj = self._apply_purging_to_fold(
                    f_idx, f_start, f_end, test_fold_indices,
                    fold_boundaries, purge_days, embargo_days, df
                )

                if f_start_adj is None or f_end_adj is None:
                    continue  # fold completamente purgado

                chunk = df.loc[f_start_adj:f_end_adj].dropna(subset=["log_return"])
                if not chunk.empty:
                    train_frames.append(chunk)

            if not train_frames:
                logger.debug(f"CPCV combo {combo_idx}: IS vacío tras purging — saltando.")
                continue

            df_train = pd.concat(train_frames).sort_index()
            df_train = df_train[~df_train.index.duplicated(keep="last")]

            # Verificar longitud mínima IS
            if len(df_train) < min_train_days:
                logger.debug(
                    f"CPCV combo {combo_idx}: IS {len(df_train)} < "
                    f"min_train_days {min_train_days} — saltando."
                )
                continue

            df_test = df_test.dropna(subset=["log_return"])
            if df_test.empty:
                continue

            meta = {
                "combo_idx":          combo_idx,
                "test_fold_indices":  list(test_fold_indices),
                "train_fold_indices": train_fold_indices,
                "test_start":         test_start,
                "test_end":           test_end,
                "n_train_days":       len(df_train),
                "n_test_days":        len(df_test),
                "purge_days":         purge_days,
                "embargo_days":       embargo_days,
                "n_folds":            n_folds,
                "n_test_folds":       n_test_folds,
            }

            cpcv_splits.append((df_train, df_test, meta))

        logger.info(
            f"CPCV: {len(cpcv_splits)}/{n_combos} combinaciones válidas generadas."
        )
        return cpcv_splits

    def _apply_purging_to_fold(
        self,
        f_idx: int,
        f_start: pd.Timestamp,
        f_end: pd.Timestamp,
        test_fold_indices: Tuple[int, ...],
        fold_boundaries: List[Tuple[pd.Timestamp, pd.Timestamp]],
        purge_days: int,
        embargo_days: int,
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Recorta los bordes de un fold IS adyacente a folds OOS.

        Aplica purging (horizonte de etiqueta) y embargo (buffer serial)
        en los bordes donde un fold IS linda con uno OOS.

        Returns:
            (start_adj, end_adj) recortados, o (None, None) si el fold
            queda completamente purgado.
        """
        dates = df.index
        buffer = purge_days + embargo_days
        start_adj, end_adj = f_start, f_end

        for t_idx in test_fold_indices:
            t_start, t_end = fold_boundaries[t_idx]

            # Fold IS está inmediatamente antes de un fold OOS
            if f_idx == t_idx - 1:
                # Recortar el final del IS: quitar (purge + embargo) días
                candidates = dates[dates < t_start]
                if len(candidates) <= buffer:
                    return None, None   # fold completamente purgado
                end_adj = min(end_adj, candidates[-buffer - 1])

            # Fold IS está inmediatamente después de un fold OOS
            if f_idx == t_idx + 1:
                # Recortar el inicio del IS: quitar embargo días
                candidates = dates[dates > t_end]
                if len(candidates) <= embargo_days:
                    return None, None
                start_adj = max(start_adj, candidates[embargo_days])

        if start_adj >= end_adj:
            return None, None

        return start_adj, end_adj

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN INTERNA
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_split_inputs(
        self,
        df: pd.DataFrame,
        n_splits: int,
        min_train_days: int,
        test_days: int,
        purge_days: int,
        embargo_days: int,
    ) -> None:
        """Verifica parámetros de splitting antes de ejecutar."""
        if df.empty:
            raise ValueError("DataFrame vacío: no se puede generar splits.")

        if "log_return" not in df.columns:
            raise ValueError(
                "Columna 'log_return' ausente. "
                "Usar DataManager.load_data() o _ensure_log_returns()."
            )

        if n_splits < 1:
            raise ValueError(f"n_splits={n_splits} debe ser >= 1.")

        if test_days < 1:
            raise ValueError(f"test_days={test_days} debe ser >= 1.")

        if purge_days < 0:
            raise ValueError(f"purge_days={purge_days} debe ser >= 0.")

        if embargo_days < 0:
            raise ValueError(f"embargo_days={embargo_days} debe ser >= 0.")

        # Regla de López de Prado: embargo >= 0.01 * min_train_days
        recommended_embargo = max(_EMBARGO_MIN_DAYS, int(0.01 * min_train_days))
        if embargo_days < recommended_embargo:
            logger.warning(
                f"embargo_days={embargo_days} < recomendado={recommended_embargo} "
                f"(regla: >= max(5, 0.01 * min_train_days)). "
                "Riesgo de correlación serial residual en OOS."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df debe tener DatetimeIndex.")

    # ──────────────────────────────────────────────────────────────────────────
    # UTILIDADES PÚBLICAS
    # ──────────────────────────────────────────────────────────────────────────

    def compute_benchmark_returns(
        self,
        symbol: str = "SPY",
        start: str = "1997-01-01",
        end: str = "2024-01-01",
    ) -> pd.Series:
        """
        Descarga retornos logarítmicos del benchmark.
        Útil para construir el 'jugador B' del modelo TTT (alpha=0).

        Returns:
            pd.Series de log_return con DatetimeIndex.
        """
        df_bm = self.load_data(symbol, start, end, adjust_prices=True)
        return df_bm["log_return"].rename(f"{symbol}_log_return")

    def describe_coverage(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """
        Resumen de cobertura temporal para auditoría del dataset.

        Returns:
            Dict con start, end, n_rows, nan_pct_close, nan_pct_log_return.
        """
        report = {
            "symbol":              symbol,
            "start":               df.index[0].date().isoformat() if not df.empty else None,
            "end":                 df.index[-1].date().isoformat() if not df.empty else None,
            "n_rows":              len(df),
            "n_trading_days":      df.index.nunique(),
        }
        if "close" in df.columns:
            report["nan_pct_close"] = float(df["close"].isna().mean())
        if "log_return" in df.columns:
            report["nan_pct_log_return"] = float(df["log_return"].isna().mean())
        return report
