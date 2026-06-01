# build_features.py
"""
DailyFeaturePipeline — Proyecto Paraguay V2 · Dataset DIARIO enriquecido (1D)
=============================================================================
Equivalente real (en este repositorio) del histórico `scripts/fetch_sp500.py`:
descarga OHLCV DIARIO del S&P 500 (SPY / ^GSPC, 1997→hoy) vía yfinance, le inyecta
la batería de indicadores técnicos de `FeatureEngineer`, recorta el periodo de
calentamiento sin look-ahead y persiste `sp500_daily_1997_to_today_features.parquet`.

FRECUENCIA: DIARIA (1D) — EXCLUSIVAMENTE.
Se descartó por completo la vía intradía de 5 minutos: es imposible adquirir datos
5-min reales y está PROHIBIDO sintetizarlos (destruiría la validez del backtest).
No existe ninguna conversión "días → barras"; una fila es siempre un día de mercado.

Diseño estrictamente POO: toda la lógica vive en la clase `DailyFeaturePipeline`.
El bloque `__main__` es solo un thin entrypoint — sin matemática procedimental suelta.

Uso:
    python build_features.py                       # SPY diario 1997→hoy
    python build_features.py ^GSPC                 # índice S&P 500 contado
    python build_features.py SPY 2005-01-01        # ticker + fecha inicio
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

from data_manager import DataManager
from feature_engineer import FeatureConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_features")


class DailyFeaturePipeline:
    """
    Orquesta descarga (yfinance) → feature engineering DIARIO → recorte seguro
    → escritura del parquet enriquecido.

    Args:
        symbol:    Ticker diario (default 'SPY'). Usar '^GSPC' para el índice.
        start:     Fecha inicio ISO-8601 (default '1997-01-01').
        end:       Fecha fin ISO-8601 (default = hoy).
        config:    FeatureConfig opcional (ventanas DIARIAS de los indicadores).
        out_dir:   Directorio de salida del parquet (default '.').
    """

    def __init__(
        self,
        symbol: str = "SPY",
        start: str = "1997-01-01",
        end: str | None = None,
        config: FeatureConfig | None = None,
        out_dir: str | Path = ".",
    ) -> None:
        self.symbol = symbol
        self.start = start
        self.end = end or dt.date.today().isoformat()
        self.out_dir = Path(out_dir)
        # DataManager en modo DIARIO con features técnicos activados.
        self.dm = DataManager(
            trading_days_per_year=252,
            add_technical_features=True,
            feature_config=config,
        )

    @property
    def output_path(self) -> Path:
        return self.out_dir / "sp500_daily_1997_to_today_features.parquet"

    def run(self, save: bool = True) -> pd.DataFrame:
        """
        Ejecuta el pipeline diario completo y devuelve el DataFrame enriquecido.

        load_data(source='yahoo') ya añade log_return + los indicadores técnicos
        (vía DataManager.add_technical_features) y recorta el warm-up diario.
        """
        df = self.dm.load_data(
            self.symbol, self.start, self.end,
            source="yahoo", adjust_prices=True,
        )
        logger.info(
            "Dataset DIARIO enriquecido: %s | %s → %s | filas=%d | columnas=%d",
            self.symbol, df.index[0].date(), df.index[-1].date(),
            len(df), df.shape[1],
        )

        if save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.output_path, index=True)
            logger.info("Guardado en %s", self.output_path)

        return df


def _report(df: pd.DataFrame) -> None:
    """Imprime df.info() y un resumen de NaN del dataset diario enriquecido."""
    print("\n" + "=" * 78)
    print("df.info():")
    df.info(show_counts=True)
    print("-" * 78)
    print(f"NaN totales restantes: {int(df.isna().sum().sum())}")
    print(f"Columnas ({df.shape[1]}): {df.columns.tolist()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = sys.argv[2] if len(sys.argv) > 2 else "1997-01-01"

    pipeline = DailyFeaturePipeline(symbol=symbol, start=start)
    enriched = pipeline.run(save=True)
    _report(enriched)
