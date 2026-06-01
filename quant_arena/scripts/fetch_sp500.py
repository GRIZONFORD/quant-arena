#!/usr/bin/env python3
"""
scripts/fetch_sp500.py
Descarga datos históricos del S&P 500 (^GSPC) desde 1997 hasta hoy
y los guarda en data/sp500_daily_1997_to_today.csv.
"""
from pathlib import Path
import yfinance as yf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "sp500_daily_1997_to_today.parquet"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

raw = yf.download("^GSPC", start="1997-01-01", progress=False)

# Aplanar MultiIndex de columnas (yfinance >= 0.2.x / 1.x)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.columns.name = None

# Conservar solo las columnas requeridas (las que existan)
target_cols = ["Open", "High", "Low", "Close", "Volume"]
df = raw[[c for c in target_cols if c in raw.columns]].copy()
df.index.name = "Date"
df = df.dropna(how="all")

df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy")
print(f"[OK] {OUTPUT_PATH.name}  —  {len(df):,} filas x {len(df.columns)} columnas")
