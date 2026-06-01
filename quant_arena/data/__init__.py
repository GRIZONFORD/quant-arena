"""
Capa de datos de `quant_arena`.

Contiene los datasets (`*.parquet`) y los gestores de datos:
  · PanelDataManager — panel cross-sectional MultiIndex [date, ticker] (Tier 1).
"""
from quant_arena.data.panel_data_manager import PanelDataManager

__all__ = ["PanelDataManager"]
