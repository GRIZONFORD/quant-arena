# =============================================================================
# FILE: quant_arena/core/excepciones.py
# Jerarquía de excepciones propias — reemplaza `except Exception` desnudo en
# los bordes donde el motor degrada con continuidad (ver motor.py, patrón ya
# existente en el manejo de TTTJuez.actualizar()).
# =============================================================================
from __future__ import annotations


class QuantArenaError(Exception):
    """Raíz de la jerarquía. Permite un único `except QuantArenaError` en el borde."""


class ConfiguracionInvalidaError(QuantArenaError):
    """Un parámetro de configuración está fuera de su dominio válido (ej. λ>1, dd_limit<0)."""


class CausalidadVioladaError(QuantArenaError):
    """Un componente causal (ej. TTTJuez en modo_causal_estricto) recibió datos del futuro."""


class SizingError(QuantArenaError):
    """El dimensionamiento de posición no puede calcularse (ej. Kelly con σ²=0 o μ no finito)."""


class SupuestoEstadisticoError(QuantArenaError):
    """Se rechazó H0 de un supuesto estadístico que el modelo exige y no hay acción definida."""
