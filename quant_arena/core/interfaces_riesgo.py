# =============================================================================
# FILE: quant_arena/core/interfaces_riesgo.py
# Interfaces segregadas (ISP) para dimensionamiento de posición y reglas de
# riesgo. Extiende el patrón ya usado en abstracciones.py (AbstractJuez,
# AbstractMetricas): el Motor depende de estas abstracciones, no de
# implementaciones concretas (DIP).
# =============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class AbstractPositionSizer(ABC):
    """
    Contrato para estrategias de dimensionamiento de posición (patrón Strategy).

    Un sizer traduce evidencia estadística (edge, varianza) en una fracción de
    capital a exponer. No conoce reglas de riesgo (stops, drawdown) — esa es
    la responsabilidad de `AbstractRiskRule`. Mantener ambas interfaces
    separadas es lo que permite añadir un sizer nuevo (OCP) sin tocar el
    motor ni obligar a implementar métodos de riesgo que no le corresponden
    (ISP).
    """

    @abstractmethod
    def exposicion(
        self,
        mu_edge: float,
        sigma_retornos: float,
        sigma_skill: float = 0.0,
    ) -> float:
        """
        Calcula la fracción de capital a exponer para una única estrategia.

        Args:
            mu_edge:        Retorno esperado (edge) por período, en unidades
                             de retorno simple (ej. 0.001 = 10 bps/período).
            sigma_retornos:  Desviación estándar de los retornos por período.
            sigma_skill:     Incertidumbre adicional del parámetro de habilidad
                              (ej. σ del posterior TTT). 0.0 si no aplica.

        Returns:
            Fracción de capital en [piso, cap] (ver implementación concreta).

        Raises:
            SizingError: si la entrada no permite un cálculo bien definido
                         (μ no finito, varianza total no positiva).
        """
        ...


class AbstractRiskRule(ABC):
    """
    Contrato para una regla de riesgo componible (Composite / Chain of
    Responsibility). Cada regla recibe la exposición vigente y un contexto
    de mercado, y retorna la exposición ajustada. Las reglas se apilan en un
    `RiskPipeline` en vez de heredar unas de otras (composición sobre
    herencia); cada una tiene una única responsabilidad (SRP).
    """

    @abstractmethod
    def aplicar(
        self,
        exposicion: pd.Series,
        contexto: Dict[str, pd.Series],
    ) -> pd.Series:
        """
        Ajusta la serie de exposición según la regla.

        Args:
            exposicion: Serie diaria de exposición fraccional vigente,
                        indexada por fecha (salida de la regla anterior en
                        el pipeline, o la exposición cruda del sizer).
            contexto:   Series auxiliares indexadas por la misma fecha
                        (ej. {'price_ret': ..., 'atr': ..., 'vol_ann': ...}).
                        Cada regla usa solo las claves que necesita.

        Returns:
            Serie de exposición ajustada, mismo índice que `exposicion`.
        """
        ...
