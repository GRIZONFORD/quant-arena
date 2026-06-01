# =============================================================================
# FILE: quant_arena/core/abstracciones.py
# Interfaces base del sistema — Principio de Inversión de Dependencias (DIP)
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricasResultado:
    """
    Contenedor tipado e inmutable para KPIs institucionales de una estrategia.
    Actúa como DTO (Data Transfer Object) entre Métricas y Juez.
    """
    sharpe:             float = np.nan
    sortino:            float = np.nan
    max_drawdown:       float = np.nan
    calmar:             float = np.nan
    information_ratio:  float = np.nan
    alpha_tstat:        float = np.nan
    turnover:           float = np.nan

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def es_valido(self, campo: str = 'sharpe') -> bool:
        """Verifica que el campo principal no sea NaN o Inf."""
        valor = getattr(self, campo, np.nan)
        return bool(np.isfinite(valor))


# -----------------------------------------------------------------------------
# Interfaz de Métricas
# -----------------------------------------------------------------------------
class AbstractMetricas(ABC):
    """Contrato para calculadoras de métricas de performance institucional."""

    @abstractmethod
    def sharpe_ratio(self, retornos: pd.Series) -> float: ...

    @abstractmethod
    def sortino_ratio(self, retornos: pd.Series) -> float: ...

    @abstractmethod
    def max_drawdown(self, retornos: pd.Series) -> float: ...

    @abstractmethod
    def calmar_ratio(self, retornos: pd.Series) -> float: ...

    @abstractmethod
    def information_ratio(
        self, retornos: pd.Series, benchmark: pd.Series
    ) -> float: ...

    @abstractmethod
    def alpha_tstat(
        self, retornos: pd.Series, benchmark: pd.Series
    ) -> float: ...

    @abstractmethod
    def turnover(self, pesos: pd.DataFrame) -> float: ...

    @abstractmethod
    def calcular_todas(
        self,
        retornos: pd.Series,
        benchmark: pd.Series,
        pesos: Optional[pd.DataFrame] = None,
    ) -> MetricasResultado: ...


# -----------------------------------------------------------------------------
# Interfaz de Estrategia (Zoo)
# -----------------------------------------------------------------------------
class AbstractStrategy(ABC):
    """
    Interfaz base para todas las sub-estrategias del Zoo.

    El Motor de Backtesting y el Juez dependen únicamente de esta abstracción;
    las implementaciones concretas (Momentum, Value, Carry, etc.) son intercambiables
    sin modificar la infraestructura central.
    """

    def __init__(self, nombre: str, universo: List[str]) -> None:
        self._nombre = nombre
        self._universo = universo

    @property
    def nombre(self) -> str:
        return self._nombre

    @property
    def universo(self) -> List[str]:
        return self._universo

    @property
    @abstractmethod
    def descripcion(self) -> str:
        """Una línea que describe la lógica de la estrategia."""
        ...

    @abstractmethod
    def generar_señales(
        self,
        datos: pd.DataFrame,
        fecha_corte: pd.Timestamp,
    ) -> pd.Series:
        """
        Genera pesos por activo estrictamente causales (solo datos <= fecha_corte).

        Args:
            datos:       DataFrame con precios/features indexado por fecha.
            fecha_corte: Límite temporal; no se pueden usar datos posteriores.

        Returns:
            pd.Series indexada por ticker con pesos (suma <= 1.0 en long-only).
        """
        ...

    @abstractmethod
    def calcular_retornos(
        self,
        datos: pd.DataFrame,
        pesos_historicos: pd.DataFrame,
    ) -> pd.Series:
        """
        Aplica pesos históricos pre-computados y retorna retornos diarios netos.

        Args:
            datos:             Precios históricos.
            pesos_historicos:  DataFrame [fecha x ticker] con pesos diarios.

        Returns:
            pd.Series de retornos diarios de la estrategia.
        """
        ...


# -----------------------------------------------------------------------------
# Interfaz del Juez
# -----------------------------------------------------------------------------
class AbstractJuez(ABC):
    """
    Contrato del Juez algorítmico de habilidad latente.

    Invariante: nunca accede a datos futuros al período de evaluación.
    """

    @abstractmethod
    def registrar_periodo(
        self,
        metricas_periodo: Dict[str, MetricasResultado],
        tiempo: float,
        metrica_ranking: str = 'sharpe',
    ) -> None:
        """
        Registra el resultado comparativo de un período de evaluación.

        Args:
            metricas_periodo: {nombre_estrategia -> MetricasResultado}
            tiempo:           Días desde época Unix (para escala temporal de TTT).
            metrica_ranking:  Campo de MetricasResultado usado para ordenar estrategias.
        """
        ...

    @abstractmethod
    def actualizar(self) -> None:
        """Ejecuta la inferencia / propagación de mensajes sobre el historial."""
        ...

    @abstractmethod
    def habilidades_latentes(self) -> Dict[str, Tuple[float, float]]:
        """Retorna {nombre_estrategia: (mu, sigma)} del estado posterior más reciente."""
        ...

    @abstractmethod
    def pesos_asignacion(self, metodo: str = 'mu_sobre_sigma') -> Dict[str, float]:
        """
        Convierte habilidades latentes en pesos de asignación de capital.
        Garantiza: sum(pesos.values()) == 1.0, pesos[i] >= 0.
        """
        ...
