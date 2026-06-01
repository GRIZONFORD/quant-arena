# =============================================================================
# FILE: quant_arena/zoo/base_estrategia.py
# Registro global del Zoo + ZooManager para descubrimiento dinámico de estrategias
# =============================================================================
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Type

import numpy as np
import pandas as pd

from quant_arena.core.abstracciones import AbstractStrategy


# =============================================================================
# Utilidades de señales compartidas por todas las estrategias
# =============================================================================

def normalizar_pesos(
    señales: pd.Series,
    long_only: bool = True,
    max_peso: Optional[float] = None,
) -> pd.Series:
    """
    Normaliza señales crudas a pesos de portafolio válidos.

    En modo long_only: clip a 0, divide por la suma → pesos no-negativos
    que suman 1.0 sobre activos seleccionados.
    En modo long-short: divide por la suma de valores absolutos → el
    portafolio tiene exposición bruta = 1.0 y puede ser neto-neutral.

    Args:
        señales:   pd.Series con puntuaciones crudas por activo (ticker → score).
        long_only: Si True, descarta señales negativas antes de normalizar.
        max_peso:  Límite máximo por activo (ej. 0.10 = máx 10%). Si None, sin límite.

    Returns:
        pd.Series con los mismos índices, pesos en [0, 1] (long_only) o [-1, 1].
        Retorna pesos cero uniformes si la suma neta es 0.
    """
    s = señales.copy()
    if long_only:
        s = s.clip(lower=0.0)

    denominador: float = s.sum() if long_only else s.abs().sum()

    if denominador == 0.0:
        return pd.Series(0.0, index=s.index, dtype=float)

    pesos = s / denominador

    if max_peso is not None and max_peso < 1.0:
        # Truncamiento iterativo: recalcula hasta que ningún peso supere el límite
        for _ in range(50):
            exceso = (pesos - max_peso).clip(lower=0.0)
            if exceso.sum() < 1e-10:
                break
            pesos = (pesos - exceso).clip(lower=0.0)
            total = pesos.sum()
            if total > 0:
                pesos /= total

    return pesos


def señales_a_binario(
    scores: pd.Series,
    top_n: int,
    ascending: bool = False,
) -> pd.Series:
    """
    Selecciona los top N activos por score y retorna una máscara binaria.

    Vectorizado: usa pd.Series.rank() en lugar de sort + slice.

    Args:
        scores:    Puntuaciones por activo.
        top_n:     Número de activos a seleccionar.
        ascending: Si True, selecciona los N de menor score.

    Returns:
        pd.Series bool-like (0.0/1.0) con 1 en las top N posiciones.
    """
    ranking = scores.rank(ascending=ascending, method='first')
    return (ranking <= top_n).astype(float)


# =============================================================================
# Registro global del Zoo
# =============================================================================

class RegistroZoo:
    """
    Registro singleton de clases de estrategias disponibles en el Zoo.

    El Motor de Backtesting usa este registro para descubrimiento dinámico:
    no necesita importar directamente ninguna estrategia concreta.
    """

    _registro: ClassVar[Dict[str, Type[AbstractStrategy]]] = {}

    @classmethod
    def registrar(cls, nombre: str):
        """
        Decorador de clase. Registra la estrategia bajo `nombre`.

        Uso:
            @RegistroZoo.registrar('mi_estrategia')
            class MiEstrategia(AbstractStrategy): ...
        """
        def decorador(klass: Type[AbstractStrategy]) -> Type[AbstractStrategy]:
            if nombre in cls._registro:
                raise ValueError(
                    f"Ya existe una estrategia registrada con el nombre '{nombre}'. "
                    f"Usa un nombre único."
                )
            cls._registro[nombre] = klass
            return klass
        return decorador

    @classmethod
    def obtener_clase(cls, nombre: str) -> Type[AbstractStrategy]:
        """Retorna la clase registrada bajo `nombre`. Lanza KeyError si no existe."""
        if nombre not in cls._registro:
            raise KeyError(
                f"Estrategia '{nombre}' no registrada. "
                f"Disponibles: {sorted(cls._registro)}"
            )
        return cls._registro[nombre]

    @classmethod
    def listar(cls) -> List[str]:
        """Lista de nombres registrados ordenados alfabéticamente."""
        return sorted(cls._registro)

    @classmethod
    def listar_con_descripcion(cls) -> Dict[str, str]:
        """
        Retorna {nombre: descripcion} para todas las estrategias registradas.
        Útil para reportes e interfaces de usuario.
        """
        out: Dict[str, str] = {}
        for nombre, klass in cls._registro.items():
            # Intenta instanciar con universo vacío para leer .descripcion
            try:
                instancia = klass(universo=[])
                out[nombre] = instancia.descripcion
            except Exception:
                out[nombre] = klass.__doc__ or "(sin descripción)"
        return out

    @classmethod
    def autodescubrir(cls, paquete: str = 'quant_arena.zoo.estrategias') -> None:
        """
        Importa automáticamente todos los módulos de `paquete` para que sus
        decoradores @RegistroZoo.registrar se ejecuten y registren las clases.

        El Motor llama a este método al inicio para poblar el registro sin
        necesidad de imports explícitos en __init__.py del Zoo.
        """
        try:
            paquete_mod = importlib.import_module(paquete)
        except ModuleNotFoundError:
            return

        paquete_path = Path(paquete_mod.__file__).parent  # type: ignore[arg-type]
        for info in pkgutil.iter_modules([str(paquete_path)]):
            importlib.import_module(f"{paquete}.{info.name}")


# =============================================================================
# ZooManager — gestiona instancias activas del Zoo
# =============================================================================

@dataclass
class ConfigEstrategia:
    """
    Configuración de instanciación para una estrategia del Zoo.

    Permite al Motor crear instancias con parámetros específicos de forma
    declarativa, sin hard-code de constructores.
    """
    nombre_registro: str
    universo: List[str]
    kwargs: Dict = field(default_factory=dict)

    def instanciar(self) -> AbstractStrategy:
        """Crea y retorna una instancia configurada de la estrategia."""
        klass = RegistroZoo.obtener_clase(self.nombre_registro)
        return klass(universo=self.universo, **self.kwargs)


class ZooManager:
    """
    Gestiona el conjunto de instancias activas de estrategias del Zoo.

    Responsabilidades:
        - Instanciar estrategias desde ConfigEstrategia.
        - Generar señales de todas las estrategias de forma unificada.
        - Calcular retornos históricos del conjunto Zoo para el Motor.

    El Motor de Backtesting usa ZooManager como fachada; nunca interactúa
    directamente con AbstractStrategy concretas.
    """

    def __init__(self) -> None:
        self._estrategias: Dict[str, AbstractStrategy] = {}

    def agregar(self, estrategia: AbstractStrategy) -> None:
        """Registra una instancia activa. Sobreescribe si el nombre ya existe."""
        self._estrategias[estrategia.nombre] = estrategia

    def agregar_desde_config(self, config: ConfigEstrategia) -> None:
        """Crea la instancia y la agrega al manager."""
        self.agregar(config.instanciar())

    def agregar_desde_configs(self, configs: List[ConfigEstrategia]) -> None:
        """Procesa una lista de configuraciones."""
        for cfg in configs:
            self.agregar_desde_config(cfg)

    def obtener(self, nombre: str) -> AbstractStrategy:
        if nombre not in self._estrategias:
            raise KeyError(f"'{nombre}' no está activo en este ZooManager.")
        return self._estrategias[nombre]

    def nombres(self) -> List[str]:
        return list(self._estrategias.keys())

    def __len__(self) -> int:
        return len(self._estrategias)

    def __iter__(self) -> Iterator[AbstractStrategy]:
        return iter(self._estrategias.values())

    def generar_señales_todas(
        self,
        datos: pd.DataFrame,
        fecha_corte: pd.Timestamp,
    ) -> Dict[str, pd.Series]:
        """
        Genera señales de todas las estrategias activas para una fecha dada.

        Garantía causal: cada estrategia recibe solo datos <= fecha_corte.
        La llamada a generar_señales de cada estrategia ya tiene esta garantía
        contractual (AbstractStrategy), pero ZooManager la refuerza filtrando
        el DataFrame antes de delegarlo.

        Returns:
            {nombre_estrategia: pd.Series de pesos por activo}
        """
        datos_causal = datos.loc[datos.index <= fecha_corte]
        return {
            nombre: estrategia.generar_señales(datos_causal, fecha_corte)
            for nombre, estrategia in self._estrategias.items()
        }

    def calcular_retornos_todos(
        self,
        datos: pd.DataFrame,
        pesos_por_estrategia: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """
        Calcula series de retornos históricos para todas las estrategias activas.

        Args:
            datos:                  Precios históricos completos.
            pesos_por_estrategia:   {nombre: DataFrame[fecha x ticker]} con
                                    pesos diarios precalculados por el Motor.

        Returns:
            {nombre_estrategia: pd.Series de retornos diarios}
        """
        return {
            nombre: estrategia.calcular_retornos(datos, pesos_por_estrategia[nombre])
            for nombre, estrategia in self._estrategias.items()
            if nombre in pesos_por_estrategia
        }
