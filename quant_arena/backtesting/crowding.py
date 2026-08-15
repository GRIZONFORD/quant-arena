# =============================================================================
# FILE: quant_arena/backtesting/crowding.py
# Arena real: acoplamiento entre estrategias por capacidad y crowding (§1.4)
# =============================================================================
"""
Resuelve H3: hasta aquí, el ranking de TTT decide CUÁNTO capital recibe
cada estrategia, pero las estrategias nunca se afectan entre sí — no hay
ningún mecanismo por el que invertir mucho capital en una estrategia
degrade su propio rendimiento futuro. Eso es lo que separa un "ranking" de
una "arena": en una arena real, competir por el mismo trade tiene un costo.

Mecanismo (capacidad / impacto de mercado, Berk & Green 2004):

    alpha_efectivo_i(t) = alpha_bruto_i(t) · exp(-κ · participación_i(t))
    participación_i(t)  = capital_dirigido_i(t) / ADV(ticker)

A mayor fracción del volumen diario que una estrategia intenta mover, mayor
la erosión de su ventaja — rendimientos decrecientes a escala. Esto por sí
solo YA acopla capital a alpha, pero sigue siendo un efecto "por estrategia
aislada": no captura que DOS estrategias distintas, cada una con baja
participación individual, pueden generar un impacto de mercado conjunto
severo si convergen en el mismo ticker al mismo tiempo — el crowding real
de una arena con múltiples participantes.

`ArenaCrowding.demanda_agregada_por_ticker` agrega la exposición de TODAS
las estrategias por ticker antes de aplicar el decaimiento, y
`ArenaCrowding.solapamiento` cuantifica cuánto se pisan las posiciones de
cada par de estrategias — la métrica de crowding que un evaluador esperaría
ver reportada, no solo el mecanismo de decaimiento.

Cierra el lazo de realimentación que le da sentido al nombre "arena": TTT
asigna capital → el alpha efectivo decae para quien está crowded → el KPI
del siguiente período refleja esa erosión → TTT reasigna. Emerge un
equilibrio entre estrategias en vez de una ganadora única concentrando todo
el capital — la respuesta directa a "¿por qué no invertir el 100% en la
mejor estrategia del ranking?".

LIMITACIÓN DOCUMENTADA: `adv_por_ticker` es un snapshot estático pasado en
la construcción, no una serie de ADV rodante actualizada por rebalanceo.
Es la simplificación deliberada de este primer corte — el mecanismo de
decaimiento y la métrica de solapamiento son el resultado central; una ADV
dinámica (rolling mean de Volume·Close) es una extensión directa que no
cambia la interfaz pública de este módulo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from quant_arena.core.excepciones import ConfiguracionInvalidaError


# =============================================================================
# CrowdingModel — decaimiento de alpha por capacidad (mecanismo puro)
# =============================================================================

@dataclass(frozen=True)
class CrowdingModel:
    """
    Decaimiento exponencial de alpha por participación en el volumen diario.

    Args:
        kappa: Sensibilidad del decaimiento a la participación. kappa=0
               desactiva el efecto (factor siempre 1.0). Default 1.0.

    Raises:
        ConfiguracionInvalidaError: si kappa < 0.
    """
    kappa: float = 1.0

    def __post_init__(self) -> None:
        if self.kappa < 0.0:
            raise ConfiguracionInvalidaError(f"kappa={self.kappa} debe ser >= 0.")

    def factor_decaimiento(self, participacion: float) -> float:
        """
        exp(-kappa * participación), acotado a (0, 1].

        Args:
            participacion: |capital_dirigido| / ADV, >= 0. Valores no
                           finitos o negativos se tratan como 0 (sin
                           penalización) — un ADV desconocido no debe
                           inventar una penalización arbitraria.
        """
        if not np.isfinite(participacion) or participacion < 0.0:
            participacion = 0.0
        return float(np.exp(-self.kappa * participacion))

    def alpha_efectivo(self, alpha_bruto: float, capital_dirigido: float, adv: float) -> float:
        """
        alpha_bruto · factor_decaimiento(|capital_dirigido| / ADV).

        Si `adv` no es válido (<=0, NaN, inf), retorna `alpha_bruto` sin
        penalizar: sin una referencia de liquidez no hay base para estimar
        impacto de mercado, y penalizar arbitrariamente sería peor que no
        penalizar.
        """
        if not np.isfinite(adv) or adv <= 0.0:
            return alpha_bruto
        participacion = abs(capital_dirigido) / adv
        return alpha_bruto * self.factor_decaimiento(participacion)


# =============================================================================
# ArenaCrowding — orquesta la interacción entre estrategias
# =============================================================================

class ArenaCrowding:
    """
    Mide el solapamiento de posiciones entre estrategias y aplica el
    decaimiento de `CrowdingModel` a la exposición de cada una, en función
    de la DEMANDA AGREGADA de capital que toda la arena dirige a cada
    ticker — no solo la participación individual de cada estrategia.

    Args:
        modelo:        `CrowdingModel` con el κ de sensibilidad al impacto.
        adv_por_ticker: {ticker: ADV en unidades monetarias}. Snapshot
                       estático (ver limitación documentada en el módulo).
    """

    def __init__(self, modelo: CrowdingModel, adv_por_ticker: Mapping[str, float]) -> None:
        self._modelo = modelo
        self._adv = dict(adv_por_ticker)

    # ------------------------------------------------------------------
    # Métrica de solapamiento — lo que un evaluador esperaría ver reportado
    # ------------------------------------------------------------------

    def solapamiento(self, pesos_por_estrategia: Mapping[str, pd.Series]) -> pd.DataFrame:
        """
        Matriz [estrategia x estrategia] de solapamiento de posiciones:
        similitud coseno entre los vectores de peso (indexados por ticker)
        de cada par de estrategias, sobre la unión de sus tickers.

        1.0 = posiciones idénticas (mismo riesgo de crowding mutuo);
        0.0 = universos disjuntos o vectores ortogonales; la diagonal es
        siempre 1.0 (una estrategia solapa completamente consigo misma).

        Returns:
            DataFrame simétrico [estrategia x estrategia].
        """
        nombres = list(pesos_por_estrategia.keys())
        n = len(nombres)
        matriz = pd.DataFrame(
            np.eye(n), index=nombres, columns=nombres, dtype=float
        )
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._similitud_coseno(
                    pesos_por_estrategia[nombres[i]], pesos_por_estrategia[nombres[j]]
                )
                matriz.loc[nombres[i], nombres[j]] = sim
                matriz.loc[nombres[j], nombres[i]] = sim
        return matriz

    @staticmethod
    def _similitud_coseno(a: pd.Series, b: pd.Series) -> float:
        tickers = a.index.union(b.index)
        va = a.reindex(tickers).fillna(0.0).to_numpy()
        vb = b.reindex(tickers).fillna(0.0).to_numpy()
        norma_a = float(np.linalg.norm(va))
        norma_b = float(np.linalg.norm(vb))
        if norma_a <= 0.0 or norma_b <= 0.0:
            return 0.0
        return float(np.dot(va, vb) / (norma_a * norma_b))

    # ------------------------------------------------------------------
    # Demanda agregada — el acoplamiento real entre estrategias
    # ------------------------------------------------------------------

    def demanda_agregada_por_ticker(
        self,
        pesos_por_estrategia: Mapping[str, pd.Series],
        capital_por_estrategia: Mapping[str, float],
        aum_total: float,
    ) -> pd.Series:
        """
        Capital absoluto que TODAS las estrategias, combinadas, dirigen a
        cada ticker en este rebalanceo:

            demanda(ticker) = Σ_i |peso_i(ticker)| · capital_i(t) · AUM_total

        Dos estrategias con baja participación individual pueden generar,
        combinadas, una demanda agregada que sí perfora el ADV de un
        ticker — esto es lo que distingue crowding real de un simple
        límite de capacidad por estrategia aislada.

        Args:
            pesos_por_estrategia:  {estrategia: pesos [ticker -> peso]}.
            capital_por_estrategia: {estrategia: fracción de AUM asignada
                                    por el Juez/sizer, ej. `pesos_juez`}.
            aum_total:              AUM total del meta-portafolio, en
                                    unidades monetarias consistentes con
                                    `adv_por_ticker`.

        Returns:
            pd.Series [ticker -> demanda agregada].
        """
        acumulado: Dict[str, float] = {}
        for nombre, pesos in pesos_por_estrategia.items():
            capital_estrategia = capital_por_estrategia.get(nombre, 0.0) * aum_total
            if capital_estrategia == 0.0:
                continue
            for ticker, peso in pesos.items():
                ticker_str = str(ticker)
                acumulado[ticker_str] = acumulado.get(ticker_str, 0.0) + abs(peso) * capital_estrategia
        return pd.Series(acumulado, dtype=float)

    def factores_decaimiento(
        self,
        pesos_por_estrategia: Mapping[str, pd.Series],
        capital_por_estrategia: Mapping[str, float],
        aum_total: float,
    ) -> Dict[str, float]:
        """
        Factor de decaimiento de alpha por estrategia — cuánto de la
        demanda AGREGADA de la arena (no solo la propia) recae sobre los
        tickers donde esa estrategia está posicionada, ponderado por su
        propia distribución de pesos.

        Returns:
            {estrategia: factor en (0, 1]}. 1.0 si la estrategia no tiene
            peso significativo (nada que penalizar).
        """
        demanda = self.demanda_agregada_por_ticker(
            pesos_por_estrategia, capital_por_estrategia, aum_total
        )

        factores: Dict[str, float] = {}
        for nombre, pesos in pesos_por_estrategia.items():
            pesos_abs = pesos.abs()
            total_pesos = float(pesos_abs.sum())
            if total_pesos <= 0.0:
                factores[nombre] = 1.0
                continue

            factor_ponderado = 0.0
            for ticker, peso in pesos.items():
                ticker_str = str(ticker)
                peso_abs = abs(peso)
                if peso_abs <= 0.0:
                    continue
                adv = self._adv.get(ticker_str, float("nan"))
                demanda_ticker = float(demanda.get(ticker_str, 0.0))
                factor_ticker = self._modelo.alpha_efectivo(1.0, demanda_ticker, adv)
                factor_ponderado += (peso_abs / total_pesos) * factor_ticker

            factores[nombre] = float(np.clip(factor_ponderado, 0.0, 1.0))

        return factores
