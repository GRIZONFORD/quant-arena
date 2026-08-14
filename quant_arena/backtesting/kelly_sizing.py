# =============================================================================
# FILE: quant_arena/backtesting/kelly_sizing.py
# Dimensionamiento de posición — Kelly bayesiano acoplado a la incertidumbre
# del posterior de habilidad del Juez TTT.
# =============================================================================
"""
Kelly clásico para retornos ~ N(μ, σ²) asigna la fracción óptima de capital

    f* = μ / σ²

pero trata μ como si fuera un parámetro CONOCIDO. En la práctica μ se estima
con error, y sobre-apostar por ese error de estimación es el fallo clásico
de Kelly (ver MacLean, Thorp & Ziemba, 2011, "The Kelly Capital Growth
Investment Criterion"). La corrección estándar en la literatura es penalizar
el denominador con la varianza del ERROR de estimación de μ.

Este módulo aprovecha que el Juez TTT ya produce esa varianza de estimación:
`TTTJuez.habilidades_latentes()` retorna `(mu, sigma)` del posterior de
habilidad de cada estrategia — `sigma` es exactamente la incertidumbre del
parámetro que Kelly clásico ignora. La fracción implementada es:

    f_kelly = λ · μ_edge / (σ²_retornos + κ · σ²_skill_TTT)

donde:
    μ_edge:        retorno esperado por período de la estrategia (edge real,
                    en unidades de retorno — NO la `mu` de TTT, que vive en
                    una escala de habilidad relativa adimensional).
    σ²_retornos:    varianza de los retornos realizados de la estrategia.
    σ²_skill_TTT:   varianza del posterior de habilidad TTT (`sigma` de
                    `habilidades_latentes()`). Penaliza automáticamente a las
                    estrategias sobre las que el Juez aún tiene poca
                    convicción (períodos tempranos, cambios de régimen).
    κ (kappa_skill): factor de recalibración que lleva σ²_skill_TTT —que vive
                    en la escala interna del modelo TTT (prior sigma≈1.6)— a
                    una magnitud comparable con σ²_retornos (que vive en
                    escala de retornos diarios, ~1e-4). Es un hiperparámetro
                    explícito, no una equivalencia teórica exacta: el valor
                    por defecto (1.0) es conservador porque en la escala
                    nativa de TTT una σ_skill típica (~1.0-1.6) domina el
                    denominador y fuerza f_kelly ≈ 0 salvo edge muy alto —
                    correcto direccionalmente (poca convicción → poca
                    exposición) aunque la calibración fina de κ requiere
                    ajuste empírico (ver docs/ttt_explicacion.md).
    λ (lam):        fracción de Kelly (Kelly fraccional). λ=1 (Kelly completo)
                    nunca es el default: la literatura documenta que Kelly
                    completo es óptimo solo asintóticamente y es frágil ante
                    error de estimación — exactamente el problema que este
                    módulo mitiga, pero no elimina. Default λ=0.5.

Referencia: Landfried (arXiv:2209.00092) para el posterior de habilidad TTT;
MacLean, Thorp & Ziemba (2011) para Kelly fraccional bajo incertidumbre.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from quant_arena.core.excepciones import ConfiguracionInvalidaError, SizingError
from quant_arena.core.interfaces_riesgo import AbstractPositionSizer


# =============================================================================
# KellyBayesianSizer — la pieza central de §1.2
# =============================================================================

@dataclass(frozen=True)
class KellyBayesianSizer(AbstractPositionSizer):
    """
    Kelly fraccional con penalización por incertidumbre del posterior TTT.

    Args:
        lam:            Fracción de Kelly aplicada, en (0, 1]. Default 0.5
                         (rango recomendado en la literatura: [0.25, 0.5]).
        kappa_skill:     Factor de recalibración de σ²_skill_TTT (ver docstring
                         del módulo). Debe ser >= 0; 0.0 desactiva la
                         penalización bayesiana (Kelly clásico puro).
        cap_individual: Exposición máxima por estrategia, en (0, ...].
                         Default 1.0 (sin apalancamiento por estrategia).
        cap_bruto:      Exposición bruta máxima sumada sobre todas las
                         estrategias, en (0, ...]. Default 1.0, coherente con
                         `RiskOverlay.max_leverage=1.0` (sin apalancamiento
                         agregado por defecto).
        piso:           Exposición mínima por estrategia (normalmente 0.0 —
                         Kelly no soporta posiciones cortas en este diseño).

    Raises:
        ConfiguracionInvalidaError: si algún parámetro está fuera de dominio.
    """

    lam:            float = 0.5
    kappa_skill:    float = 1.0
    cap_individual: float = 1.0
    cap_bruto:      float = 1.0
    piso:           float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.lam <= 1.0):
            raise ConfiguracionInvalidaError(
                f"lam={self.lam} fuera de (0, 1]. Kelly fraccional exige λ>0; "
                "λ=1 (Kelly completo) no está prohibido pero requiere pasarlo "
                "explícitamente a sabiendas de su fragilidad ante error de estimación."
            )
        if self.kappa_skill < 0.0:
            raise ConfiguracionInvalidaError(f"kappa_skill={self.kappa_skill} < 0.")
        if self.cap_individual <= 0.0:
            raise ConfiguracionInvalidaError(f"cap_individual={self.cap_individual} <= 0.")
        if self.cap_bruto <= 0.0:
            raise ConfiguracionInvalidaError(f"cap_bruto={self.cap_bruto} <= 0.")
        if self.piso < 0.0:
            raise ConfiguracionInvalidaError(f"piso={self.piso} < 0.")

    # ------------------------------------------------------------------
    # AbstractPositionSizer
    # ------------------------------------------------------------------

    def exposicion(
        self,
        mu_edge: float,
        sigma_retornos: float,
        sigma_skill: float = 0.0,
    ) -> float:
        """Fracción de Kelly para una única estrategia. Ver docstring del módulo."""
        if not np.isfinite(mu_edge):
            raise SizingError(f"mu_edge={mu_edge} no es finito.")
        if not np.isfinite(sigma_retornos) or sigma_retornos < 0.0:
            raise SizingError(f"sigma_retornos={sigma_retornos} inválido (debe ser >= 0 y finito).")
        if not np.isfinite(sigma_skill) or sigma_skill < 0.0:
            raise SizingError(f"sigma_skill={sigma_skill} inválido (debe ser >= 0 y finito).")

        varianza_total = sigma_retornos ** 2 + self.kappa_skill * sigma_skill ** 2
        if varianza_total <= 0.0:
            raise SizingError(
                "Varianza total nula: σ²_retornos + κ·σ²_skill = 0. "
                "Kelly no está definido sin dispersión (edge determinista)."
            )

        f_crudo = self.lam * mu_edge / varianza_total
        return float(np.clip(f_crudo, self.piso, self.cap_individual))

    # ------------------------------------------------------------------
    # Conveniencia: dimensionar todo el Zoo de una vez
    # ------------------------------------------------------------------

    def pesos_exposicion(
        self,
        mu_edge: Mapping[str, float],
        sigma_retornos: Mapping[str, float],
        sigma_skill: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Traduce (μ_edge, σ_retornos, σ_skill) por estrategia en exposiciones
        absolutas, respetando `cap_bruto` sobre la suma.

        A diferencia de `TTTJuez.pesos_asignacion()` (que reparte pesos que
        SIEMPRE suman 1), esta exposición puede sumar MENOS de 1: si el Juez
        no tiene convicción en ninguna estrategia (σ_skill alto en todas),
        la exposición total baja sola en vez de invertir el 100% igual.

        Estrategias sin datos suficientes (SizingError individual) reciben
        exposición 0.0 en vez de abortar el cálculo para el resto del Zoo.

        Returns:
            {nombre_estrategia: exposicion} con suma <= cap_bruto.
        """
        crudo: Dict[str, float] = {}
        for nombre, mu in mu_edge.items():
            try:
                crudo[nombre] = self.exposicion(
                    mu_edge=mu,
                    sigma_retornos=sigma_retornos.get(nombre, np.nan),
                    sigma_skill=sigma_skill.get(nombre, 0.0),
                )
            except SizingError:
                crudo[nombre] = 0.0

        bruto = sum(crudo.values())
        if bruto > self.cap_bruto:
            factor = self.cap_bruto / bruto
            crudo = {k: v * factor for k, v in crudo.items()}

        return crudo


# =============================================================================
# Sizers adicionales — completan la jerarquía Strategy (OCP/ISP)
# =============================================================================

@dataclass(frozen=True)
class FixedFractionSizer(AbstractPositionSizer):
    """
    Sizer trivial: expone siempre la misma fracción de capital, ignorando
    edge y varianza. Útil como control/baseline en tests A/B contra Kelly.
    """
    fraccion: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.fraccion <= 1.0):
            raise ConfiguracionInvalidaError(f"fraccion={self.fraccion} fuera de [0, 1].")

    def exposicion(
        self,
        mu_edge: float,
        sigma_retornos: float,
        sigma_skill: float = 0.0,
    ) -> float:
        return self.fraccion


@dataclass(frozen=True)
class VolTargetSizer(AbstractPositionSizer):
    """
    Sizer de volatilidad objetivo: f = clip(target_vol / sigma_retornos, 0, cap).
    Ignora μ_edge y σ_skill por diseño (dimensiona por riesgo, no por señal) —
    equivalente al mecanismo ya usado dentro de `RiskOverlay.compute_risk_weight`,
    expuesto aquí como sizer intercambiable para comparación directa con Kelly.
    """
    target_vol:   float = 0.15
    cap:          float = 1.0
    trading_days: int   = 252

    def __post_init__(self) -> None:
        if self.target_vol <= 0.0:
            raise ConfiguracionInvalidaError(f"target_vol={self.target_vol} <= 0.")
        if self.cap <= 0.0:
            raise ConfiguracionInvalidaError(f"cap={self.cap} <= 0.")
        if self.trading_days <= 0:
            raise ConfiguracionInvalidaError(f"trading_days={self.trading_days} <= 0.")

    def exposicion(
        self,
        mu_edge: float,
        sigma_retornos: float,
        sigma_skill: float = 0.0,
    ) -> float:
        if not np.isfinite(sigma_retornos) or sigma_retornos <= 0.0:
            raise SizingError(f"sigma_retornos={sigma_retornos} inválido para VolTargetSizer.")
        vol_anualizada = sigma_retornos * np.sqrt(self.trading_days)
        f_crudo = self.target_vol / vol_anualizada
        return float(np.clip(f_crudo, 0.0, self.cap))
