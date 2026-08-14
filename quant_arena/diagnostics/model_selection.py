# =============================================================================
# FILE: quant_arena/diagnostics/model_selection.py
# Selección del número de estados K de un Gaussian HMM por AIC/BIC +
# validación cruzada temporal + diagnóstico real de convergencia (H9).
# =============================================================================
"""
`HMMGARCHStrategy` (zoo/estrategias/hmm_garch_strategy.py) usaba K=3 fijo,
sin ningún criterio de selección, y suprimía por completo el warning de
no-convergencia de Baum-Welch (`warnings.simplefilter("ignore")`) sin
verificar `monitor_.converged` — un HMM que no convergió puede seguir
produciendo un `predict()` que parece razonable pero con μ/σ por régimen mal
identificados.

Este módulo resuelve ambas cosas:
  - Barrido K ∈ k_range, seleccionando por AIC y BIC:
        BIC = -2·ln(L̂) + p·ln(n),   p = K² + 2K − 1
    (p = transición K(K−1) + distribución inicial (K−1) + medias K +
    varianzas K, emisiones gaussianas 1D — ver docstring del plan de diseño).
  - Validación cruzada temporal: ajusta en el primer tramo de la serie,
    evalúa log-verosimilitud OOS en el tramo siguiente (walk-forward, no
    k-fold — mezclar folds temporales rompería la causalidad).
  - `monitor_.converged` se LEE explícitamente por cada K y se reporta; el
    warning de hmmlearn se suprime solo a nivel de texto de consola, nunca
    la señal misma.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except ImportError:
    _HMM_OK = False

from quant_arena.core.excepciones import ConfiguracionInvalidaError, SupuestoEstadisticoError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultadoK:
    """Métricas de ajuste de un GaussianHMM con K estados."""
    k:                   int
    log_likelihood:       float
    n_parametros:         int
    aic:                  float
    bic:                  float
    convergio:            bool
    log_likelihood_oos:   Optional[float]  # None si no se pudo evaluar CV


@dataclass(frozen=True)
class ReporteSeleccionK:
    """Resultado completo del barrido de K."""
    resultados:      Tuple[ResultadoK, ...]
    k_optimo_bic:     int
    k_optimo_aic:     int
    k_optimo_cv:      Optional[int]  # None si ningún K tuvo CV válida


def n_parametros_hmm(k: int) -> int:
    """p = K² + 2K − 1 para un GaussianHMM 1D con covarianza diagonal/full escalar."""
    return k * k + 2 * k - 1


def seleccionar_k_hmm(
    log_rets: np.ndarray,
    k_range: Sequence[int] = range(2, 7),
    n_iter: int = 100,
    random_state: int = 42,
    covariance_type: str = "full",
    fraccion_train_cv: float = 0.7,
    tol: float = 1e-4,
) -> ReporteSeleccionK:
    """
    Ajusta un GaussianHMM para cada K en `k_range` y compara por AIC, BIC y
    log-verosimilitud OOS (validación cruzada temporal walk-forward).

    Args:
        log_rets:          array (T,) de log-retornos.
        k_range:            Valores de K a probar (todos >= 1).
        n_iter:             Iteraciones máximas de Baum-Welch por ajuste.
        random_state:       Semilla, compartida entre todos los K para
                            comparación justa.
        covariance_type:    Tipo de covarianza de emisión (ver hmmlearn).
        fraccion_train_cv:  Fracción inicial de la serie usada para entrenar
                            en la validación cruzada temporal; el resto es
                            el tramo OOS de evaluación. En (0, 1).
        tol:                Tolerancia de convergencia de Baum-Welch.

    Returns:
        ReporteSeleccionK con un ResultadoK por cada K que ajustó sin error,
        y los K óptimos según cada criterio.

    Raises:
        ConfiguracionInvalidaError: si `k_range` o `fraccion_train_cv` son
            inválidos, o si `hmmlearn` no está instalado.
        SupuestoEstadisticoError: si NINGÚN K del rango pudo ajustarse.
    """
    if not _HMM_OK:
        raise ConfiguracionInvalidaError("hmmlearn no está instalado (pip install hmmlearn).")
    if any(k < 1 for k in k_range):
        raise ConfiguracionInvalidaError(f"k_range={list(k_range)} contiene valores < 1.")
    if not (0.0 < fraccion_train_cv < 1.0):
        raise ConfiguracionInvalidaError(f"fraccion_train_cv={fraccion_train_cv} debe estar en (0, 1).")

    x = np.asarray(log_rets, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 30:
        raise SupuestoEstadisticoError(
            f"seleccionar_k_hmm requiere n>=30 log-retornos válidos, recibido n={len(x)}."
        )

    X = x.reshape(-1, 1)
    corte = int(len(x) * fraccion_train_cv)
    X_train, X_test = X[:corte], X[corte:]

    resultados: list[ResultadoK] = []
    for k in k_range:
        try:
            modelo = _ajustar_hmm(X, k, n_iter, random_state, covariance_type, tol)
        except Exception as exc:
            logger.warning(f"seleccionar_k_hmm: K={k} falló al ajustar ({exc}). Se omite.")
            continue

        convergio = bool(modelo.monitor_.converged)
        if not convergio:
            logger.warning(
                f"seleccionar_k_hmm: K={k} NO convergió en {n_iter} iteraciones "
                f"(tol={tol}) — el ajuste se reporta igual, pero μ/σ por régimen "
                "pueden no estar bien identificados."
            )

        log_lik = float(modelo.score(X))
        p = n_parametros_hmm(k)
        n = len(x)
        aic = -2.0 * log_lik + 2.0 * p
        bic = -2.0 * log_lik + p * np.log(n)

        log_lik_oos: Optional[float] = None
        if len(X_test) > k and len(X_train) > k:
            try:
                modelo_cv = _ajustar_hmm(X_train, k, n_iter, random_state, covariance_type, tol)
                log_lik_oos = float(modelo_cv.score(X_test))
            except Exception as exc:
                logger.warning(f"seleccionar_k_hmm: CV temporal de K={k} falló ({exc}).")

        resultados.append(ResultadoK(
            k=k, log_likelihood=log_lik, n_parametros=p, aic=aic, bic=bic,
            convergio=convergio, log_likelihood_oos=log_lik_oos,
        ))

    if not resultados:
        raise SupuestoEstadisticoError(
            f"Ningún K en {list(k_range)} pudo ajustarse sobre la serie dada."
        )

    k_optimo_bic = min(resultados, key=lambda r: r.bic).k
    k_optimo_aic = min(resultados, key=lambda r: r.aic).k
    validos_cv = [
        (r.k, r.log_likelihood_oos) for r in resultados if r.log_likelihood_oos is not None
    ]
    k_optimo_cv = max(validos_cv, key=lambda par: par[1])[0] if validos_cv else None

    return ReporteSeleccionK(
        resultados=tuple(resultados),
        k_optimo_bic=k_optimo_bic,
        k_optimo_aic=k_optimo_aic,
        k_optimo_cv=k_optimo_cv,
    )


def _ajustar_hmm(
    X: np.ndarray,
    k: int,
    n_iter: int,
    random_state: int,
    covariance_type: str,
    tol: float,
) -> "GaussianHMM":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # solo el texto de consola de hmmlearn
        modelo = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
        )
        modelo.fit(X)
    return modelo
