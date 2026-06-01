# =============================================================================
# FILE: quant_arena/calibracion/optimizador.py
# Calibración Empírica de Bayes de los hiperparámetros del modelo TTT.
#
# Principio matemático:
#   Dado un historial de competencias D = {composition, times}, buscamos:
#
#       (σ*, γ*) = argmax_{σ,γ>0} log p(D | σ, γ)
#
#   donde log p(D | σ, γ) es la log-evidencia marginal del grafo factorial de
#   TrueSkill Through Time, aproximada vía Expectation Propagation (EP).
#
# Estabilidad numérica — optimización en espacio logarítmico:
#   Paramétrizamos θ = (log σ, log γ) ∈ ℝ² y pasamos exp(θ) al modelo.
#   Esto:
#     1. Garantiza σ, γ > 0 sin necesidad de restricciones de desigualdad.
#     2. Convierte el espacio de búsqueda semi-infinito (0, ∞)² en ℝ²,
#        donde los métodos de gradiente son más estables.
#     3. Evita que L-BFGS-B evalúe puntos donde la matriz de covarianza
#        del proceso de Wiener colapsa (σ → 0 o γ → 0).
#
# Uso canónico:
#   composition, times = juez.exportar_historial()
#   opt = OptimizadorTTT()
#   resultado = opt.calibrar(composition, times)
#   juez_calibrado = TTTJuez(sigma=resultado['sigma_optimo'],
#                             gamma=resultado['gamma_optimo'])
# =============================================================================
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from trueskillthroughtime import History  # type: ignore[import]


class OptimizadorTTT:
    """
    Calibrador Empírico de Bayes para los hiperparámetros σ y γ del modelo TTT.

    Maximiza la log-evidencia marginal ``log p(D | σ, γ)`` usando
    ``scipy.optimize.minimize`` con el método ``L-BFGS-B`` en el espacio
    logarítmico de los parámetros.

    El optimizador es **stateless respecto al BacktestEngine**: recibe los
    datos de competencia pre-construidos y devuelve un diccionario de
    parámetros.  No modifica ningún objeto externo.

    Args:
        sigma_inicial:  Valor inicial de σ (prior de habilidad). Default: 1.6
                        (calibrado en datos ATP, Landfried 2024).
        gamma_inicial:  Valor inicial de γ (drift de Wiener). Default: 0.036.
        epsilon_ep:     Tolerancia EP durante la optimización. Valor suelto
                        (0.1) para reducir el costo por evaluación de función.
        max_iter_ep:    Iteraciones EP máximas por evaluación. Default: 5.
        max_iter_opt:   Iteraciones máximas de L-BFGS-B. Default: 150.
        p_draw:         Probabilidad de empate (debe coincidir con TTTJuez).
    """

    # Sentinela para evaluaciones de función que fallaron.
    # scipy recibirá +1e9 como objetivo (muy malo), descartando el punto.
    _LOG_EV_FALLBACK: float = -1.0e9

    def __init__(
        self,
        sigma_inicial: float = 1.6,
        gamma_inicial: float = 0.036,
        epsilon_ep: float = 0.1,
        max_iter_ep: int = 5,
        max_iter_opt: int = 150,
        p_draw: float = 0.0,
    ) -> None:
        self.sigma_inicial = sigma_inicial
        self.gamma_inicial = gamma_inicial
        self.epsilon_ep = epsilon_ep
        self.max_iter_ep = max_iter_ep
        self.max_iter_opt = max_iter_opt
        self.p_draw = p_draw

    # ------------------------------------------------------------------
    # Función objetivo
    # ------------------------------------------------------------------

    def _evaluar_log_evidencia(
        self,
        log_params: np.ndarray,
        composition: List[List[List[str]]],
        times: List[float],
    ) -> float:
        """
        Evalúa ``log p(D | σ, γ)`` para un punto en el espacio log-paramétrico.

        Matemáticamente:

            f(θ) = log p(D | exp(θ₀), exp(θ₁))

        donde θ₀ = log σ, θ₁ = log γ.

        Se crea un objeto ``History`` temporal por cada llamada para garantizar
        evaluaciones independientes (``History`` es stateful).

        Args:
            log_params: Array ``[log_sigma, log_gamma]``.
            composition: Historial de competencias en formato TTT.
            times: Marcas temporales en días desde epoch Unix.

        Returns:
            Log-evidencia como float.  Retorna ``_LOG_EV_FALLBACK`` si la
            convergencia EP falla o el resultado no es finito.
        """
        sigma = float(np.exp(log_params[0]))
        gamma = float(np.exp(log_params[1]))

        try:
            h = History(
                composition=composition,
                times=times,
                sigma=sigma,
                gamma=gamma,
                p_draw=self.p_draw,
            )
            h.convergence(epsilon=self.epsilon_ep, iterations=self.max_iter_ep)
            valor = float(h.log_evidence())
            return valor if np.isfinite(valor) else self._LOG_EV_FALLBACK
        except Exception:
            return self._LOG_EV_FALLBACK

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def calibrar(
        self,
        composition: List[List[List[str]]],
        times: List[float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1e-4, 10.0),   # (sigma_min, sigma_max)
            (1e-4, 10.0),   # (gamma_min, gamma_max)
        ),
    ) -> Dict[str, float]:
        """
        Busca ``(σ*, γ*)`` que maximizan ``log p(D | σ, γ)``.

        Internamente minimiza la log-evidencia negativa en el espacio
        logarítmico: ``min_{θ} -log p(D | exp(θ))``.

        Args:
            composition: Historial de competencias en formato nativo TTT.
                         Obtenido mediante ``TTTJuez.exportar_historial()``.
            times: Marcas temporales en días desde epoch Unix.
            bounds: ``((sigma_min, sigma_max), (gamma_min, gamma_max))``.
                    Los bounds se transforman al espacio log internamente.

        Returns:
            Diccionario con claves:
              - ``'sigma_optimo'``:        σ* encontrado.
              - ``'gamma_optimo'``:        γ* encontrado.
              - ``'log_evidencia_maxima'``: ``log p(D | σ*, γ*)``.

        Raises:
            RuntimeWarning: Si la optimización no converge o los datos son
                            insuficientes.  En todos los casos retorna un
                            diccionario con valores válidos (> 0).
        """
        defaults: Dict[str, float] = {
            'sigma_optimo': self.sigma_inicial,
            'gamma_optimo': self.gamma_inicial,
            'log_evidencia_maxima': float('-inf'),
        }

        # ----------------------------------------------------------------
        # Guardia: mínimo de períodos para calibrar γ de forma significativa
        # ----------------------------------------------------------------
        if len(composition) < 2:
            warnings.warn(
                f"composition contiene {len(composition)} período(s); "
                "se requieren >= 2 para calibrar la dinámica temporal (γ). "
                "Retornando parámetros iniciales.",
                RuntimeWarning,
                stacklevel=2,
            )
            return defaults

        sigma_bounds, gamma_bounds = bounds

        # ----------------------------------------------------------------
        # Punto inicial en espacio log
        # ----------------------------------------------------------------
        x0 = np.array([
            np.log(self.sigma_inicial),
            np.log(self.gamma_inicial),
        ])

        # ----------------------------------------------------------------
        # Bounds en espacio log: (log(min), log(max)) para cada parámetro
        # ----------------------------------------------------------------
        log_bounds = [
            (np.log(sigma_bounds[0]), np.log(sigma_bounds[1])),
            (np.log(gamma_bounds[0]), np.log(gamma_bounds[1])),
        ]

        # ----------------------------------------------------------------
        # Función objetivo: minimizar −log_evidencia
        # ----------------------------------------------------------------
        def objetivo(lp: np.ndarray) -> float:
            """
            Objetivo escalar para scipy:

                f(θ) = −log p(D | exp(θ₀), exp(θ₁))

            scipy minimiza f → equivalente a maximizar log p(D | σ, γ).
            """
            return -self._evaluar_log_evidencia(lp, composition, times)

        # ----------------------------------------------------------------
        # Optimización L-BFGS-B
        # ----------------------------------------------------------------
        try:
            resultado = minimize(
                objetivo,
                x0=x0,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={
                    'maxiter': self.max_iter_opt,
                    'ftol': 1e-9,
                    'gtol': 1e-6,
                },
            )

            log_ev_max = float(-resultado.fun)

            # Detectar si todas las evaluaciones devolvieron el sentinela
            if log_ev_max <= self._LOG_EV_FALLBACK + 1.0:
                warnings.warn(
                    "Todas las evaluaciones de log_evidencia fallaron durante "
                    "la optimización (EP divergió en todos los puntos evaluados). "
                    "Retornando parámetros iniciales.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return {
                    'sigma_optimo': self.sigma_inicial,
                    'gamma_optimo': self.gamma_inicial,
                    'log_evidencia_maxima': log_ev_max,
                }

            # Informar si scipy no convergió formalmente pero encontró un punto
            if not resultado.success:
                warnings.warn(
                    f"scipy.optimize.minimize no convergió: {resultado.message}. "
                    "Se retorna el mejor punto visitado durante la búsqueda.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Mapear de vuelta al espacio original y aplicar clip defensivo
            sigma_opt = float(np.clip(np.exp(resultado.x[0]), *sigma_bounds))
            gamma_opt = float(np.clip(np.exp(resultado.x[1]), *gamma_bounds))

            return {
                'sigma_optimo': sigma_opt,
                'gamma_optimo': gamma_opt,
                'log_evidencia_maxima': log_ev_max,
            }

        except Exception as exc:
            warnings.warn(
                f"Error inesperado en scipy.optimize.minimize: {exc}. "
                "Retornando parámetros iniciales por defecto.",
                RuntimeWarning,
                stacklevel=2,
            )
            return defaults
