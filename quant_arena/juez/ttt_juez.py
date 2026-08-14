# =============================================================================
# FILE: quant_arena/juez/ttt_juez.py
# Adaptador del paquete TrueSkillThroughTime (Landfried, 2024)
# Referencia API: https://github.com/glandfried/TrueSkillThroughTime.py
#
# Clases clave del paquete externo (README, sección "Basic Example"):
#   History(composition, times, sigma, gamma, p_draw)
#   Player(Gaussian(mu, sigma), beta, gamma)
#   Gaussian(mu, sigma)
#
# Métodos relevantes (README, sección "Methods"):
#   history.convergence(epsilon, iterations)  -> EP hasta convergencia
#   history.learning_curves()                 -> {agente: [(t, Gaussian), ...]}
#   history.log_evidence()                    -> evidencia marginal del modelo
#
# Formato de composition (README, sección "Composition"):
#   composition[k] = [[team1_player1, ...], [team2_player1, ...], ...]
#   Orden ascendente de índice = ranking de mejor a peor (índice 0 gana).
# =============================================================================
from __future__ import annotations

import copy
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Instalar con: pip install trueskillthroughtime
from trueskillthroughtime import History, Player, Gaussian  # type: ignore[import]

from quant_arena.core.abstracciones import AbstractJuez, MetricasResultado
from quant_arena.core.excepciones import CausalidadVioladaError

logger = logging.getLogger(__name__)


class TTTJuez(AbstractJuez):
    """
    Juez algorítmico basado en TrueSkill Through Time (Landfried, 2024).

    Modela la habilidad latente de cada estrategia del Zoo como un proceso
    de Wiener gaussiano. La inferencia se realiza mediante Expectation
    Propagation (EP) sobre el grafo factorial completo de competencias.

    Mapa conceptual (Estrategia -> TTT):
      Estrategia   ->  Jugador (Player con Gaussian prior)
      Período t    ->  Partida (Game) multi-equipo de N estrategias
      Ranking KPI  ->  Orden de equipos en composition[t] (índice 0 = ganador)
      mu posterior ->  Señal de asignación táctica de capital
    """

    # ------------------------------------------------------------------
    # Constantes de calibración (referencia: ejemplo ATP en el README)
    # ------------------------------------------------------------------
    # Ref. README: h = History(composition=composition, times=days, sigma=1.6, gamma=0.036)
    _SIGMA_DEFAULT: float = 1.6     # Incertidumbre inicial (prior de habilidad)
    _GAMMA_DEFAULT: float = 0.036   # Drift del proceso de Wiener entre períodos
    # Ref. README: h.convergence(epsilon=0.01, iterations=10)
    _EPSILON_DEFAULT: float = 0.01  # Tolerancia de convergencia EP
    _MAX_ITER_DEFAULT: int = 30     # Iteraciones máximas de propagación de mensajes

    def __init__(
        self,
        sigma: float = _SIGMA_DEFAULT,
        gamma: float = _GAMMA_DEFAULT,
        p_draw: float = 0.0,
        epsilon: float = _EPSILON_DEFAULT,
        max_iteraciones: int = _MAX_ITER_DEFAULT,
        modo_causal_estricto: bool = True,
        ventana: Optional[int] = None,
    ) -> None:
        """
        Args:
            sigma:   Desv. estándar del prior de habilidad.
                     Ref. README: sigma=1.6 calibrado en datos ATP (1968-2021).
            gamma:   Velocidad de cambio de habilidad inter-período.
                     Ref. README: gamma=0.036 para procesos con régimen cambiante.
            p_draw:  Probabilidad de empate entre estrategias.
                     Ref. README: Game(teams, result, p_draw=0.25). 0.0 para rankings.
            epsilon: Tolerancia de convergencia del algoritmo EP.
                     Ref. README: history.convergence(epsilon=0.01).
            max_iteraciones: Límite de iteraciones EP.
                     Ref. README: history.convergence(iterations=10).
            modo_causal_estricto: Si True (default), `registrar_periodo()`
                                  EXIGE que los períodos se registren en
                                  orden temporal no decreciente — lanza
                                  `CausalidadVioladaError` si `tiempo` es
                                  anterior al último período ya registrado
                                  (ver `registrar_periodo`, H6).

                                  Esto es un guardia de ORDEN DE REGISTRO,
                                  no una garantía completa contra el
                                  look-ahead del suavizador de TTT: "Through
                                  Time" propaga información hacia adelante
                                  Y HACIA ATRÁS sobre TODO el historial
                                  registrado en `_composition`. La garantía
                                  causal real del sistema depende de que el
                                  LLAMADOR (`BacktestEngine`) nunca registre
                                  un período t+1 antes de consultar
                                  `pesos_asignacion()` para el período t —
                                  el motor ya cumple esto por construcción
                                  (registra y consulta dentro de la misma
                                  iteración del walk-forward, antes de
                                  avanzar `fecha_corte`). Este guardia
                                  detecta el error más común que rompería
                                  esa disciplina: registrar fuera de orden.
            ventana:  Si se especifica, `actualizar()` ajusta `History` solo
                     sobre los últimos `ventana` períodos registrados en vez
                     de todo el historial — el costo por rebalanceo pasa de
                     O(T) a O(ventana) (H5). Trade-off: cada período fuera
                     de la ventana pierde su influencia en el prior de
                     habilidad; el "arranque" de cada ventana parte del
                     prior sigma puro, no del posterior acumulado. None
                     (default) usa el historial completo — comportamiento
                     original sin cambios.
        """
        self.sigma: float = sigma
        self.gamma: float = gamma
        self.p_draw: float = p_draw
        self.epsilon: float = epsilon
        self.max_iteraciones: int = max_iteraciones
        self.modo_causal_estricto: bool = modo_causal_estricto
        self.ventana: Optional[int] = ventana

        # Estado interno: historial de competencias en formato TTT
        # Ref. README: composition = lista de matchups
        # composition[k] = [[ganador], [2do], ..., [perdedor]]
        self._composition: List[List[List[str]]] = []
        self._times: List[float] = []
        self._metrica_por_periodo: List[str] = []

        # Cache del modelo ajustado; se invalida con _dirty=True al agregar datos
        self._history: Optional[History] = None
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Implementación de AbstractJuez
    # ------------------------------------------------------------------

    def registrar_periodo(
        self,
        metricas_periodo: Dict[str, MetricasResultado],
        tiempo: float,
        metrica_ranking: str = 'sharpe',
    ) -> None:
        """
        Transforma las métricas de un período en un evento de competencia TTT.

        La estrategia con mayor valor de `metrica_ranking` ocupa la posición 0
        en la composition (= ganadora en el lenguaje de TTT).

        Args:
            metricas_periodo: {nombre_estrategia -> MetricasResultado}
            tiempo:           Días desde época Unix (ej. timestamp / 86400).
                              Ref. README: days = [datetime.strptime(t, "%Y-%m-%d")
                                                   .timestamp()/(60*60*24)]
            metrica_ranking:  Campo de MetricasResultado para ordenar estrategias.

        Garantía causal: este método SOLO registra el evento; NO ejecuta inferencia.
        La actualización del modelo ocurre explícitamente en self.actualizar().

        Raises:
            CausalidadVioladaError: si `modo_causal_estricto=True` y `tiempo`
                es anterior al último período ya registrado (H6) — ver
                docstring de `modo_causal_estricto` en `__init__`.
        """
        if self.modo_causal_estricto and self._times:
            tiempo_maximo_previo = max(self._times)
            if float(tiempo) < tiempo_maximo_previo:
                raise CausalidadVioladaError(
                    f"registrar_periodo recibió tiempo={tiempo}, anterior al último "
                    f"período ya registrado (t_max={tiempo_maximo_previo}). En "
                    "modo_causal_estricto, los períodos deben registrarse en orden "
                    "temporal no decreciente."
                )

        if not metricas_periodo:
            return

        valores: Dict[str, float] = {
            nombre: getattr(m, metrica_ranking, np.nan)
            for nombre, m in metricas_periodo.items()
        }

        valores_validos: Dict[str, float] = {
            k: v for k, v in valores.items() if np.isfinite(v)
        }

        if len(valores_validos) < 2:
            return

        # Ordenar de mejor (mayor) a peor; primer elemento = ganador en TTT
        # Ref. README: "Orden determina resultado: equipos con índice menor ganan"
        ranking: List[Tuple[str, float]] = sorted(
            valores_validos.items(), key=lambda x: x[1], reverse=True
        )

        # Cada estrategia = un equipo unipersonal
        # Ref. README: team_a = [a1]; teams = [team_a, team_b]
        # Para N estrategias: [[mejor], [2do], ..., [peor]]
        game: List[List[str]] = [[nombre] for nombre, _ in ranking]

        self._composition.append(game)
        self._times.append(float(tiempo))
        self._metrica_por_periodo.append(metrica_ranking)
        self._dirty = True
        self._history = None

    def actualizar(self) -> None:
        """
        Ejecuta la propagación de mensajes EP sobre el grafo factorial.

        Ref. README:
          h = History(composition=composition, times=days, sigma=1.6, gamma=0.036)
          h.convergence(epsilon=0.01, iterations=10)

        Nota sobre causalidad: en modo_causal_estricto=True, el Motor de Backtesting
        es responsable de llamar a este método con self._composition[:t] antes de
        solicitar predicciones para el período t.

        Ventana deslizante (H5): si `self.ventana` está configurado, el grafo
        factorial se construye solo sobre los últimos `self.ventana` períodos
        (composición y tiempos), no sobre el historial completo — el costo
        por rebalanceo pasa de O(T) a O(ventana). Cada ventana nueva
        "olvida" el posterior acumulado antes de su primer período (arranca
        desde el prior `sigma` puro), a cambio de un costo acotado.
        """
        if not self._composition:
            return

        composition = self._composition
        times = self._times
        if self.ventana is not None and len(composition) > self.ventana:
            composition = composition[-self.ventana:]
            times = times[-self.ventana:]

        # Ref. README: History(composition=composition, times=days, sigma=1.6, gamma=0.036)
        self._history = History(
            composition=composition,
            times=times,
            sigma=self.sigma,
            gamma=self.gamma,
            p_draw=self.p_draw,
        )

        # Ref. README: h.convergence(epsilon=0.01, iterations=10)
        self._history.convergence(
            epsilon=self.epsilon,
            iterations=self.max_iteraciones,
        )
        self._dirty = False

    def exportar_historial(self) -> Tuple[List[List[List[str]]], List[float]]:
        """
        Retorna copias defensivas del historial de competencias registrado.

        Diseñado para alimentar a ``OptimizadorTTT`` sin exponer referencias
        mutables al estado interno del Juez.

        Notas sobre las copias:
          - ``_composition`` → ``copy.deepcopy``: la estructura anidada
            ``List[List[List[str]]]`` contiene contenedores mutables en todos
            los niveles.  Una copia superficial (``list()``) dejaría los
            sub-objetos compartidos; cualquier mutación en el resultado
            —``comp[0][0].append("x")`` o ``comp[0] = [...]``— propagaría
            silenciosamente al estado interno.
          - ``_times`` → ``list()``: ``float`` es inmutable en Python; compartir
            referencias a los mismos flotantes es seguro.  Una reasignación
            ``times[i] = 999.0`` solo afecta la copia, nunca el original.

        Returns:
            Tuple ``(composition, times)`` listos para pasar a
            ``OptimizadorTTT.calibrar()``.
        """
        return copy.deepcopy(self._composition), list(self._times)

    def habilidades_latentes(self) -> Dict[str, Tuple[float, float]]:
        """
        Retorna la distribución posterior de habilidad más reciente por estrategia.

        Returns:
            {nombre_estrategia: (mu, sigma)} del último punto temporal disponible.

        Ref. README:
          lc = h.learning_curves()
          # {"player": [(time, Gaussian(mu=value, sigma=value)), ...]}
          mu    = [v[1].mu for v in lc[agent]]
          sigma = [v[1].sigma for v in lc[agent]]
        """
        if self._history is None or self._dirty:
            self.actualizar()

        if self._history is None:
            return {}

        # Ref. README: lc = h.learning_curves()
        curvas: Dict = self._history.learning_curves()

        resultado: Dict[str, Tuple[float, float]] = {}
        for agente, puntos_temporales in curvas.items():
            if puntos_temporales:
                # Ref. README: [(time, Gaussian(mu=value, sigma=value)), ...]
                gaussiano_final = puntos_temporales[-1][1]
                resultado[agente] = (
                    float(gaussiano_final.mu),
                    float(gaussiano_final.sigma),
                )

        return resultado

    def pesos_asignacion(
        self,
        metodo: str = 'mu_sobre_sigma',
        fallback: str = 'uniforme',
    ) -> Dict[str, float]:
        """
        Convierte habilidades latentes en pesos de asignación táctica de capital.

        Métodos disponibles:
          'mu_sobre_sigma': ratio señal/ruido (análogo a Sharpe bayesiano).
          'mu':             Media posterior bruta. Ignora incertidumbre.
          'softmax_mu':     Softmax sobre mu. Garantiza peso > 0 para todas.
          'kelly_bayes':    mu / sigma² sobre la escala nativa del posterior
                            TTT — forma de Kelly usando solo información
                            interna del Juez (sin retornos realizados; para
                            Kelly acoplado a retornos reales, ver
                            `backtesting.kelly_sizing.KellyBayesianSizer`,
                            que además NO fuerza suma=1). Penaliza la
                            incertidumbre cuadráticamente en vez de
                            linealmente frente a 'mu_sobre_sigma' (H4).

        Args:
            metodo:   Ver arriba.
            fallback: Qué hacer cuando TODOS los scores brutos son <= 0
                     (nadie tiene ventaja positiva aparente):
                       'uniforme' (default): reparte 1/N — comportamiento
                            original, preserva sum(pesos)==1.0 siempre.
                       'cash':    expone 0.0 a todas las estrategias en vez
                            de invertir igual sin convicción alguna (H4).
                            Rompe la garantía sum==1.0 EN ESTE CASO
                            específico — ver nota de Garantías.
                     Ignorado por 'softmax_mu' (nunca colapsa a score=0).

        Garantías: pesos[i] >= 0 siempre. sum(pesos.values()) == 1.0 salvo
        con fallback='cash' en el caso de scores todos no-positivos, donde
        sum(pesos.values()) == 0.0 (sin exposición, a propósito).

        Raises:
            ValueError: si `metodo` o `fallback` no son reconocidos.
        """
        if fallback not in ('uniforme', 'cash'):
            raise ValueError(
                f"fallback desconocido: '{fallback}'. Opciones: 'uniforme', 'cash'."
            )

        habilidades: Dict[str, Tuple[float, float]] = self.habilidades_latentes()
        if not habilidades:
            return {}

        scores_raw: Dict[str, float]

        if metodo == 'mu_sobre_sigma':
            scores_raw = {k: v[0] / v[1] for k, v in habilidades.items()}
        elif metodo == 'mu':
            scores_raw = {k: v[0] for k, v in habilidades.items()}
        elif metodo == 'kelly_bayes':
            scores_raw = {k: v[0] / (v[1] ** 2) for k, v in habilidades.items()}
        elif metodo == 'softmax_mu':
            nombres = list(habilidades.keys())
            mus = np.array([habilidades[n][0] for n in nombres])
            exp_mus = np.exp(mus - mus.max())
            pesos_vals = (exp_mus / exp_mus.sum()).tolist()
            return dict(zip(nombres, pesos_vals))
        else:
            raise ValueError(
                f"Método desconocido: '{metodo}'. "
                f"Opciones: 'mu_sobre_sigma', 'mu', 'kelly_bayes', 'softmax_mu'."
            )

        scores: Dict[str, float] = {k: max(v, 0.0) for k, v in scores_raw.items()}
        total: float = sum(scores.values())

        if total == 0.0:
            if fallback == 'cash':
                return {k: 0.0 for k in scores}
            n = len(scores)
            return {k: 1.0 / n for k in scores}

        return {k: v / total for k, v in scores.items()}

    # ------------------------------------------------------------------
    # Utilidades de análisis y visualización
    # ------------------------------------------------------------------

    def curvas_aprendizaje(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Trayectoria temporal completa de habilidad para cada estrategia.

        Returns:
            {nombre: [(tiempo, mu, sigma), ...]}

        Ref. README:
          t     = [v[0] for v in lc[agent]]
          mu    = [v[1].mu for v in lc[agent]]
          sigma = [v[1].sigma for v in lc[agent]]
        """
        if self._history is None or self._dirty:
            self.actualizar()

        if self._history is None:
            return {}

        curvas_raw: Dict = self._history.learning_curves()
        return {
            agente: [
                (float(t), float(g.mu), float(g.sigma))
                for t, g in puntos
            ]
            for agente, puntos in curvas_raw.items()
        }

    def log_evidencia(self) -> float:
        """
        Log-evidencia marginal del modelo TTT.
        Permite comparar configuraciones de hiperparámetros (sigma, gamma)
        bajo el marco bayesiano: mayor log_evidencia = mejor ajuste.

        Ref. README: history.log_evidence()
        """
        if self._history is None or self._dirty:
            self.actualizar()
        if self._history is None:
            return float('-inf')
        return float(self._history.log_evidence())

    def calibrar_hiperparametros(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1e-4, 10.0),
            (1e-4, 10.0),
        ),
    ) -> Dict[str, float]:
        """
        Recalibra (σ, γ) maximizando la log-evidencia marginal sobre el
        historial ya registrado, vía `calibracion.optimizador.OptimizadorTTT`
        (roadmap v1.1 del README del proyecto).

        Actualiza `self.sigma`/`self.gamma` IN-PLACE con los valores óptimos
        encontrados e invalida el caché (`self._dirty = True`), por lo que
        la siguiente llamada a `actualizar()`/`habilidades_latentes()`
        reconstruye `History` con los nuevos hiperparámetros.

        Args:
            bounds: ((sigma_min, sigma_max), (gamma_min, gamma_max)) pasado
                    a `OptimizadorTTT.calibrar()`.

        Returns:
            El diccionario de `OptimizadorTTT.calibrar()`:
            {'sigma_optimo', 'gamma_optimo', 'log_evidencia_maxima'}.
            Con < 2 períodos registrados, retorna (σ, γ) sin cambios
            (mismo comportamiento de degradación que `OptimizadorTTT`,
            que emite un `RuntimeWarning` en ese caso).
        """
        from quant_arena.calibracion.optimizador import OptimizadorTTT

        composition, times = self.exportar_historial()
        optimizador = OptimizadorTTT(
            sigma_inicial=self.sigma,
            gamma_inicial=self.gamma,
            p_draw=self.p_draw,
        )
        resultado = optimizador.calibrar(composition, times, bounds=bounds)

        self.sigma = resultado['sigma_optimo']
        self.gamma = resultado['gamma_optimo']
        self._dirty = True
        self._history = None

        return resultado

    def snapshot_estado(self) -> pd.DataFrame:
        """
        DataFrame con el estado posterior actual de todas las estrategias.
        Útil para logging, monitoreo y reportes del módulo Resultados.

        Columnas: mu | sigma | mu_sobre_sigma | peso_asignado | n_periodos
        """
        habilidades: Dict[str, Tuple[float, float]] = self.habilidades_latentes()
        pesos: Dict[str, float] = self.pesos_asignacion()

        filas: List[Dict] = []
        for nombre, (mu, sigma) in habilidades.items():
            n_activo: int = sum(
                1 for game in self._composition
                if any(nombre in equipo for equipo in game)
            )
            filas.append({
                'estrategia':     nombre,
                'mu':             round(mu, 6),
                'sigma':          round(sigma, 6),
                'mu_sobre_sigma': round(mu / sigma if sigma > 0 else 0.0, 6),
                'peso_asignado':  round(pesos.get(nombre, 0.0), 6),
                'n_periodos':     n_activo,
            })

        return (
            pd.DataFrame(filas)
            .set_index('estrategia')
            .sort_values('mu', ascending=False)
        )

    def exportar_curvas_df(self) -> pd.DataFrame:
        """
        Long-format DataFrame de curvas de aprendizaje para matplotlib/seaborn.

        Columnas: estrategia | tiempo | mu | sigma | banda_sup | banda_inf
        """
        curvas = self.curvas_aprendizaje()
        filas: List[Dict] = []
        for nombre, puntos in curvas.items():
            for tiempo, mu, sigma in puntos:
                filas.append({
                    'estrategia': nombre,
                    'tiempo':     tiempo,
                    'mu':         mu,
                    'sigma':      sigma,
                    'banda_sup':  mu + sigma,
                    'banda_inf':  mu - sigma,
                })
        return pd.DataFrame(filas)
