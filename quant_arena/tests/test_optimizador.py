# =============================================================================
# FILE: quant_arena/tests/test_optimizador.py
# Suite pytest para OptimizadorTTT y TTTJuez.exportar_historial().
#
# Estrategia de aislamiento:
#   - conftest.py ya inyecta un stub de trueskillthroughtime en sys.modules.
#   - Los tests que requieren un log_evidence con forma definida usan
#     unittest.mock.patch para reemplazar History con una parábola en log-space.
#   - Datos sintéticos: 3 períodos, 2 estrategias → ejecución < 10 ms.
#
# Cobertura:
#   TTTJuez.exportar_historial():
#     - Retorna copias, no referencias (deepcopy verification)
#   OptimizadorTTT.calibrar():
#     - Claves correctas en el dict de retorno
#     - sigma_optimo > 0, gamma_optimo > 0
#     - Parámetros dentro de los bounds especificados
#     - log_evidencia_maxima es float finito
#     - Optimizador mejora sobre puntos extremos (el mock penaliza extremos)
#     - Fallback con < 2 períodos válidos → RuntimeWarning + iniciales
#     - Fallback cuando History.convergence() siempre falla → RuntimeWarning
#   _evaluar_log_evidencia():
#     - Retorna _LOG_EV_FALLBACK cuando convergence() lanza excepción
#     - Retorna float finito con mock válido
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from quant_arena.calibracion.optimizador import OptimizadorTTT
from quant_arena.juez.ttt_juez import TTTJuez
from quant_arena.core.abstracciones import MetricasResultado


# ---------------------------------------------------------------------------
# Datos sintéticos compartidos
# ---------------------------------------------------------------------------

# 3 períodos, 2 estrategias — formato nativo TTT
COMPOSITION_3P_2S: list = [
    [["momentum"], ["value"]],   # período 0: momentum ganó
    [["value"],    ["momentum"]], # período 1: value    ganó
    [["momentum"], ["value"]],   # período 2: momentum ganó
]

# Días desde epoch Unix ≈ 2020-01-02, 2020-02-03, 2020-03-02
TIMES_3P_2S: list = [18263.0, 18295.0, 18323.0]


# ---------------------------------------------------------------------------
# Mock factory: parábola en log-space con mínimo en (sigma_opt, gamma_opt)
# ---------------------------------------------------------------------------

def _make_history_factory(sigma_opt: float = 1.6, gamma_opt: float = 0.036):
    """
    Devuelve una callable que instancia mocks de History cuya log_evidence
    es una parábola en el espacio logarítmico, con máximo en (sigma_opt, gamma_opt):

        log_ev(σ, γ) = −3·(log σ − log σ*)² − 3·(log γ − log γ*)²

    El optimizador debe encontrar σ ≈ σ* y γ ≈ γ* partiendo de cualquier
    punto inicial dentro de los bounds.
    """
    def factory(*args: object, **kwargs: object) -> MagicMock:
        sigma = float(kwargs.get('sigma', sigma_opt))   # type: ignore[arg-type]
        gamma = float(kwargs.get('gamma', gamma_opt))   # type: ignore[arg-type]
        h = MagicMock()
        log_ev = (
            -3.0 * (np.log(sigma) - np.log(sigma_opt)) ** 2
            - 3.0 * (np.log(gamma) - np.log(gamma_opt)) ** 2
        )
        h.log_evidence.return_value = float(log_ev)
        return h
    return factory


# ---------------------------------------------------------------------------
# Tests: TTTJuez.exportar_historial()
# ---------------------------------------------------------------------------

class TestExportarHistorial:
    def _juez_con_historia(self) -> TTTJuez:
        """TTTJuez con 2 períodos registrados y mock _history inyectado."""
        juez = TTTJuez()
        juez.registrar_periodo(
            {"momentum": MetricasResultado(sharpe=1.5),
             "value":    MetricasResultado(sharpe=0.8)},
            tiempo=18263.0,
        )
        juez.registrar_periodo(
            {"momentum": MetricasResultado(sharpe=0.3),
             "value":    MetricasResultado(sharpe=1.1)},
            tiempo=18295.0,
        )
        return juez

    def test_retorna_tupla_de_dos_elementos(self) -> None:
        juez = self._juez_con_historia()
        resultado = juez.exportar_historial()
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2

    def test_composition_y_times_tienen_misma_longitud(self) -> None:
        juez = self._juez_con_historia()
        comp, times = juez.exportar_historial()
        assert len(comp) == len(times) == 2

    def test_composition_es_deepcopy_no_referencia(self) -> None:
        """Mutar la composición retornada NO debe afectar el estado interno."""
        juez = self._juez_con_historia()
        comp_orig_len = len(juez._composition[0])   # equipos en período 0

        comp, _ = juez.exportar_historial()
        comp[0].append(["intruso"])                  # muta la copia

        # El estado interno del juez no debe haber cambiado
        assert len(juez._composition[0]) == comp_orig_len

    def test_times_es_copia_independiente(self) -> None:
        """Mutar times retornado NO debe afectar _times interno."""
        juez = self._juez_con_historia()
        tiempo_original = juez._times[0]

        _, times = juez.exportar_historial()
        times[0] = 99999.0                           # muta la copia

        assert juez._times[0] == tiempo_original

    def test_ganador_en_posicion_cero(self) -> None:
        """El primer equipo de cada game debe ser el de mayor sharpe."""
        juez = self._juez_con_historia()
        comp, _ = juez.exportar_historial()
        # Período 0: momentum (sharpe=1.5) > value (sharpe=0.8) → ganador = momentum
        assert comp[0][0] == ["momentum"]
        # Período 1: value (sharpe=1.1) > momentum (sharpe=0.3) → ganador = value
        assert comp[1][0] == ["value"]


# ---------------------------------------------------------------------------
# Tests: OptimizadorTTT.calibrar() — estructura del resultado
# ---------------------------------------------------------------------------

class TestCalibracionEstructura:
    @pytest.fixture
    def opt(self) -> OptimizadorTTT:
        return OptimizadorTTT(sigma_inicial=1.6, gamma_inicial=0.036)

    def test_retorna_tres_claves_exactas(self, opt: OptimizadorTTT) -> None:
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=_make_history_factory(),
        ):
            resultado = opt.calibrar(COMPOSITION_3P_2S, TIMES_3P_2S)

        assert set(resultado.keys()) == {'sigma_optimo', 'gamma_optimo', 'log_evidencia_maxima'}

    def test_sigma_optimo_positivo(self, opt: OptimizadorTTT) -> None:
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=_make_history_factory(),
        ):
            resultado = opt.calibrar(COMPOSITION_3P_2S, TIMES_3P_2S)

        assert resultado['sigma_optimo'] > 0.0

    def test_gamma_optimo_positivo(self, opt: OptimizadorTTT) -> None:
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=_make_history_factory(),
        ):
            resultado = opt.calibrar(COMPOSITION_3P_2S, TIMES_3P_2S)

        assert resultado['gamma_optimo'] > 0.0

    def test_log_evidencia_es_float_finito(self, opt: OptimizadorTTT) -> None:
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=_make_history_factory(),
        ):
            resultado = opt.calibrar(COMPOSITION_3P_2S, TIMES_3P_2S)

        assert isinstance(resultado['log_evidencia_maxima'], float)
        assert np.isfinite(resultado['log_evidencia_maxima'])


# ---------------------------------------------------------------------------
# Tests: OptimizadorTTT.calibrar() — respeto de bounds
# ---------------------------------------------------------------------------

class TestCalibracionBounds:
    @pytest.mark.parametrize(
        "sigma_bounds, gamma_bounds",
        [
            ((0.5, 5.0),  (0.005, 0.5)),
            ((0.1, 3.0),  (0.001, 0.2)),
            ((1.0, 10.0), (0.01,  1.0)),
        ],
    )
    def test_parametros_dentro_de_bounds(
        self,
        sigma_bounds: tuple,
        gamma_bounds: tuple,
    ) -> None:
        opt = OptimizadorTTT(sigma_inicial=1.6, gamma_inicial=0.036)
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=_make_history_factory(),
        ):
            resultado = opt.calibrar(
                COMPOSITION_3P_2S,
                TIMES_3P_2S,
                bounds=(sigma_bounds, gamma_bounds),
            )

        assert sigma_bounds[0] <= resultado['sigma_optimo'] <= sigma_bounds[1]
        assert gamma_bounds[0] <= resultado['gamma_optimo'] <= gamma_bounds[1]


# ---------------------------------------------------------------------------
# Tests: la función objetivo penaliza parámetros extremos
# ---------------------------------------------------------------------------

class TestPenalizacionExtremos:
    """
    Verifica que log_ev en el punto óptimo del mock sea estrictamente mayor
    que en puntos extremos de sigma y gamma.  Esto confirma que el mock es
    coherente y que _evaluar_log_evidencia lo consume correctamente.
    """

    SIGMA_OPT = 1.6
    GAMMA_OPT = 0.036

    def _log_ev_en(self, sigma: float, gamma: float) -> float:
        opt = OptimizadorTTT()
        log_params = np.array([np.log(sigma), np.log(gamma)])
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=_make_history_factory(self.SIGMA_OPT, self.GAMMA_OPT),
        ):
            return opt._evaluar_log_evidencia(log_params, COMPOSITION_3P_2S, TIMES_3P_2S)

    def test_optimo_mejor_que_gamma_casi_cero(self) -> None:
        log_ev_opt    = self._log_ev_en(self.SIGMA_OPT, self.GAMMA_OPT)
        log_ev_extrem = self._log_ev_en(self.SIGMA_OPT, 0.001)
        assert log_ev_opt > log_ev_extrem

    def test_optimo_mejor_que_sigma_extremo(self) -> None:
        log_ev_opt    = self._log_ev_en(self.SIGMA_OPT, self.GAMMA_OPT)
        log_ev_extrem = self._log_ev_en(8.0, self.GAMMA_OPT)
        assert log_ev_opt > log_ev_extrem

    def test_optimo_mejor_que_ambos_extremos(self) -> None:
        log_ev_opt    = self._log_ev_en(self.SIGMA_OPT, self.GAMMA_OPT)
        log_ev_extrem = self._log_ev_en(0.11, 0.9)
        assert log_ev_opt > log_ev_extrem


# ---------------------------------------------------------------------------
# Tests: comportamiento fallback y manejo de errores
# ---------------------------------------------------------------------------

class TestFallback:
    def test_menos_de_dos_periodos_emite_warning(self) -> None:
        opt = OptimizadorTTT()
        composition_1p = [[["A"], ["B"]]]   # solo 1 período
        times_1p       = [18263.0]

        with pytest.warns(RuntimeWarning, match=">= 2"):
            resultado = opt.calibrar(composition_1p, times_1p)

        assert resultado['sigma_optimo'] > 0.0
        assert resultado['gamma_optimo'] > 0.0

    def test_composition_vacia_emite_warning(self) -> None:
        opt = OptimizadorTTT()
        with pytest.warns(RuntimeWarning):
            resultado = opt.calibrar([], [])

        assert resultado['sigma_optimo'] == opt.sigma_inicial
        assert resultado['gamma_optimo'] == opt.gamma_inicial

    def test_ep_siempre_falla_emite_warning_y_retorna_positivos(self) -> None:
        """Cuando convergence() lanza excepción en todos los puntos."""
        def history_que_explota(*args: object, **kwargs: object) -> MagicMock:
            h = MagicMock()
            h.convergence.side_effect = RuntimeError("EP diverged")
            return h

        opt = OptimizadorTTT()
        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=history_que_explota,
        ):
            with pytest.warns(RuntimeWarning):
                resultado = opt.calibrar(COMPOSITION_3P_2S, TIMES_3P_2S)

        assert resultado['sigma_optimo'] > 0.0
        assert resultado['gamma_optimo'] > 0.0

    def test_evaluar_log_evidencia_retorna_fallback_ante_excepcion(self) -> None:
        """_evaluar_log_evidencia no propaga excepciones; retorna _LOG_EV_FALLBACK."""
        opt = OptimizadorTTT()

        def history_corrupta(*args: object, **kwargs: object) -> MagicMock:
            h = MagicMock()
            h.convergence.side_effect = ValueError("datos corruptos")
            return h

        with patch(
            'quant_arena.calibracion.optimizador.History',
            side_effect=history_corrupta,
        ):
            log_ev = opt._evaluar_log_evidencia(
                np.array([np.log(1.6), np.log(0.036)]),
                COMPOSITION_3P_2S,
                TIMES_3P_2S,
            )

        assert log_ev == pytest.approx(OptimizadorTTT._LOG_EV_FALLBACK)

    def test_parametros_iniciales_fuera_de_defaults_preservados_en_fallback(self) -> None:
        """Con fallo total, los parámetros retornados son los iniciales del optimizador."""
        sigma_i, gamma_i = 2.5, 0.08
        opt = OptimizadorTTT(sigma_inicial=sigma_i, gamma_inicial=gamma_i)

        with pytest.warns(RuntimeWarning):
            resultado = opt.calibrar([], [])

        assert resultado['sigma_optimo'] == pytest.approx(sigma_i)
        assert resultado['gamma_optimo'] == pytest.approx(gamma_i)
