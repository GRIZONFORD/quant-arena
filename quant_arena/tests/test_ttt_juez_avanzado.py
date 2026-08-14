# =============================================================================
# FILE: quant_arena/tests/test_ttt_juez_avanzado.py
# Suite pytest para las cuatro mejoras de §1.3 sobre TTTJuez:
#   - Causalidad exigida en registrar_periodo() (H6)
#   - Ventana deslizante en actualizar() (H5)
#   - metodo='kelly_bayes' y fallback='cash' en pesos_asignacion() (H4)
#   - calibrar_hiperparametros() conectado a OptimizadorTTT
#
# Usa el paquete trueskillthroughtime REAL (no el stub de conftest.py) para
# cobertura end-to-end genuina donde es factible; los tests de
# pesos_asignacion() siguen el patrón de mock de test_juez.py para
# aislar el cálculo de pesos de la inferencia EP.
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from quant_arena.core.abstracciones import MetricasResultado
from quant_arena.core.excepciones import CausalidadVioladaError
from quant_arena.juez.ttt_juez import TTTJuez


# ---------------------------------------------------------------------------
# Helpers (mismo patrón que test_juez.py)
# ---------------------------------------------------------------------------

class _G:
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)


def _juez_con_habilidades(habilidades: dict) -> TTTJuez:
    juez = TTTJuez()
    mock_history = MagicMock()
    mock_history.learning_curves.return_value = {
        nombre: [(1.0, _G(mu, sigma))] for nombre, (mu, sigma) in habilidades.items()
    }
    juez._history = mock_history
    juez._dirty = False
    return juez


def _registrar_n_periodos(juez: TTTJuez, n: int, n_jugadores: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    jugadores = [f"strat_{i}" for i in range(n_jugadores)]
    for t in range(1, n + 1):
        metricas = {j: MetricasResultado(sharpe=float(rng.normal(0.3, 0.4))) for j in jugadores}
        juez.registrar_periodo(metricas, tiempo=float(t))


# ---------------------------------------------------------------------------
# Causalidad exigida (H6)
# ---------------------------------------------------------------------------

def test_causalidad_estricta_lanza_al_registrar_fuera_de_orden():
    juez = TTTJuez(modo_causal_estricto=True)
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=10.0,
    )
    with pytest.raises(CausalidadVioladaError):
        juez.registrar_periodo(
            {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
            tiempo=5.0,
        )


def test_causalidad_estricta_permite_orden_no_decreciente():
    juez = TTTJuez(modo_causal_estricto=True)
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=10.0,
    )
    # Mismo tiempo (empate) y tiempo estrictamente posterior: ambos válidos.
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=10.0,
    )
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=11.0,
    )
    assert len(juez._composition) == 3


def test_causalidad_desactivada_permite_cualquier_orden():
    juez = TTTJuez(modo_causal_estricto=False)
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=10.0,
    )
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=5.0,
    )
    assert len(juez._composition) == 2


def test_causalidad_violacion_no_modifica_estado():
    """Un registro que viola causalidad no debe dejar el estado a medio actualizar."""
    juez = TTTJuez(modo_causal_estricto=True)
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=10.0,
    )
    n_antes = len(juez._composition)
    with pytest.raises(CausalidadVioladaError):
        juez.registrar_periodo(
            {"A": MetricasResultado(sharpe=2.0), "B": MetricasResultado(sharpe=1.0)},
            tiempo=1.0,
        )
    assert len(juez._composition) == n_antes


# ---------------------------------------------------------------------------
# Ventana deslizante (H5)
# ---------------------------------------------------------------------------

def test_ventana_deslizante_acota_composition_pasada_a_history(monkeypatch):
    capturado = {}

    import quant_arena.juez.ttt_juez as modulo

    class _HistoryEspia:
        def __init__(self, composition, times, sigma, gamma, p_draw):
            capturado["composition"] = composition
            capturado["times"] = times

        def convergence(self, epsilon, iterations):
            pass

    monkeypatch.setattr(modulo, "History", _HistoryEspia)

    juez = TTTJuez(ventana=5, modo_causal_estricto=False)
    _registrar_n_periodos(juez, n=20)
    juez.actualizar()

    assert len(capturado["composition"]) == 5
    assert len(capturado["times"]) == 5
    # Los últimos 5 tiempos registrados (16..20), no los primeros.
    assert capturado["times"] == [16.0, 17.0, 18.0, 19.0, 20.0]


def test_sin_ventana_usa_historial_completo(monkeypatch):
    capturado = {}
    import quant_arena.juez.ttt_juez as modulo

    class _HistoryEspia:
        def __init__(self, composition, times, sigma, gamma, p_draw):
            capturado["composition"] = composition

        def convergence(self, epsilon, iterations):
            pass

    monkeypatch.setattr(modulo, "History", _HistoryEspia)

    juez = TTTJuez(ventana=None, modo_causal_estricto=False)
    _registrar_n_periodos(juez, n=20)
    juez.actualizar()

    assert len(capturado["composition"]) == 20


def test_ventana_mayor_que_historial_no_trunca():
    juez = TTTJuez(ventana=1000, modo_causal_estricto=False)
    _registrar_n_periodos(juez, n=10)
    juez.actualizar()  # no debe lanzar; ventana > longitud real
    assert juez._history is not None


# ---------------------------------------------------------------------------
# metodo='kelly_bayes' y fallback='cash' (H4)
# ---------------------------------------------------------------------------

TRES_ESTRATEGIAS = {"strat_A": (1.5, 0.5), "strat_B": (0.8, 0.3), "strat_C": (0.2, 0.4)}


def test_kelly_bayes_suma_uno_y_no_negativos():
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    pesos = juez.pesos_asignacion("kelly_bayes")
    assert abs(sum(pesos.values()) - 1.0) < 1e-10
    for p in pesos.values():
        assert p >= 0.0


def test_kelly_bayes_penaliza_incertidumbre_mas_que_mu_sobre_sigma():
    """
    Dos estrategias con el mismo mu pero sigma muy distinto: kelly_bayes
    (mu/sigma²) debe favorecer a la de menor sigma MÁS agresivamente que
    mu_sobre_sigma (mu/sigma), porque penaliza sigma cuadráticamente.
    """
    habilidades = {"precisa": (1.0, 0.2), "incierta": (1.0, 2.0)}
    juez = _juez_con_habilidades(habilidades)

    pesos_lineal = juez.pesos_asignacion("mu_sobre_sigma")
    pesos_kelly = juez.pesos_asignacion("kelly_bayes")

    ratio_lineal = pesos_lineal["precisa"] / pesos_lineal["incierta"]
    ratio_kelly = pesos_kelly["precisa"] / pesos_kelly["incierta"]
    assert ratio_kelly > ratio_lineal


def test_fallback_cash_da_exposicion_cero_con_scores_no_positivos():
    habilidades = {"A": (-1.0, 0.5), "B": (-2.0, 0.3)}
    juez = _juez_con_habilidades(habilidades)
    pesos = juez.pesos_asignacion("mu", fallback="cash")
    assert sum(pesos.values()) == 0.0
    for p in pesos.values():
        assert p == 0.0


def test_fallback_uniforme_es_default_y_no_regresivo():
    habilidades = {"A": (-1.0, 0.5), "B": (-2.0, 0.3), "C": (-0.5, 0.4)}
    juez = _juez_con_habilidades(habilidades)
    pesos_default = juez.pesos_asignacion("mu")
    pesos_explicito = juez.pesos_asignacion("mu", fallback="uniforme")
    assert pesos_default == pesos_explicito
    assert abs(sum(pesos_default.values()) - 1.0) < 1e-10


def test_fallback_desconocido_lanza_error():
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    with pytest.raises(ValueError, match="fallback desconocido"):
        juez.pesos_asignacion("mu", fallback="inexistente")


def test_metodo_invalido_menciona_kelly_bayes_en_opciones():
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    with pytest.raises(ValueError, match="kelly_bayes"):
        juez.pesos_asignacion("inexistente")


def test_cash_con_scores_positivos_se_comporta_igual_que_uniforme():
    """Con scores positivos (caso normal), fallback no tiene efecto —
    ninguna rama de fallback se activa."""
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    pesos_uniforme = juez.pesos_asignacion("mu_sobre_sigma", fallback="uniforme")
    pesos_cash = juez.pesos_asignacion("mu_sobre_sigma", fallback="cash")
    assert pesos_uniforme == pesos_cash


# ---------------------------------------------------------------------------
# calibrar_hiperparametros() (conexión con OptimizadorTTT)
# ---------------------------------------------------------------------------

def test_calibrar_hiperparametros_actualiza_sigma_gamma():
    juez = TTTJuez(sigma=1.6, gamma=0.036)
    _registrar_n_periodos(juez, n=25, n_jugadores=4, seed=1)

    sigma_inicial, gamma_inicial = juez.sigma, juez.gamma
    resultado = juez.calibrar_hiperparametros()

    assert set(resultado.keys()) == {"sigma_optimo", "gamma_optimo", "log_evidencia_maxima"}
    assert juez.sigma == resultado["sigma_optimo"]
    assert juez.gamma == resultado["gamma_optimo"]
    assert juez._dirty is True
    assert juez._history is None

    # El juez sigue siendo funcional tras la recalibración.
    habilidades = juez.habilidades_latentes()
    assert len(habilidades) == 4


def test_calibrar_hiperparametros_pocos_periodos_no_cambia_valores():
    juez = TTTJuez(sigma=1.6, gamma=0.036)
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=1.0,
    )
    with pytest.warns(RuntimeWarning):
        resultado = juez.calibrar_hiperparametros()

    assert resultado["sigma_optimo"] == pytest.approx(1.6)
    assert resultado["gamma_optimo"] == pytest.approx(0.036)
    assert juez.sigma == pytest.approx(1.6)
    assert juez.gamma == pytest.approx(0.036)


def test_calibrar_hiperparametros_respeta_bounds():
    juez = TTTJuez(sigma=1.6, gamma=0.036)
    _registrar_n_periodos(juez, n=15, seed=2)
    resultado = juez.calibrar_hiperparametros(bounds=((0.5, 0.6), (0.01, 0.02)))
    assert 0.5 <= resultado["sigma_optimo"] <= 0.6
    assert 0.01 <= resultado["gamma_optimo"] <= 0.02
