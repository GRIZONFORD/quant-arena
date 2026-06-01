# =============================================================================
# FILE: quant_arena/tests/test_juez.py
# Suite pytest para TTTJuez — sin requerir trueskillthroughtime instalado.
#
# Estrategia de aislamiento:
#   conftest.py inyecta un stub en sys.modules["trueskillthroughtime"] ANTES
#   de que este módulo se importe, por lo que ttt_juez.py puede importarse
#   libremente. Los tests que verifican pesos_asignacion() inyectan un
#   _history mock directamente para evitar llamar a History.convergence().
#
# Cobertura:
#   - pesos_asignacion() suma exactamente 1.0 para los 3 métodos
#   - fallback equal-weight cuando todos los scores raw son <= 0
#   - todos los pesos son >= 0
#   - softmax_mu garantiza peso > 0 incluso con mu negativos
#   - registrar_periodo() ignora periodos con < 2 estrategias válidas
#   - habilidades_latentes() retorna dict vacío si no hay datos
#   - pesos_asignacion() retorna dict vacío si no hay habilidades
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from quant_arena.juez.ttt_juez import TTTJuez
from quant_arena.core.abstracciones import MetricasResultado


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _G:
    """Gaussiana mínima: sólo .mu y .sigma, igual que el stub de conftest."""
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)


def _juez_con_habilidades(habilidades: dict[str, tuple[float, float]]) -> TTTJuez:
    """
    Construye un TTTJuez con _history mock precargado.
    `habilidades` es {nombre: (mu, sigma)}.
    """
    juez = TTTJuez()

    mock_history = MagicMock()
    mock_history.learning_curves.return_value = {
        nombre: [(1.0, _G(mu, sigma))]
        for nombre, (mu, sigma) in habilidades.items()
    }
    mock_history.log_evidence.return_value = -10.0

    juez._history = mock_history
    juez._dirty = False
    return juez


# ---------------------------------------------------------------------------
# pesos_asignacion() — suma = 1.0
# ---------------------------------------------------------------------------

TRES_ESTRATEGIAS = {"strat_A": (1.5, 0.5), "strat_B": (0.8, 0.3), "strat_C": (0.2, 0.4)}


@pytest.mark.parametrize("metodo", ["mu_sobre_sigma", "mu", "softmax_mu"])
def test_pesos_suman_uno_caso_normal(metodo: str) -> None:
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    pesos = juez.pesos_asignacion(metodo)
    assert abs(sum(pesos.values()) - 1.0) < 1e-10, (
        f"[{metodo}] suma = {sum(pesos.values()):.15f}"
    )


@pytest.mark.parametrize("metodo", ["mu_sobre_sigma", "mu", "softmax_mu"])
def test_todos_los_pesos_no_negativos(metodo: str) -> None:
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    pesos = juez.pesos_asignacion(metodo)
    for nombre, p in pesos.items():
        assert p >= 0.0, f"[{metodo}] peso de '{nombre}' = {p} < 0"


@pytest.mark.parametrize("metodo", ["mu_sobre_sigma", "mu", "softmax_mu"])
def test_pesos_suman_uno_con_dos_estrategias(metodo: str) -> None:
    juez = _juez_con_habilidades({"X": (2.0, 1.0), "Y": (0.5, 0.5)})
    pesos = juez.pesos_asignacion(metodo)
    assert len(pesos) == 2
    assert abs(sum(pesos.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Fallback equal-weight: scores raw todos <= 0
# ---------------------------------------------------------------------------

def test_fallback_equal_weight_cuando_todos_mu_negativos() -> None:
    """
    Con método 'mu', si todos los mu < 0 → max(v, 0) = 0 para todos →
    total = 0 → fallback a 1/N por estrategia.
    """
    habilidades = {"A": (-1.0, 0.5), "B": (-2.0, 0.3), "C": (-0.5, 0.4)}
    juez = _juez_con_habilidades(habilidades)
    pesos = juez.pesos_asignacion("mu")

    assert abs(sum(pesos.values()) - 1.0) < 1e-10
    for p in pesos.values():
        assert abs(p - 1 / 3) < 1e-10, f"peso esperado 1/3, obtenido {p}"


def test_fallback_equal_weight_cuando_todos_mu_sobre_sigma_negativos() -> None:
    """
    Con método 'mu_sobre_sigma' y todos mu < 0: igual fallback a 1/N.
    """
    habilidades = {"A": (-0.5, 0.1), "B": (-1.0, 0.2)}
    juez = _juez_con_habilidades(habilidades)
    pesos = juez.pesos_asignacion("mu_sobre_sigma")

    assert abs(sum(pesos.values()) - 1.0) < 1e-10
    for p in pesos.values():
        assert abs(p - 0.5) < 1e-10


def test_softmax_mu_da_pesos_positivos_con_todos_mu_negativos() -> None:
    """
    softmax nunca produce peso = 0, incluso si todos los mu son muy negativos.
    """
    habilidades = {"A": (-5.0, 0.5), "B": (-10.0, 0.3), "C": (-3.0, 0.4)}
    juez = _juez_con_habilidades(habilidades)
    pesos = juez.pesos_asignacion("softmax_mu")

    assert abs(sum(pesos.values()) - 1.0) < 1e-10
    for p in pesos.values():
        assert p > 0.0


def test_softmax_mu_con_mu_mixtos_suman_uno() -> None:
    habilidades = {"A": (3.0, 0.8), "B": (-1.0, 0.5), "C": (0.5, 0.3)}
    juez = _juez_con_habilidades(habilidades)
    pesos = juez.pesos_asignacion("softmax_mu")
    assert abs(sum(pesos.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Método inválido
# ---------------------------------------------------------------------------

def test_metodo_invalido_lanza_error() -> None:
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    with pytest.raises(ValueError, match="inexistente"):
        juez.pesos_asignacion("inexistente")


# ---------------------------------------------------------------------------
# Sin habilidades
# ---------------------------------------------------------------------------

def test_pesos_asignacion_sin_historia_retorna_vacio() -> None:
    juez = TTTJuez()
    # _history = None y _composition = [] → actualizar() retorna sin crear History
    assert juez.pesos_asignacion() == {}


def test_habilidades_latentes_sin_datos_retorna_vacio() -> None:
    juez = TTTJuez()
    assert juez.habilidades_latentes() == {}


# ---------------------------------------------------------------------------
# registrar_periodo()
# ---------------------------------------------------------------------------

def test_registrar_periodo_ignora_menos_de_dos_estrategias_validas() -> None:
    juez = TTTJuez()
    juez.registrar_periodo(
        {"solo_una": MetricasResultado(sharpe=1.5)},
        tiempo=1000.0,
    )
    assert len(juez._composition) == 0


def test_registrar_periodo_ignora_un_nan_y_un_valido() -> None:
    juez = TTTJuez()
    juez.registrar_periodo(
        {
            "valida": MetricasResultado(sharpe=1.0),
            "invalida": MetricasResultado(sharpe=np.nan),
        },
        tiempo=1000.0,
    )
    assert len(juez._composition) == 0


def test_registrar_periodo_acepta_dos_validas() -> None:
    juez = TTTJuez()
    juez.registrar_periodo(
        {
            "A": MetricasResultado(sharpe=1.5),
            "B": MetricasResultado(sharpe=0.8),
        },
        tiempo=1000.0,
    )
    assert len(juez._composition) == 1
    assert juez._dirty is True


def test_registrar_periodo_ordena_mejor_primero() -> None:
    """El ganador (mayor sharpe) debe quedar en el índice 0 de la composition."""
    juez = TTTJuez()
    juez.registrar_periodo(
        {
            "peor": MetricasResultado(sharpe=-0.5),
            "mejor": MetricasResultado(sharpe=2.0),
            "medio": MetricasResultado(sharpe=0.5),
        },
        tiempo=1000.0,
    )
    game = juez._composition[0]
    assert game[0] == ["mejor"]
    assert game[-1] == ["peor"]


def test_registrar_periodo_vacio_no_modifica_estado() -> None:
    juez = TTTJuez()
    juez.registrar_periodo({}, tiempo=1000.0)
    assert len(juez._composition) == 0
    assert juez._dirty is False


# ---------------------------------------------------------------------------
# Propiedades de estado
# ---------------------------------------------------------------------------

def test_dirty_flag_se_activa_al_registrar() -> None:
    juez = TTTJuez()
    juez._history = MagicMock()  # simula history ya calculado
    juez._dirty = False
    juez.registrar_periodo(
        {"A": MetricasResultado(sharpe=1.0), "B": MetricasResultado(sharpe=0.5)},
        tiempo=1.0,
    )
    assert juez._dirty is True
    assert juez._history is None  # se invalida el cache


def test_pesos_consistentes_entre_llamadas_consecutivas() -> None:
    """Dos llamadas a pesos_asignacion() con mismo estado dan el mismo resultado."""
    juez = _juez_con_habilidades(TRES_ESTRATEGIAS)
    pesos_1 = juez.pesos_asignacion("mu_sobre_sigma")
    pesos_2 = juez.pesos_asignacion("mu_sobre_sigma")
    for nombre in pesos_1:
        assert pesos_1[nombre] == pytest.approx(pesos_2[nombre], abs=1e-12)
