# =============================================================================
# FILE: quant_arena/tests/conftest.py
# Stub de trueskillthroughtime para ejecutar la suite sin la dependencia real.
# El stub se inyecta en sys.modules ANTES de que cualquier módulo de test
# importe ttt_juez.py, garantizando que el import de nivel de módulo
# "from trueskillthroughtime import History, Player, Gaussian" no falle.
# =============================================================================
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _inyectar_stub_ttt() -> None:
    """Registra un stub mínimo de trueskillthroughtime si no está instalado."""
    try:
        import trueskillthroughtime  # noqa: F401 — verifica instalación real
        return  # paquete real disponible; no hacer nada
    except ImportError:
        pass

    class _Gaussian:
        """Gaussiana mínima compatible con la API que usa ttt_juez.py."""
        def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
            self.mu = float(mu)
            self.sigma = float(sigma)

        def __repr__(self) -> str:
            return f"Gaussian(mu={self.mu}, sigma={self.sigma})"

    stub = ModuleType("trueskillthroughtime")
    stub.Gaussian = _Gaussian          # type: ignore[attr-defined]
    stub.Player = MagicMock()          # type: ignore[attr-defined]
    stub.History = MagicMock()         # type: ignore[attr-defined]
    sys.modules["trueskillthroughtime"] = stub


# Ejecutar al cargar conftest.py (antes de la recolección de tests)
_inyectar_stub_ttt()
