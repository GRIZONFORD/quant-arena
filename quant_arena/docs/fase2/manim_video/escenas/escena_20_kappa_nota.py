"""[11:15-11:30] Texto: kappa requiere calibracion empirica."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase


class KappaNota(EscenaBase):
    def construct(self):
        self.setup_voz()
        self.tarjeta_simple(
            "σ_skill,TTT vive en escala ~1.6\nσ_retornos vive en escala ~1e-4\n"
            "κ requiere calibración empírica",
            "Con el kappa por defecto, la penalización es agresiva — el "
            "modelo se vuelve muy conservador porque la escala de "
            "incertidumbre de TTT no está calibrada contra la escala de "
            "los retornos. Lo documentamos como limitación abierta, no "
            "como resultado final.",
            font_size=28,
        )
