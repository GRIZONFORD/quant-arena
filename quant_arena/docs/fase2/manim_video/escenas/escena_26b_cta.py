"""[16:20-16:35] CTA nueva en v5: cierre con invitacion explicita, no solo reflexion."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL


class CTA(EscenaBase):
    def construct(self):
        self.setup_voz()

        frase = Text(
            "La incertidumbre no se ignora:\nse modela, y se usa.",
            font_size=36, weight=BOLD, color=AZUL,
        )

        self.narrar(
            "La estadística bayesiana no es solo teoría: es una "
            "herramienta que puede redefinir cómo gestionamos riesgo y "
            "capital en mercados reales.",
            Write(frase),
        )
        self.narrar(
            "Los invito a mirar el código, cuestionar los supuestos, y "
            "decidir ustedes mismos dónde está el límite entre lo que "
            "ya funciona y lo que todavía falta calibrar.",
        )
        self.wait(1)
