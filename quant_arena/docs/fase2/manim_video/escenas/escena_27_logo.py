"""[15:15-15:30] Fade a logo del proyecto."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL


class Logo(EscenaBase):
    def construct(self):
        self.setup_voz()

        nombre = Text("quant-arena", font_size=56, weight=BOLD, color=AZUL)
        referencia = Text("TrueSkill Through Time — G. Landfried, arXiv:2209.00092",
                           font_size=22, color=BLACK)
        grupo = VGroup(nombre, referencia).arrange(DOWN, buff=0.4)

        self.narrar(
            "TrueSkill Through Time, de Gustavo Landfried. quant-arena.",
            Write(grupo),
        )
        self.wait(2)
