"""Helper compartido: tarjeta de transicion de 'acto' (pantalla completa, sin locucion)."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL


class TarjetaActoBase(EscenaBase):
    """Subclases fijan self.numero y self.nombre y usan wait() en vez de narrar
    (son transiciones mudas de ~10-12s, se leen, no se escuchan)."""

    numero = "I"
    nombre = "ACTO"

    def construct(self):
        self.setup_voz()
        numero = Text(f"ACTO {self.numero}", font_size=28, color=AZUL, weight=BOLD)
        nombre = Text(self.nombre, font_size=48, weight=BOLD, color=BLACK)
        linea = Line(LEFT * 2, RIGHT * 2, color=AZUL, stroke_width=3)
        grupo = VGroup(numero, linea, nombre).arrange(DOWN, buff=0.35)

        self.play(FadeIn(numero, shift=UP * 0.2))
        self.play(Create(linea))
        self.play(Write(nombre))
        self.wait(1.2)
        self.play(FadeOut(grupo))
