"""[01:30-02:00] Tres viñetas: los tres problemas del ranking simple."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO


class TresProblemas(EscenaBase):
    def construct(self):
        self.setup_voz()

        v1 = Text("1. Sharpe rodante confunde suerte\ncon habilidad en ventanas cortas",
                   font_size=26, color=BLACK)
        v2 = Text("2. Sin incertidumbre — un Sharpe de 0.8\ncon 20 obs. pesa igual que con 200",
                   font_size=26, color=BLACK)
        v3 = Text("3. Sin memoria — un mal trimestre\nborra un historial de tres años",
                   font_size=26, color=BLACK)
        viñetas = VGroup(v1, v2, v3).arrange(DOWN, buff=0.6, aligned_edge=LEFT)

        self.narrar(
            "Uno: en ventanas cortas, un Sharpe alto puede ser tres apuestas "
            "ganadoras seguidas, no una ventaja real.",
            FadeIn(v1, shift=UP * 0.3),
        )
        self.narrar(
            "Dos: el ranking simple no distingue una estimación precisa de "
            "una ruidosa — trata igual un Sharpe de 0.8 con veinte "
            "observaciones que con doscientas.",
            FadeIn(v2, shift=UP * 0.3),
        )
        self.narrar(
            "Tres: no tiene memoria estructurada — un trimestre malo puede "
            "borrar tres años de evidencia acumulada de que una estrategia "
            "es genuinamente buena.",
            FadeIn(v3, shift=UP * 0.3),
        )
        self.play(viñetas.animate.set_color(ROJO))
        self.wait(0.5)
