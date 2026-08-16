"""[14:10-14:45] Lista de limitaciones abiertas, sin maquillaje."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, NARANJA


class Limitaciones(EscenaBase):
    def construct(self):
        self.setup_voz()

        titulo = self.titulo("Limitaciones abiertas")
        items = VGroup(*[
            Text(f"• {t}", font_size=24, color=BLACK)
            for t in [
                "κ de Kelly sin calibrar",
                "Crowding: diseño validado, ADV estático (no dinámico)",
                "9 de 11 estrategias del Zoo con dependencias ML",
                "aún no auditadas en profundidad",
            ]
        ]).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        items.next_to(titulo, DOWN, buff=0.7)
        items.set_color(NARANJA)

        self.narrar(
            "Y las limitaciones, sin maquillaje: el factor kappa que "
            "conecta la incertidumbre de TTT con la varianza de "
            "retornos necesita calibración empírica que todavía no "
            "hicimos.",
            Write(titulo), FadeIn(items[0]),
        )
        self.narrar(
            "El mecanismo de crowding ya está implementado y con tests, "
            "pero la demanda de volumen diario sigue siendo un snapshot "
            "estático, no dinámico.",
            FadeIn(items[1]),
        )
        self.narrar(
            "Y de las once estrategias del Zoo, solo pudimos auditar en "
            "profundidad las que no dependen de librerías de deep "
            "learning pesadas — el resto queda para el siguiente ciclo.",
            FadeIn(items[2]), FadeIn(items[3]),
        )
        self.wait(1)
