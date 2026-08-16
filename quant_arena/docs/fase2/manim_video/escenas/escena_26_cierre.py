"""[14:45-15:15] Cierre: curva de equity inicial, ahora con overlay aplicado."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, VERDE, GRIS


class Cierre(EscenaBase):
    def construct(self):
        self.setup_voz()

        ax = Axes(x_range=[0, 10, 2], y_range=[0.8, 2.2, 0.4],
                   x_length=10, y_length=5, tips=False,
                   axis_config={"color": GRIS}).to_edge(DOWN, buff=0.7)

        def equity_contenida(x):
            return 1 + 0.09 * x - 0.22 * np.exp(-((x - 6.5) ** 2) / 1.2) + 0.03 * np.sin(3 * x)

        curva = ax.plot(equity_contenida, color=VERDE, x_range=[0, 10])
        etiqueta = Text("con RiskOverlay — drawdown contenido", font_size=22, color=VERDE)
        etiqueta.next_to(ax, UP, buff=0.2)

        self.narrar(
            "La pregunta con la que abrimos era si la habilidad de una "
            "estrategia es estacionaria. La respuesta que construimos no "
            "es un modelo que finge que sí lo es — es un sistema que "
            "modela explícitamente la incertidumbre sobre esa habilidad,",
            Create(ax), Create(curva), FadeIn(etiqueta),
        )
        self.narrar(
            "y que ahora, además, actúa en consecuencia con esa "
            "incertidumbre en cada capa: en el ranking, en el tamaño de "
            "la apuesta, y en el riesgo que tolera.",
        )
        self.wait(1)
