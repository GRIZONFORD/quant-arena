"""[00:30-01:00] Split screen: dos curvas de Sharpe rodante cruzandose en 10 anios."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA


class SplitSharpe(EscenaBase):
    def construct(self):
        self.setup_voz()

        ax_izq = Axes(x_range=[0, 10, 2], y_range=[-0.5, 1.5, 0.5],
                       x_length=5.5, y_length=4.5, tips=False).shift(LEFT * 3.3)
        ax_der = Axes(x_range=[0, 10, 2], y_range=[-0.5, 1.5, 0.5],
                       x_length=5.5, y_length=4.5, tips=False).shift(RIGHT * 3.3)

        c1 = ax_izq.plot(lambda x: 0.5 + 0.4 * np.sin(0.9 * x), color=AZUL)
        c2 = ax_izq.plot(lambda x: 0.5 + 0.4 * np.sin(0.9 * x + 1.4), color=NARANJA)
        c3 = ax_der.plot(lambda x: 0.5 + 0.35 * np.cos(1.1 * x), color=AZUL)
        c4 = ax_der.plot(lambda x: 0.5 + 0.35 * np.cos(1.1 * x + 2.1), color=NARANJA)

        divisoria = Line(UP * 3.2, DOWN * 3.2, color=GREY)

        self.narrar(
            "quant-arena parte de esa premisa: si la habilidad no es "
            "estacionaria, medirla con un promedio histórico congelado es, "
            "en el mejor caso, ruido; en el peor, una falsa sensación de "
            "control. Necesitábamos un modelo que tratara la habilidad "
            "como lo que es — una variable latente que cambia con el "
            "tiempo.",
            Create(ax_izq), Create(ax_der), Create(divisoria),
            Create(c1), Create(c2), Create(c3), Create(c4),
        )
        self.wait(1)
