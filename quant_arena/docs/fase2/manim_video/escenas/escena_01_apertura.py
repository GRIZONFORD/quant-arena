"""[00:00-00:30] Curva de equity con recuadro rojo donde el Sharpe rodante colapsa."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, ROJO, GRIS


class Apertura(EscenaBase):
    def construct(self):
        self.setup_voz()

        ax = Axes(x_range=[0, 10, 2], y_range=[0.8, 2.2, 0.4],
                   x_length=10, y_length=5, tips=False,
                   axis_config={"color": GRIS})
        ax.to_edge(DOWN, buff=0.7)

        def equity(x):
            base = 1 + 0.09 * x
            colapso = -0.55 * np.exp(-((x - 6.5) ** 2) / 0.8) if x > 4 else 0
            return base + colapso + 0.05 * np.sin(3 * x)

        curva = ax.plot(equity, color=AZUL, x_range=[0, 10])

        recuadro = Rectangle(width=1.6, height=1.6, color=ROJO, stroke_width=4)
        recuadro.move_to(ax.c2p(6.5, 1.55))

        etiqueta = Text("Sharpe rodante colapsa\nsin aviso", font_size=22, color=ROJO)
        etiqueta.next_to(recuadro, UP, buff=0.2)

        self.narrar(
            "¿Qué pasaría si el modelo que usamos para invertir no solo "
            "midiera resultados pasados, sino que aprendiera en tiempo "
            "real de su propia incertidumbre?",
            Create(ax), Create(curva),
        )
        self.narrar(
            "Toda estrategia cuantitativa tiene una ventana de vigencia. "
            "El problema no es que las estrategias fallen — es que "
            "fallan en momentos distintos, y ninguna medida estática de "
            "qué tan bien le fue nos dice cuándo.",
        )
        self.play(Create(recuadro), FadeIn(etiqueta))
        self.wait(1)
