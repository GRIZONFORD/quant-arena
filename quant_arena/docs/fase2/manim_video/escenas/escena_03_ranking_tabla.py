"""[01:00-01:30] Tabla de ranking simple reordenandose violentamente mes a mes."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL


class RankingTabla(EscenaBase):
    def construct(self):
        self.setup_voz()

        nombres = ["Momentum", "OLPS-RMR", "XGBoost", "HMM-GARCH", "TFT"]
        orden_1 = [0, 1, 2, 3, 4]
        orden_2 = [3, 0, 4, 1, 2]
        orden_3 = [2, 4, 0, 3, 1]

        def fila(nombre, idx):
            r = RoundedRectangle(width=4, height=0.7, corner_radius=0.1,
                                  color=AZUL, fill_opacity=0.15)
            t = Text(f"{idx+1}. {nombre}", font_size=24, color=BLACK)
            t.move_to(r)
            return VGroup(r, t)

        filas = VGroup(*[fila(n, i) for i, n in enumerate(nombres)])
        filas.arrange(DOWN, buff=0.15).to_edge(LEFT, buff=1.5)

        self.narrar(
            "La solución ingenua es rankear por Sharpe rodante y asignar "
            "capital al líder. Es simple, y es frágil por tres razones "
            "concretas, no genéricas.",
            FadeIn(filas),
        )

        for orden in (orden_2, orden_3, orden_1):
            objetivo = VGroup(*[filas[i] for i in orden]).arrange(DOWN, buff=0.15)
            objetivo.move_to(filas)
            self.play(*[
                filas[i].animate.move_to(objetivo[pos])
                for pos, i in enumerate(orden)
            ], run_time=0.8)
        self.wait(0.5)
