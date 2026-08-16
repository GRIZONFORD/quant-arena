"""[03:10-04:00] Curva gaussiana N(mu, sigma^2) angostandose con mas partidas."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL


class Gaussianas(EscenaBase):
    def construct(self):
        self.setup_voz()

        ax = Axes(x_range=[-4, 4, 1], y_range=[0, 1.2, 0.5],
                   x_length=9, y_length=4.5, tips=False).add_coordinates()
        titulo = self.titulo("Creencia sobre la habilidad de una estrategia")

        def gaussiana(mu, sigma):
            return ax.plot(
                lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color=AZUL, x_range=[-4, 4],
            )

        curva_ancha = gaussiana(mu=0, sigma=1.5)
        curva_angosta = gaussiana(mu=0.8, sigma=0.4)

        self.narrar(
            "TrueSkill Through Time mantiene, para cada estrategia, una "
            "creencia — no un número, una distribución de probabilidad "
            "con forma de campana. El centro de la campana, mu, es la "
            "habilidad estimada; qué tan ancha es, sigma, es cuánta "
            "incertidumbre tenemos sobre esa estimación.",
            Write(titulo), Create(ax), Create(curva_ancha),
        )
        self.narrar(
            "Con pocas partidas, sigma es grande — el modelo admite que no "
            "sabe. Con más evidencia, se angosta. Eso ya resuelve el "
            "problema dos del ranking ingenuo: la incertidumbre es "
            "explícita, no implícita.",
            Transform(curva_ancha, curva_angosta),
        )
        self.wait(1)
