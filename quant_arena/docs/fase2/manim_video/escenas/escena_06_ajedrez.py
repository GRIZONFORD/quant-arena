"""[02:40-03:10] Diagrama de ajedrez conceptual: jugadores = estrategias."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA


class Ajedrez(EscenaBase):
    def construct(self):
        self.setup_voz()

        jugador1 = VGroup(Circle(radius=0.5, color=AZUL, fill_opacity=0.3),
                           Text("Momentum", font_size=18, color=BLACK)).arrange(DOWN, buff=0.1)
        jugador2 = VGroup(Circle(radius=0.5, color=NARANJA, fill_opacity=0.3),
                           Text("OLPS-RMR", font_size=18, color=BLACK)).arrange(DOWN, buff=0.1)
        jugador1.shift(LEFT * 3)
        jugador2.shift(RIGHT * 3)

        vs = Text("PARTIDA (rebalanceo)", font_size=22, color=GREY).next_to(
            VGroup(jugador1, jugador2), UP, buff=0.8)
        flecha = DoubleArrow(jugador1.get_right(), jugador2.get_left(), color=GREY)

        resultado = Text("Sharpe rodante decide quién gana esta partida",
                          font_size=22, color=BLACK).to_edge(DOWN, buff=1)

        self.narrar(
            "La analogía correcta es un sistema de ranking de ajedrez. "
            "Cada estrategia es un jugador. Cada período de rebalanceo es "
            "una partida.",
            FadeIn(jugador1), FadeIn(jugador2), FadeIn(vs), Create(flecha),
        )
        self.narrar(
            "El Sharpe rodante decide el resultado de esa partida — quién "
            "quedó primero, quién último. Y como en el ajedrez, lo que nos "
            "interesa no es el resultado de una partida aislada, sino la "
            "habilidad que ese resultado revela.",
            FadeIn(resultado),
        )
        self.wait(1)
