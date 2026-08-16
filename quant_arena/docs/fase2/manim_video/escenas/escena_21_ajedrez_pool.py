"""[11:30-12:10] El diagrama de ajedrez regresa con flechas de competencia por capital."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA, VERDE


class AjedrezPool(EscenaBase):
    def construct(self):
        self.setup_voz()

        titulo = self.titulo("§1.4 — Próximo hito (no implementado al escribir el guión)")

        pool = Circle(radius=0.8, color=GREY, fill_opacity=0.2).shift(DOWN * 0.3)
        pool_txt = Text("Capital", font_size=20, color=BLACK).move_to(pool)

        colores = [AZUL, NARANJA, VERDE]
        jugadores = VGroup(*[
            Circle(radius=0.4, color=c, fill_opacity=0.3)
            for c in colores
        ])
        posiciones = [UP * 2.5 + LEFT * 3, UP * 2.5, UP * 2.5 + RIGHT * 3]
        for j, p in zip(jugadores, posiciones):
            j.move_to(p)

        flechas = VGroup(*[
            Arrow(j.get_bottom(), pool.get_top(), color=j.get_color(), buff=0.15)
            for j in jugadores
        ])

        self.narrar(
            "Todo lo anterior corrige cómo medimos y cómo dimensionamos. "
            "Pero hasta acá, el ranking sigue siendo eso: un ranking.",
            Write(titulo), FadeIn(jugadores), FadeIn(pool), FadeIn(pool_txt),
        )
        self.narrar(
            "Las once estrategias no interactúan entre sí — TTT las "
            "puntúa, pero no hay ninguna fuerza económica que impida "
            "concentrar todo el capital en una sola. El siguiente hito, "
            "todavía en diseño, cierra esa brecha.",
            *[GrowArrow(f) for f in flechas],
        )
        self.wait(1)
