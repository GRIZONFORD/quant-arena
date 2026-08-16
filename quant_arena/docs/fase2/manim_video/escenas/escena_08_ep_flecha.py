"""[04:00-04:40] Flecha de una partida reciente ajustando retroactivamente una vieja."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA, VERDE


class EPFlecha(EscenaBase):
    def construct(self):
        self.setup_voz()

        eje = NumberLine(x_range=[0, 10, 1], length=10, color=GREY).shift(DOWN * 0.5)
        etiqueta_eje = Text("tiempo →", font_size=20, color=GREY).next_to(eje, DOWN)

        partidas_x = [1, 3, 5, 7, 9]
        puntos = VGroup(*[
            Dot(eje.n2p(x), color=AZUL, radius=0.12) for x in partidas_x
        ])
        etiquetas = VGroup(*[
            Text(f"t={x}", font_size=16, color=BLACK).next_to(p, UP, buff=0.15)
            for x, p in zip(partidas_x, puntos)
        ])

        flecha_fwd = Arrow(puntos[0].get_center(), puntos[-1].get_center(),
                            color=VERDE, buff=0.2, stroke_width=3).shift(UP * 0.6)
        etq_fwd = Text("mensajes hacia adelante", font_size=18, color=VERDE).next_to(flecha_fwd, UP, buff=0.1)

        flecha_bwd = Arrow(puntos[-1].get_center(), puntos[0].get_center(),
                            color=NARANJA, buff=0.2, stroke_width=3).shift(DOWN * 1.6)
        etq_bwd = Text("mensajes hacia atrás — 'Through Time'", font_size=18, color=NARANJA).next_to(flecha_bwd, DOWN, buff=0.1)

        self.narrar(
            "Y resuelve el problema tres, la memoria: el algoritmo de "
            "Expectation Propagation propaga mensajes hacia adelante y "
            "hacia atrás sobre todo el historial de partidas — 'Through "
            "Time'.",
            Create(eje), FadeIn(etiqueta_eje), FadeIn(puntos), FadeIn(etiquetas),
            GrowArrow(flecha_fwd), FadeIn(etq_fwd),
        )
        self.narrar(
            "Un resultado reciente puede refinar retroactivamente lo que "
            "creíamos saber sobre la habilidad de esa estrategia hace un "
            "año. Ese suavizado bidireccional es la fuente de su poder — "
            "y, como veremos en el minuto 8, también la fuente del riesgo "
            "de look-ahead que tuvimos que blindar explícitamente.",
            GrowArrow(flecha_bwd), FadeIn(etq_bwd),
            Indicate(puntos[0], color=NARANJA, scale_factor=1.5),
        )
        self.wait(1)
