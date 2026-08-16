"""[12:10-12:50] Ecuacion de decaimiento del alpha efectivo por crowding."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO


class CrowdingEcuacion(EscenaBase):
    def construct(self):
        self.setup_voz()

        formula = MathTex(
            r"\alpha_{\text{efectivo}}(t) = \alpha_{\text{bruto}}(t) \cdot "
            r"e^{-\kappa \cdot w_i(t) \cdot \text{AUM}/\text{ADV}_i}",
            font_size=48,
        )
        formula.set_color_by_tex("AUM", ROJO)

        glosa = Text(
            "AUM: plata invertida en la estrategia.\n"
            "ADV: cuánto de ese activo se mueve\npor día en el mercado.",
            font_size=20, color=BLACK,
        ).next_to(formula, DOWN, buff=0.6)

        self.narrar(
            "El mecanismo propuesto es lo que se llama crowding — "
            "amontonamiento: la ventaja real de una estrategia decae "
            "exponencialmente con la fracción de capital que recibe, "
            "relativa al volumen que ese activo mueve en un día "
            "normal.",
            Write(formula),
        )
        self.narrar(
            "Cuanto más grande la apuesta comparada con ese volumen "
            "diario, más se nota la propia operación en el precio, y "
            "menos ventaja real queda. Es el mismo principio de "
            "rendimientos decrecientes a escala documentado por Berk y "
            "Green — cuanto más capital persigue la misma ventaja, "
            "menor rendimiento por unidad invertida.",
            FadeIn(glosa), Indicate(formula, scale_factor=1.1),
        )
        self.wait(1)
