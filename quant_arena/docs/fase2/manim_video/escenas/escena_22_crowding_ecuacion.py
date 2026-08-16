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

        self.narrar(
            "El mecanismo propuesto es crowding: el alpha efectivo de "
            "una estrategia decae exponencialmente con la fracción de "
            "capital que recibe, relativa al volumen diario del activo "
            "que opera.",
            Write(formula),
        )
        self.narrar(
            "Es el mismo principio de rendimientos decrecientes a "
            "escala documentado por Berk y Green — cuanto más capital "
            "persigue la misma ventaja, menor rendimiento marginal por "
            "unidad invertida.",
            Indicate(formula, scale_factor=1.1),
        )
        self.wait(1)
