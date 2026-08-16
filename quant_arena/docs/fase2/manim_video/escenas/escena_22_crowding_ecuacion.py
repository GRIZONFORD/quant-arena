"""[12:10-12:50] Ecuacion de decaimiento del alpha efectivo por crowding.

Reestructurada con una analogia antes de la formula (pedido explicito:
la explicacion original no se entendia, tiraba la ecuacion sin preparar
el concepto).
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO, AZUL, NEGRO


class CrowdingEcuacion(EscenaBase):
    def construct(self):
        self.setup_voz()

        analogia = Text(
            "Un recital que vende 1,000 boletos por día.\n"
            "Si intentás comprar 10,000 de una,\n"
            "el precio se dispara antes de que termines.",
            font_size=26, color=NEGRO,
        )

        traduccion = Text(
            "Lo mismo le pasa a una estrategia: si mete\n"
            "más plata de la que el mercado absorbe en un día,\n"
            "su propia orden mueve el precio en contra suyo.",
            font_size=24, color=AZUL,
        )

        formula = MathTex(
            r"\alpha_{\text{efectivo}}(t) = \alpha_{\text{bruto}}(t) \cdot "
            r"e^{-\kappa \cdot w_i(t) \cdot \text{AUM}/\text{ADV}_i}",
            font_size=44,
        )
        formula.set_color_by_tex("AUM", ROJO)

        glosa = Text(
            "AUM: cuánta plata metiste. ADV: cuántos\n"
            "'boletos' vende el mercado por día. Cuando\n"
            "AUM se acerca a ADV, el alpha efectivo se derrumba.",
            font_size=20, color=NEGRO,
        ).next_to(formula, DOWN, buff=0.6)

        self.narrar(
            "El mecanismo propuesto es lo que se llama crowding — "
            "amontonamiento. Piensen en un recital que vende mil "
            "boletos por día. Si alguien intenta comprar diez mil de "
            "una sola vez, el precio se dispara antes de que termine "
            "de comprar — se estorba a sí mismo.",
            Write(analogia),
        )
        self.narrar(
            "Eso mismo le pasa a una estrategia cuantitativa: si le "
            "meten más plata de la que el mercado puede absorber en un "
            "día, la propia operación mueve el precio en contra, y la "
            "ventaja que tenían se evapora.",
            FadeOut(analogia), FadeIn(traduccion),
        )
        self.narrar(
            "Formalizado, la ventaja real de una estrategia decae "
            "exponencialmente con la fracción de capital que recibe, "
            "relativa al volumen diario del activo que opera.",
            FadeOut(traduccion), Write(formula),
        )
        self.narrar(
            "AUM es cuánta plata metieron; ADV es cuántos 'boletos' "
            "vende el mercado por día. Cuando AUM se acerca a ADV, la "
            "ventaja efectiva se derrumba — es el mismo principio de "
            "rendimientos decrecientes a escala documentado por Berk y "
            "Green.",
            FadeIn(glosa), Indicate(formula, scale_factor=1.1),
        )
        self.wait(1)
