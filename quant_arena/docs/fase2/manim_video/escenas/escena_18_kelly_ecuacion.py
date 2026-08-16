"""[10:00-10:45] Ecuacion de Kelly bayesiano con sigma^2_skill,TTT en el denominador."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA


class KellyEcuacion(EscenaBase):
    def construct(self):
        self.setup_voz()

        formula = MathTex(
            r"f_{\text{Kelly}} = \lambda \cdot "
            r"\frac{\mu_{\text{edge}}}"
            r"{\sigma^2_{\text{retornos}} + \kappa \cdot \sigma^2_{\text{skill,TTT}}}",
            font_size=52,
        )
        formula.set_color_by_tex(r"\sigma^2_{\text{skill,TTT}}", NARANJA)

        self.narrar(
            "El segundo hallazgo era más sutil: los pesos del Juez "
            "siempre suman uno y siempre se invierten al cien por "
            "ciento, sin importar cuánta convicción tenga realmente el "
            "modelo.",
            Write(formula),
        )
        self.narrar(
            "La corrección es Kelly bayesiano: la fracción óptima de "
            "capital es mu sobre sigma al cuadrado — pero aquí sigma al "
            "cuadrado no es solo la varianza de los retornos. Sumamos la "
            "varianza del posterior de habilidad que TTT ya calcula. "
            "Cuando el Juez tiene poca convicción en una estrategia, esa "
            "incertidumbre penaliza directamente el tamaño de la "
            "apuesta.",
            Indicate(formula, color=NARANJA, scale_factor=1.1),
        )
        self.wait(1)
