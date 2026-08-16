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
            "El segundo hallazgo era más sutil: el dinero siempre se "
            "repartía sumando cien por ciento entre las estrategias, "
            "sin importar cuánta convicción tuviera realmente el "
            "modelo.",
            Write(formula),
        )
        self.narrar(
            "La corrección usa el criterio de Kelly — una fórmula "
            "clásica de teoría de apuestas, que dice qué fracción de tu "
            "capital arriesgar dado cuánta ventaja creés tener y cuánta "
            "certeza tenés de esa ventaja. Acá la certeza no viene solo "
            "de la variabilidad de los retornos — sumamos también la "
            "incertidumbre del posterior de habilidad que TTT ya "
            "calcula. Cuando el Juez tiene poca convicción en una "
            "estrategia, esa incertidumbre penaliza directamente cuánto "
            "se le arriesga.",
            Indicate(formula, color=NARANJA, scale_factor=1.1),
        )
        self.wait(1)
