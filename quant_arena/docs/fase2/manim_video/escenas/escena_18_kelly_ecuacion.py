"""[10:00-10:45] Ecuacion de Kelly bayesiano con sigma^2_skill,TTT en el denominador.

Reestructurada: primero un ejemplo numerico simple del criterio de Kelly
(apuesta de moneda sesgada), despues se conecta con la formula real del
proyecto — pedido explicito de explicar el concepto antes de aplicarlo.
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA, VERDE, NEGRO


class KellyEcuacion(EscenaBase):
    def construct(self):
        self.setup_voz()

        problema = Text(
            "El dinero se repartía sumando 100%,\nsin importar la convicción real del modelo.",
            font_size=26, color=NEGRO,
        )

        # --- Ejemplo simple, antes de la fórmula del proyecto ---
        ejemplo_titulo = Text("Un ejemplo simple primero", font_size=30, weight=BOLD, color=AZUL)
        ejemplo_enunciado = Text(
            "Una apuesta que ganás el 60% de las veces,\nduplicando lo que arriesgás en cada una.",
            font_size=24, color=NEGRO,
        )
        ejemplo_formula = MathTex(
            r"f^* = 2p - 1 = 2(0.6) - 1 = 0.20",
            font_size=40, color=VERDE,
        )
        ejemplo_conclusion = Text(
            "20% del capital en cada apuesta — ni todo\n(te arruinás tarde o temprano), ni poco\n(dejás plata sobre la mesa).",
            font_size=22, color=NEGRO,
        )
        ejemplo = VGroup(
            ejemplo_titulo, ejemplo_enunciado, ejemplo_formula, ejemplo_conclusion
        ).arrange(DOWN, buff=0.4)

        # --- La fórmula real del proyecto ---
        formula = MathTex(
            r"f_{\text{Kelly}} = \lambda \cdot "
            r"\frac{\mu_{\text{edge}}}"
            r"{\sigma^2_{\text{retornos}} + \kappa \cdot \sigma^2_{\text{skill,TTT}}}",
            font_size=48,
        )
        formula.set_color_by_tex(r"\sigma^2_{\text{skill,TTT}}", NARANJA)

        self.narrar(
            "El segundo hallazgo era más sutil: el dinero siempre se "
            "repartía sumando cien por ciento entre las estrategias, "
            "sin importar cuánta convicción tuviera realmente el "
            "modelo.",
            Write(problema),
        )
        self.narrar(
            "Un ejemplo simple antes de ir a la fórmula real: imaginen "
            "una apuesta que ganan el sesenta por ciento de las veces, "
            "duplicando lo que arriesgan en cada una.",
            FadeOut(problema), FadeIn(ejemplo_titulo), FadeIn(ejemplo_enunciado),
        )
        self.narrar(
            "El criterio de Kelly dice: arriesguen dos veces esa "
            "probabilidad, menos uno — acá, veinte por ciento del "
            "capital en cada apuesta. Ni todo, porque tarde o temprano "
            "se arruinan; ni poco, porque dejan plata sobre la mesa.",
            Write(ejemplo_formula), FadeIn(ejemplo_conclusion),
        )
        self.narrar(
            "Así lo aplicamos acá: la fracción óptima de capital es la "
            "ventaja estimada, dividida por qué tan riesgosa es esa "
            "ventaja. Pero esta vez el riesgo no es solo la variabilidad "
            "de los retornos — sumamos también la incertidumbre del "
            "posterior de habilidad que TTT ya calcula.",
            FadeOut(ejemplo), Write(formula),
        )
        self.narrar(
            "Cuando el Juez tiene poca convicción en una estrategia, esa "
            "incertidumbre penaliza directamente cuánto se le arriesga.",
            Indicate(formula, color=NARANJA, scale_factor=1.1),
        )
        self.wait(1)
