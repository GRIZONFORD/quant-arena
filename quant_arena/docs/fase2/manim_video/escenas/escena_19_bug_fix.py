"""[10:45-11:15] Captura del bug real corregido: renormalizacion de pesos Kelly."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO, VERDE


class BugFix(EscenaBase):
    def construct(self):
        self.setup_voz()

        codigo = Code(
            code_string=(
                "# _ensamblar_meta (antes del fix)\n"
                "if renormalizar:\n"
                "    pesos = pesos / pesos.sum()  # BUG: anula la exposición\n"
                "                                 # reducida de Kelly"
            ),
            language="python", background="window",
            paragraph_config={"font_size": 20},
        ).scale(0.85)

        etiqueta = Text("bug encontrado y corregido en esta misma tanda",
                         font_size=20, color=ROJO).next_to(codigo, DOWN, buff=0.5)

        self.narrar(
            "Y aquí una honestidad de proceso, no de resultado: al "
            "verificar esto con datos reales encontramos que el motor "
            "renormalizaba los pesos a suma uno incondicionalmente, lo "
            "cual anulaba en la práctica la exposición reducida de "
            "Kelly.",
            Create(codigo), FadeIn(etiqueta),
        )
        self.narrar(
            "Lo detectamos precisamente porque insistimos en correr el "
            "backtest real, no solo los tests unitarios — y lo "
            "corregimos antes de reportar el resultado final.",
            etiqueta.animate.set_color(VERDE),
        )
        self.wait(1)
