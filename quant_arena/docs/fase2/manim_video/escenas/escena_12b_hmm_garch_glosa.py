"""[07:40-08:10] Escena nueva v4: explicacion en criollo de HMM y GARCH."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA


class HMMGarchGlosa(EscenaBase):
    def construct(self):
        self.setup_voz()

        hmm = VGroup(
            Text("HMM (Hidden Markov Model)", font_size=26, weight=BOLD, color=AZUL),
            Text("El mercado pasa por 'estados' ocultos\n(calma, pánico) que no vemos directo,\npero podemos inferir.", font_size=20, color=BLACK),
        ).arrange(DOWN, buff=0.25)

        garch = VGroup(
            Text("GARCH", font_size=26, weight=BOLD, color=NARANJA),
            Text("Predice cuánto va a moverse el precio\nmañana, mirando cuánto se movió\nrecientemente.", font_size=20, color=BLACK),
        ).arrange(DOWN, buff=0.25)

        grupo = VGroup(hmm, garch).arrange(RIGHT, buff=1.2)

        self.narrar(
            "Un momento para explicar dos términos que van a aparecer "
            "seguido. Un HMM, o modelo de Markov oculto, asume que el "
            "mercado pasa por distintos 'estados de ánimo' — calma, "
            "pánico, tendencia — que no observamos directamente, pero "
            "podemos inferir mirando cómo se comportan los precios.",
            FadeIn(hmm),
        )
        self.narrar(
            "Un GARCH es un modelo que predice cuánto va a variar el "
            "precio mañana, mirando cuánto varió en los días recientes "
            "— la idea de que la volatilidad viene en rachas, no es "
            "constante.",
            FadeIn(garch),
        )
        self.wait(1)
