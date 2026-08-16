"""[00:00-00:30] Escena nueva v4: glosario expres para audiencia no estadistica."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA, VERDE, ROJO


class Glosario(EscenaBase):
    def construct(self):
        self.setup_voz()

        def tarjeta(palabra, definicion, color):
            p = Text(palabra, font_size=30, weight=BOLD, color=color)
            d = Text(definicion, font_size=18, color=BLACK)
            d.width = min(d.width, 5.5)
            return VGroup(p, d).arrange(DOWN, buff=0.15)

        c1 = tarjeta("SHARPE", "rendimiento ajustado\npor qué tan parejo es", AZUL)
        c2 = tarjeta("HABILIDAD LATENTE", "algo real que no vemos,\nsolo inferimos por resultados", NARANJA)
        c3 = tarjeta("INCERTIDUMBRE", "cuánto confiamos\nen una estimación", VERDE)
        c4 = tarjeta("TTT", "el modelo bayesiano que\njunta las tres anteriores", ROJO)

        tarjetas = VGroup(c1, c2, c3, c4).arrange(RIGHT, buff=0.6)
        tarjetas.width = 12.5

        self.narrar(
            "Antes de arrancar, cuatro palabras que van a aparecer todo "
            "el video. Sharpe: una forma de medir si a una inversión le "
            "fue bien, dividiendo cuánto ganó por cuánto se sacudió para "
            "lograrlo — más alto es mejor.",
            FadeIn(c1),
        )
        self.narrar(
            "Habilidad latente: algo que existe pero no se puede medir "
            "directamente, como la inteligencia de una persona — solo la "
            "inferimos observando resultados.",
            FadeIn(c2),
        )
        self.narrar(
            "Incertidumbre: qué tan seguros estamos de una estimación, "
            "no solo cuál es esa estimación.",
            FadeIn(c3),
        )
        self.narrar(
            "Y TTT, TrueSkill Through Time, el modelo que junta las tres "
            "ideas anteriores para armar un ranking honesto. Con eso "
            "alcanza para seguir todo lo que sigue.",
            FadeIn(c4),
        )
        self.wait(1)
