"""[02:00-02:40] Titulo TTT + tachar terminos incorrectos."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO, VERDE


class TTTTitulo(EscenaBase):
    def construct(self):
        self.setup_voz()

        correcto = Text("TrueSkill Through Time", font_size=40, weight=BOLD, color=VERDE)
        autor = Text("Gustavo Landfried — arXiv:2209.00092", font_size=24, color=BLACK)
        correcto_grupo = VGroup(correcto, autor).arrange(DOWN, buff=0.3)

        incorrecto1 = Text("Test-Time Training", font_size=30, color=ROJO)
        incorrecto2 = Text("\"Landfield\"", font_size=30, color=ROJO)
        incorrectos = VGroup(incorrecto1, incorrecto2).arrange(DOWN, buff=0.4)
        incorrectos.next_to(correcto_grupo, DOWN, buff=1.0)

        tachas = VGroup(*[
            Line(t.get_left(), t.get_right(), color=ROJO, stroke_width=5)
            for t in (incorrecto1, incorrecto2)
        ])

        self.narrar(
            "Aquí es importante ser precisos, porque es el tipo de error "
            "que un tribunal ataca primero: este proyecto usa TrueSkill "
            "Through Time, de Gustavo Landfried — un método bayesiano, es "
            "decir, que no calcula la habilidad de cero cada vez, sino que "
            "actualiza una creencia previa con cada nuevo resultado, usando "
            "un algoritmo llamado Expectation Propagation.",
            Write(correcto_grupo),
        )
        self.narrar(
            "No es Test-Time Training, que es un campo completamente "
            "distinto de inteligencia artificial. Confundirlos no es un "
            "detalle cosmético, es citar mal el método central del "
            "proyecto.",
            FadeIn(incorrectos), Create(tachas),
        )
        self.wait(1)
