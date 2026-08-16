"""[08:15-08:45] Codigo: GARCHModeler(dist='normal') tachado -> dist='t' condicional."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO, VERDE


class GarchDistT(EscenaBase):
    def construct(self):
        self.setup_voz()

        antes = Code(
            code_string='GARCHModeler(dist="normal")  # hardcodeado',
            language="python", background="window", formatter_style="monokai",
            paragraph_config={"font_size": 22},
        ).scale(0.9)

        despues = Code(
            code_string=(
                "dist = 't' if se_rechaza_normalidad else 'normal'\n"
                "GARCHModeler(dist=dist)  # opt-in, verificado"
            ),
            language="python", background="window", formatter_style="monokai",
            paragraph_config={"font_size": 22},
        ).scale(0.9)

        # Agrupados y centrados como bloque — antes "antes" quedaba en el
        # centro por defecto y "despues" colgaba de él sin anclar el
        # conjunto, dando una composición descentrada.
        bloques = VGroup(antes, despues).arrange(DOWN, buff=0.8)
        bloques.move_to(ORIGIN)

        tacha = Line(antes.get_left(), antes.get_right(), color=ROJO, stroke_width=5)

        self.narrar(
            "La acción, no solo el diagnóstico: cuando se rechaza "
            "normalidad, el modelo ahora cambia automáticamente a "
            "emisiones t de Student en el GARCH, en vez de asumir "
            "gaussianidad por defecto.",
            Create(antes), Create(tacha),
        )
        self.narrar(
            "Es opt-in — no rompe el comportamiento original — pero está "
            "disponible y verificado, no es una promesa.",
            Create(despues), despues.animate.set_color(VERDE),
        )
        self.wait(1)
