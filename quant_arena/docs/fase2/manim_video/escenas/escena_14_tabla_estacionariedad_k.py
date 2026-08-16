"""[07:45-08:15] Estacionariedad/independencia + comparacion K=3 vs K=4."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, VERDE, ROJO, NARANJA


class TablaKOptimo(EscenaBase):
    def construct(self):
        self.setup_voz()

        tabla = Table(
            [["Estacionariedad", "ADF + KPSS", "NO se rechaza"],
             ["Independencia (lag=20)", "Ljung-Box", "SE RECHAZA"]],
            col_labels=[Text(c, font_size=20, weight=BOLD) for c in
                        ["Test", "Método", "Resultado"]],
            include_outer_lines=True,
        ).scale(0.55).to_edge(UP, buff=1.2)

        comparacion = VGroup(
            Text("K=3", font_size=32, color=ROJO, weight=BOLD),
            Text("(hardcodeado, código original)", font_size=18, color=BLACK),
            Text("→", font_size=32, color=BLACK),
            Text("K=4", font_size=32, color=VERDE, weight=BOLD),
            Text("(óptimo: BIC + AIC + CV temporal)", font_size=18, color=BLACK),
        )
        fila1 = VGroup(comparacion[0], comparacion[1]).arrange(RIGHT, buff=0.3)
        fila3 = VGroup(comparacion[3], comparacion[4]).arrange(RIGHT, buff=0.3)
        comp_grupo = VGroup(fila1, comparacion[2], fila3).arrange(DOWN, buff=0.4)
        comp_grupo.next_to(tabla, DOWN, buff=0.8)

        self.narrar(
            "La estacionariedad no se rechaza — coherente con que ya "
            "trabajamos en log-retornos.",
            Create(tabla),
        )
        self.narrar(
            "Y encontramos algo más interesante: el número de regímenes "
            "ocultos del HMM estaba fijo en 3 en el código original, sin "
            "ningún criterio de selección.",
            FadeIn(fila1),
        )
        self.narrar(
            "Corrimos un barrido de K con BIC, AIC y validación cruzada "
            "temporal — las tres métricas coinciden en que el K óptimo "
            "sobre el historial completo es 4, no 3.",
            FadeIn(comparacion[2]), FadeIn(fila3),
        )
        self.wait(1)
