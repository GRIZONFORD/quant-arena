"""[12:50-13:30] Lazo de realimentacion completo: TTT asigna -> alpha decae -> reasigna."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, NARANJA, VERDE, ROJO


class FeedbackLoop(EscenaBase):
    def construct(self):
        self.setup_voz()

        nodos_texto = ["TTT asigna\ncapital", "Alpha decae\npor crowding",
                       "KPI del siguiente\nperíodo baja", "TTT reasigna"]
        colores = [AZUL, NARANJA, ROJO, AZUL]
        nodos = VGroup()
        radio = 2.6
        for i, (txt, col) in enumerate(zip(nodos_texto, colores)):
            angulo = PI / 2 - i * (2 * PI / 4)
            pos = radio * np.array([np.cos(angulo), np.sin(angulo), 0])
            caja = RoundedRectangle(width=2.6, height=1.1, corner_radius=0.15,
                                     color=col, fill_opacity=0.15).move_to(pos)
            texto = Text(txt, font_size=18, color=BLACK).move_to(pos)
            nodos.add(VGroup(caja, texto))

        def punto_borde(centro, hacia, margen=1.0):
            """Punto sobre la línea centro->hacia, a `margen` unidades de
            centro — apenas fuera del cuadro (los nodos adyacentes están
            a ~3.68 de distancia; 1.7 dejaba casi sin flecha, 1.0 despeja
            el borde del cuadro y conserva largo visible)."""
            direccion = hacia - centro
            direccion = direccion / np.linalg.norm(direccion)
            return centro + direccion * margen

        flechas = VGroup(*[
            CurvedArrow(
                punto_borde(nodos[i].get_center(), nodos[(i + 1) % 4].get_center()),
                punto_borde(nodos[(i + 1) % 4].get_center(), nodos[i].get_center()),
                angle=-TAU / 8, color=GREY,
            )
            for i in range(4)
        ])

        equilibrio = Text("→ equilibrio multi-estrategia", font_size=22,
                           color=VERDE, weight=BOLD).to_edge(DOWN, buff=0.4)

        self.narrar(
            "Eso cierra un lazo de realimentación genuino: el Juez "
            "asigna, el alpha se erosiona con la asignación, el KPI del "
            "siguiente período refleja esa erosión, y el Juez reasigna.",
            *[FadeIn(n) for n in nodos], *[Create(f) for f in flechas],
        )
        self.narrar(
            "Ya no hay una sola estrategia ganadora — emerge un "
            "equilibrio. Y es la respuesta directa a la pregunta obvia: "
            "'¿por qué no concentran todo el capital en la mejor "
            "estrategia del ranking?' — porque hacerlo, en un mercado "
            "real, degradaría exactamente la ventaja que la hizo "
            "ganadora.",
            FadeIn(equilibrio),
        )
        self.wait(1)
