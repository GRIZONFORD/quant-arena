"""[05:50-06:30] Patron Adapter: TTTJuez(AbstractJuez)."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, VERDE, GRIS


class Adapter(EscenaBase):
    def construct(self):
        self.setup_voz()

        motor = RoundedRectangle(width=3, height=1.2, corner_radius=0.15, color=AZUL,
                                  fill_opacity=0.15)
        motor_txt = Text("BacktestEngine", font_size=20, color=BLACK).move_to(motor)
        motor_grupo = VGroup(motor, motor_txt).shift(LEFT * 3.5)

        interfaz = RoundedRectangle(width=3, height=1.0, corner_radius=0.15, color=GRIS,
                                     fill_opacity=0.1)
        interfaz_txt = Text("AbstractJuez", font_size=18, color=BLACK).move_to(interfaz)
        interfaz_grupo = VGroup(interfaz, interfaz_txt)

        ttt = RoundedRectangle(width=3, height=1.2, corner_radius=0.15, color=VERDE,
                                fill_opacity=0.15)
        ttt_txt = Text("TTTJuez\n(Adapter)", font_size=18, color=BLACK).move_to(ttt)
        ttt_grupo = VGroup(ttt, ttt_txt).shift(RIGHT * 3.5)

        f1 = Arrow(motor_grupo.get_right(), interfaz_grupo.get_left(), color=GRIS)
        f2 = Arrow(interfaz_grupo.get_right(), ttt_grupo.get_left(), color=GRIS)

        codigo = Code(
            code_string="class TTTJuez(AbstractJuez):\n    ...",
            language="python", background="window", formatter_style="monokai",
            paragraph_config={"font_size": 20},
        ).scale(0.8).to_edge(DOWN, buff=0.6)

        self.narrar(
            "Arquitectónicamente, TTTJuez es un Adapter sobre el paquete "
            "trueskillthroughtime — el motor de backtesting depende de la "
            "interfaz AbstractJuez, no de la librería externa.",
            FadeIn(motor_grupo), FadeIn(interfaz_grupo), FadeIn(ttt_grupo),
            GrowArrow(f1), GrowArrow(f2),
        )
        self.narrar(
            "Eso significa que sustituir TTT por, digamos, un sistema "
            "Elo, no requiere tocar una sola línea de motor.py. Es la "
            "razón de que este proyecto pueda evolucionar sin "
            "reescrituras.",
            Create(codigo),
        )
        self.wait(1)
