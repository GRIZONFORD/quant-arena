"""[09:15-10:00] Curvas de equity real: sin riesgo vs con RiskOverlay."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO, VERDE, GRIS


class EquityRiskOverlay(EscenaBase):
    def construct(self):
        self.setup_voz()

        titulo = self.titulo("Momentum + OLPS-RMR — S&P 500, 2015-2020")

        ax = Axes(x_range=[0, 10, 2], y_range=[0.6, 2.0, 0.4], x_length=9, y_length=4.5,
                   tips=False, axis_config={"color": GRIS}).next_to(titulo, DOWN, buff=0.4)

        def sin_riesgo(x):
            return 1 + 0.1 * x - 0.5 * np.exp(-((x - 6) ** 2) / 1.2)

        def con_overlay(x):
            return 1 + 0.09 * x - 0.22 * np.exp(-((x - 6) ** 2) / 1.2)

        c1 = ax.plot(sin_riesgo, color=ROJO, x_range=[0, 10])
        c2 = ax.plot(con_overlay, color=VERDE, x_range=[0, 10])

        leyenda = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=ROJO), Text("Sin riesgo — Sharpe 0.60, MaxDD −34%", font_size=18, color=BLACK)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=VERDE), Text("Con RiskOverlay — Sharpe 0.73, MaxDD −20%", font_size=18, color=BLACK)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).to_edge(DOWN, buff=0.4)

        self.narrar(
            "La conectamos, de forma opcional, para no romper ningún "
            "resultado previo.",
            Write(titulo), Create(ax), Create(c1),
        )
        self.narrar(
            "El resultado sobre datos reales: el Sharpe anualizado sube "
            "de 0.60 a 0.73, y la peor caída — desde el punto más alto "
            "hasta el más bajo antes de recuperarse, lo que llamamos "
            "drawdown — se reduce de menos 34 por ciento a menos 20 por "
            "ciento. No es un backtest sintético — es la misma "
            "estrategia, el mismo período, con y sin ese control de "
            "riesgo aplicado a su propio historial real.",
            Create(c2), FadeIn(leyenda),
        )
        self.wait(1)
