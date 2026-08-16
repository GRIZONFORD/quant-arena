"""[05:20-05:50] Escena nueva de v3: filtro de Kalman antes del Juez TTT."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL, ROJO, GRIS


class Kalman(EscenaBase):
    def construct(self):
        self.setup_voz()

        codigo1 = Code(
            code_string=(
                "# backtesting/motor.py\n"
                "serie_filt = self._kalman.filtrar(serie_bruta)\n"
                "d[self._metrica_ranking] = valor_filtrado"
            ),
            language="python", background="window", formatter_style="monokai",
            paragraph_config={"font_size": 18},
        ).scale(0.7)

        codigo2 = Code(
            code_string=(
                "# juez/ttt_juez.py\n"
                "ranking = sorted(valores_validos.items(),\n"
                "                  key=lambda x: x[1], reverse=True)\n"
                "game = [[nombre] for nombre, _ in ranking]  # solo orden"
            ),
            language="python", background="window", formatter_style="monokai",
            paragraph_config={"font_size": 18},
        ).scale(0.7)

        # Ambos bloques apilados y anclados arriba, en vez de encadenados
        # con next_to (eso los dejaba pisando el gráfico de abajo).
        codigos = VGroup(codigo1, codigo2).arrange(DOWN, buff=0.35)
        codigos.to_edge(UP, buff=0.5)

        self.narrar(
            "Un detalle que suele pasar desapercibido: antes de que el "
            "Sharpe de cada estrategia entre a competir, pasa por un "
            "filtro de Kalman causal.",
            Create(codigo1),
        )
        self.narrar(
            "¿Por qué? Porque TrueSkill Through Time, tal como lo usamos "
            "acá, no ve el número — ve el orden: sabe que una estrategia "
            "quedó tercera de once, no que sacó cero coma setenta y tres "
            "de Sharpe.",
            Create(codigo2),
        )

        ax = Axes(x_range=[0, 8, 2], y_range=[0, 1, 0.5], x_length=6, y_length=2.2,
                   tips=False, axis_config={"color": GRIS}).to_edge(DOWN, buff=0.4)
        cruda = ax.plot(lambda x: 0.5 + 0.3 * np.sin(1.6 * x), color=GRIS)
        suave = ax.plot(lambda x: 0.5 + 0.15 * np.sin(1.6 * x), color=AZUL)
        etq = Text("cruda (se cruza seguido) vs. filtrada (orden estable)",
                    font_size=16, color=BLACK).next_to(ax, UP, buff=0.1)

        self.narrar(
            "Un ranking basado en una ventana corta y ruidosa se voltea "
            "todo el tiempo por pura casualidad estadística. El Kalman no "
            "le da más precisión a TTT — le da un ranking más estable "
            "antes de que ese ranking se vuelva evidencia permanente en "
            "el historial de partidas.",
            Create(ax), Create(cruda), Create(suave), FadeIn(etq),
        )
        self.wait(1)
