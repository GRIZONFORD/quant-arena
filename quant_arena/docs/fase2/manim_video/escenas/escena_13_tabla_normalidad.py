"""[07:00-07:45] Tabla en vivo: normalidad y homocedasticidad, resultados reales."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, ROJO


class TablaNormalidad(EscenaBase):
    def construct(self):
        self.setup_voz()

        titulo = self.titulo("S&P 500 real — 7,394 observaciones (1997-2026)")

        tabla = Table(
            [["Normalidad", "Jarque-Bera", "31,024", "p ≈ 0", "SE RECHAZA"],
             ["Homocedasticidad", "ARCH-LM", "—", "p ≈ 6e-155", "SE RECHAZA"]],
            col_labels=[Text(c, font_size=20, weight=BOLD) for c in
                        ["Test", "Estadístico", "Valor", "p-valor", "Resultado"]],
            include_outer_lines=True,
        ).scale(0.55)
        tabla.next_to(titulo, DOWN, buff=0.6)

        self.narrar(
            "Corrimos la batería completa de contrastes sobre los "
            "log-retornos reales.",
            Write(titulo), Create(tabla),
        )
        self.narrar(
            "Normalidad: se rechaza, con un estadístico de Jarque-Bera de "
            "treinta y un mil — los retornos financieros tienen colas "
            "mucho más pesadas que una gaussiana.",
            Indicate(tabla.get_rows()[1], color=ROJO),
        )
        self.narrar(
            "Homocedasticidad: se rechaza con un p-valor de 6 por 10 a la "
            "menos 155 vía ARCH-LM — hay clustering de volatilidad "
            "severo. Ninguno de los dos es sorprendente en finanzas. Lo "
            "que importa es que no lo asumimos: lo medimos, y actuamos.",
            Indicate(tabla.get_rows()[2], color=ROJO),
        )
        self.wait(1)
