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
            "Corrimos la batería completa de contrastes estadísticos "
            "sobre los retornos reales.",
            Write(titulo), Create(tabla),
        )
        self.narrar(
            "Normalidad, o sea si los datos siguen esa forma de campana "
            "que mencionamos antes: se rechaza, con un test de "
            "Jarque-Bera cuyo resultado numérico es treinta y un mil — "
            "cuanto más grande ese número, más lejos está de parecerse "
            "a una campana perfecta. En criollo: los retornos "
            "financieros tienen eventos extremos mucho más frecuentes "
            "de lo que una campana predeciría.",
            Indicate(tabla.get_rows()[1], color=ROJO),
        )
        self.narrar(
            "Homocedasticidad, o sea que la variabilidad del mercado "
            "sea pareja en el tiempo: también se rechaza, con una "
            "probabilidad de que sea casualidad de prácticamente cero "
            "— hay 'rachas' de volatilidad, períodos tranquilos y "
            "violentos que se agrupan. Ninguno de los dos es "
            "sorprendente en finanzas. Lo que importa es que no lo "
            "asumimos: lo medimos, y actuamos en consecuencia.",
            Indicate(tabla.get_rows()[2], color=ROJO),
        )
        self.wait(1)
