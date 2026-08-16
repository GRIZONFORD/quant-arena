"""[13:30-14:10] Tabla resumen final de hitos y verificacion.

Nota: la locucion original dice "175/175 tests"; el texto en pantalla de
esta escena usa 194/194, la cifra vigente al momento de grabar (ver nota
de reconciliacion en video_guion_15min_v3.md). Ajustar el audio si se
graba con ElevenLabs para decir "ciento noventa y cuatro" en vez de
"ciento setenta y cinco".
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, VERDE


class TablaResumen(EscenaBase):
    def construct(self):
        self.setup_voz()

        titulo = self.titulo("Hitos completados")

        tabla = Table(
            [["§1.1 Riesgo", "Sharpe 0.60→0.73 sobre datos reales"],
             ["§1.2 Kelly", "acoplado a incertidumbre TTT"],
             ["§1.3 Juez TTT", "causalidad exigida, costo acotado"],
             ["§1.5 Diagnósticos", "K=4 vs K=3 por criterio real"],
             ["Pruebas automatizadas", "194/194 — sin errores de tipos"]],
            include_outer_lines=True,
        ).scale(0.5)
        tabla.next_to(titulo, DOWN, buff=0.5)

        self.narrar(
            "En resumen, lo que está construido y verificado hoy: "
            "control de riesgo conectado, con mejora medida en Sharpe y "
            "en la peor caída, sobre datos reales; el criterio de Kelly "
            "acoplado a la incertidumbre del Juez, con su limitación de "
            "calibración documentada.",
            Write(titulo), Create(tabla),
        )
        self.narrar(
            "Validación de los supuestos estadísticos con resultados "
            "reales sobre el S&P 500, no simulados; y un Juez TTT que "
            "no puede ver el futuro por construcción, con costo "
            "computacional acotado y ajuste de sus parámetros basado en "
            "evidencia. Ciento noventa y cuatro de ciento noventa y "
            "cuatro pruebas automatizadas pasando, sin ninguna "
            "regresión sobre el comportamiento original.",
            tabla.get_rows()[-1].animate.set_color(VERDE),
        )
        self.wait(1)
