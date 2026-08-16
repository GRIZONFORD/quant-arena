"""[04:40-05:20] Codigo: pesos_asignacion(metodo='mu_sobre_sigma')."""
from manim import *
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase, AZUL


class MuSobreSigma(EscenaBase):
    def construct(self):
        self.setup_voz()

        codigo = Code(
            code_string=(
                "# juez/ttt_juez.py\n"
                "def pesos_asignacion(self, metodo='mu_sobre_sigma'):\n"
                "    mu, sigma = self.habilidades_latentes()\n"
                "    return (mu / sigma).clip(lower=0)  # Sharpe bayesiano"
            ),
            language="python", background="window", formatter_style="monokai",
            paragraph_config={"font_size": 22},
        )
        codigo.scale(0.85)

        formula = MathTex(r"\text{peso} \propto \frac{\mu}{\sigma}", font_size=48, color=AZUL)
        formula.next_to(codigo, DOWN, buff=0.8)

        self.narrar(
            "De esa creencia se deriva directamente cuánto capital "
            "recibe cada estrategia: el cociente mu sobre sigma — la "
            "habilidad estimada dividida por la incertidumbre.",
            Create(codigo),
        )
        self.narrar(
            "Estrategias con alta habilidad estimada y alta certeza "
            "reciben más capital que estrategias con la misma habilidad "
            "estimada pero mucha incertidumbre. La incertidumbre ya no "
            "es ruido que se descarta — es información para decidir "
            "cuánto arriesgar.",
            Write(formula),
        )
        self.wait(1)
