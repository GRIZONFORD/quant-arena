"""[06:30-07:00] Texto: la pregunta de si el modelo viola sus propios supuestos."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase


class PreguntaSupuestos(EscenaBase):
    def construct(self):
        self.setup_voz()
        self.tarjeta_simple(
            "§1.5 — ¿Y si el modelo mismo\nviola sus propios supuestos?",
            "Ahora, la pregunta que casi ningún proyecto de este tipo "
            "responde: los modelos internos de las estrategias —en "
            "particular la que combina un HMM de régimen con GARCH— "
            "asumen normalidad, estacionariedad, homocedasticidad. ¿Se "
            "cumplen esos supuestos en los datos reales que estamos "
            "usando?",
        )
