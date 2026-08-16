"""[08:45-09:15] Texto: el backtest original corria sin ninguna gestion de riesgo."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase


class TextoRiesgo(EscenaBase):
    def construct(self):
        self.setup_voz()
        self.tarjeta_simple(
            "§1.1 + §1.2\nEncontramos un fallo real en el código",
            "Con los supuestos auditados, encontramos un fallo real en "
            "el código — el hallazgo más grave de todo el proyecto: "
            "existía una clase completa para controlar el riesgo, con "
            "reducción de exposición y corte de pérdidas, pero el motor "
            "principal nunca la usaba. El sistema invertía el cien por "
            "ciento del capital, siempre, sin ninguna protección activa. "
            "Fue justamente por encontrar ese fallo que hicimos la "
            "modificación que van a ver ahora.",
        )
