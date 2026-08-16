"""[08:45-09:15] Texto: el backtest original corria sin ninguna gestion de riesgo."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base import EscenaBase


class TextoRiesgo(EscenaBase):
    def construct(self):
        self.setup_voz()
        self.tarjeta_simple(
            "§1.1 + §1.2\nEl backtest original corría sin\nninguna gestión de riesgo",
            "Con los supuestos auditados, pasamos al hallazgo más grave "
            "que encontramos en el motor de backtesting: existía una "
            "clase RiskOverlay completa — target volatility, trailing "
            "stop con estado global — pero motor.py nunca la importaba. "
            "El sistema invertía el cien por ciento del capital, "
            "siempre, sin ningún mecanismo de protección activo.",
        )
