"""[13:15-13:30] Tarjeta de acto nueva en v5: ACTO V - CONCLUSION."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acto_card import TarjetaActoBase


class Acto5Conclusion(TarjetaActoBase):
    numero = "V"
    nombre = "CONCLUSIÓN"
