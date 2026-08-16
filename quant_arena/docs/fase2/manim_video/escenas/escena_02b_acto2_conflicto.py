"""[01:45-02:05] Tarjeta de acto nueva en v5: ACTO II - EL CONFLICTO."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acto_card import TarjetaActoBase


class Acto2Conflicto(TarjetaActoBase):
    numero = "II"
    nombre = "EL CONFLICTO"
