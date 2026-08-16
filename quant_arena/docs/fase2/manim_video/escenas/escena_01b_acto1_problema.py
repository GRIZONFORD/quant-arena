"""[01:05-01:15] Tarjeta de acto nueva en v5: ACTO I - EL PROBLEMA."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acto_card import TarjetaActoBase


class Acto1Problema(TarjetaActoBase):
    numero = "I"
    nombre = "EL PROBLEMA"
