"""[10:30-10:45] Tarjeta de acto nueva en v5: ACTO IV - APLICACION PRACTICA."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acto_card import TarjetaActoBase


class Acto4Aplicacion(TarjetaActoBase):
    numero = "IV"
    nombre = "APLICACIÓN PRÁCTICA"
