"""[03:05-03:20] Tarjeta de acto nueva en v5: ACTO III - LA SOLUCION BAYESIANA."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acto_card import TarjetaActoBase


class Acto3Solucion(TarjetaActoBase):
    numero = "III"
    nombre = "LA SOLUCIÓN BAYESIANA"
