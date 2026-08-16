"""Clase base compartida por las 27 escenas del video (guión v3, ~15:30).

Centraliza: servicio de voz, paleta de colores, y helpers para los dos
patrones que se repiten en el guión — tarjeta de texto narrada, y
tabla/código resaltado narrado.
"""
from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

# Voz por defecto: gTTS (gratis, sin API key) para poder previsualizar ya.
# Para la grabación final, cambiar por ElevenLabs (mejor prosodia en español):
#   from manim_voiceover.services.elevenlabs import ElevenLabsService
#   VOZ = ElevenLabsService(voice_name="Nombre_de_tu_voz_ES")
VOZ = GTTSService(lang="es", tld="com.mx")

AZUL = "#4C72B0"
ROJO = "#C44E52"
VERDE = "#55A868"
NARANJA = "#DD8452"
GRIS = "#808080"
NEGRO = "#1A1A1A"


class EscenaBase(VoiceoverScene):
    """Toda escena del video hereda de acá y llama self.setup_voz() primero."""

    def setup_voz(self) -> None:
        self.set_speech_service(VOZ)
        self.camera.background_color = WHITE

    def titulo(self, texto: str, font_size: int = 32) -> Text:
        return Text(texto, font_size=font_size, weight=BOLD, color=NEGRO).to_edge(UP)

    def narrar(self, texto: str, *animaciones, run_time: float | None = None):
        """Ejecuta animaciones sincronizadas con la duración real del audio narrado."""
        with self.voiceover(text=texto) as tracker:
            dur = run_time if run_time is not None else tracker.duration
            if animaciones:
                self.play(*animaciones, run_time=dur)
            else:
                self.wait(dur)

    def tarjeta_simple(self, titulo_txt: str, texto_narracion: str, font_size: int = 34):
        """Patrón repetido: título centrado en pantalla + narración sincronizada."""
        t = Text(titulo_txt, font_size=font_size, color=NEGRO, weight=BOLD)
        t.width = min(t.width, 11)
        self.narrar(texto_narracion, Write(t))
        self.wait(0.6)
        return t
