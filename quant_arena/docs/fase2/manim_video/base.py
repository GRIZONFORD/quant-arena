"""Clase base compartida por las 27 escenas del video (guión v3, ~15:30).

Centraliza: servicio de voz, paleta de colores, y helpers para los dos
patrones que se repiten en el guión — tarjeta de texto narrada, y
tabla/código resaltado narrado.
"""
from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene

# Voz por defecto: intenta gTTS (gratis, requiere internet, mejor calidad).
# Si no hay salida a internet (ej. entornos sandboxed/CI), cae automáticamente
# a pyttsx3 (100% offline, usa espeak-ng — más robótico pero no depende de red).
# Para la grabación final en tu compu con internet, cambiar por ElevenLabs
# (mejor prosodia en español):
#   from manim_voiceover.services.elevenlabs import ElevenLabsService
#   VOZ = ElevenLabsService(voice_name="Nombre_de_tu_voz_ES")
def _hay_internet_para_gtts() -> bool:
    """Prueba una conexión HTTPS real (no solo DNS) a Google Translate.

    DNS puede resolver aunque la conexión HTTPS esté bloqueada por un proxy
    corporativo/sandbox, así que gethostbyname() solo no alcanza.
    """
    import urllib.request

    try:
        urllib.request.urlopen("https://translate.google.com", timeout=4)
        return True
    except Exception:
        return False


try:
    if not _hay_internet_para_gtts():
        raise OSError("sin salida HTTPS a translate.google.com")
    from manim_voiceover.services.gtts import GTTSService

    VOZ = GTTSService(lang="es", tld="com.mx")
except OSError:
    import subprocess
    from pathlib import Path

    import pyttsx3
    from manim_voiceover.services.pyttsx3 import PyTTSX3Service

    class _PyTTSX3ServiceMP3Real(PyTTSX3Service):
        """El driver espeak de pyttsx3 en Linux escribe WAV real aunque se le
        pida .mp3 (el nombre no cambia el contenido) — mutagen después falla
        al leer el header MP3. Se genera a .wav y se convierte con ffmpeg."""

        def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
            if cache_dir is None:
                cache_dir = self.cache_dir
            input_data = {"input_text": text, "service": "pyttsx3"}
            cached = self.get_cached_result(input_data, cache_dir)
            if cached is not None:
                return cached

            audio_path = (path if path else self.get_audio_basename(input_data) + ".mp3")
            final_path = Path(cache_dir) / audio_path
            tmp_wav = final_path.with_suffix(".wav")

            self.engine.save_to_file(text, str(tmp_wav))
            self.engine.runAndWait()
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_wav), str(final_path)],
                check=True,
            )
            tmp_wav.unlink(missing_ok=True)

            return {
                "input_text": text,
                "input_data": input_data,
                "original_audio": str(audio_path),
            }

    _engine = pyttsx3.init()
    _engine.setProperty("rate", 165)
    for _voz in _engine.getProperty("voices"):
        if _voz.id.startswith("roa/es") or "es" in getattr(_voz, "languages", []):
            _engine.setProperty("voice", _voz.id)
            break
    VOZ = _PyTTSX3ServiceMP3Real(engine=_engine)

AZUL = "#4C72B0"
ROJO = "#C44E52"
VERDE = "#55A868"
NARANJA = "#DD8452"
GRIS = "#808080"
NEGRO = "#1A1A1A"

# Cambiar a True para usar los audios grabados en audio_manual/<Escena>/
# en vez de sintetizar voz. No hace falta tocar ninguna escena — el
# switch es acá, una sola vez.
USAR_AUDIO_MANUAL = False


class EscenaBase(VoiceoverScene):
    """Toda escena del video hereda de acá y llama self.setup_voz() primero."""

    def setup_voz(self) -> None:
        if USAR_AUDIO_MANUAL:
            from pathlib import Path

            from manual_audio_service import ManualAudioService

            carpeta = Path(__file__).parent / "audio_manual" / self.__class__.__name__
            self.set_speech_service(ManualAudioService(audio_dir=carpeta))
        else:
            self.set_speech_service(VOZ)
        self.camera.background_color = WHITE

    def titulo(self, texto: str, font_size: int = 32) -> Text:
        return Text(texto, font_size=font_size, weight=BOLD, color=NEGRO).to_edge(UP)

    def narrar(
        self,
        texto: str,
        *animaciones,
        run_time: float | None = None,
        anim_max: float = 1.3,
    ):
        """Reproduce animaciones a ritmo natural mientras narra.

        Antes la animación se estiraba para durar exactamente lo mismo que
        el audio completo (a veces 8-10s) — se veía en cámara lenta. Ahora
        la animación corre rápido (tope `anim_max` segundos, con easing
        suave tipo `smooth`) y el resto de la narración se cubre con una
        pausa quieta, como en un explicador real: el gesto visual es
        breve, la voz sigue y la imagen se sostiene.

        `run_time` sigue disponible para forzar una duración específica
        cuando una animación puntual necesita más o menos tiempo que el
        default.
        """
        with self.voiceover(text=texto) as tracker:
            if animaciones:
                dur = run_time if run_time is not None else min(anim_max, tracker.duration)
                self.play(*animaciones, run_time=dur, rate_func=smooth)
                resto = tracker.duration - dur
                if resto > 0:
                    self.wait(resto)
            else:
                self.wait(tracker.duration)

    def tarjeta_simple(self, titulo_txt: str, texto_narracion: str, font_size: int = 34):
        """Patrón repetido: título centrado en pantalla + narración sincronizada."""
        t = Text(titulo_txt, font_size=font_size, color=NEGRO, weight=BOLD)
        t.width = min(t.width, 11)
        self.narrar(texto_narracion, Write(t))
        self.wait(0.6)
        return t
