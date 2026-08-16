"""Servicio de voz que usa audios pre-grabados en vez de sintetizar.

Uso: activar USAR_AUDIO_MANUAL=True en base.py. Cada escena busca sus
archivos en audio_manual/<NombreClaseEscena>/01.mp3, 02.mp3, ... — un
archivo por cada llamada a self.narrar()/tarjeta_simple() en el orden en
que aparecen en el código. Si falta un archivo, se usa un silencio de
2s como placeholder para no romper el render mientras van llegando las
grabaciones ("parte por parte").
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from manim_voiceover.services.base import SpeechService


class ManualAudioService(SpeechService):
    def __init__(self, audio_dir: Path, **kwargs) -> None:
        self.audio_dir = Path(audio_dir)
        self._contador = 0
        super().__init__(**kwargs)

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        if cache_dir is None:
            cache_dir = self.cache_dir
        self._contador += 1
        origen = self.audio_dir / f"{self._contador:02d}.mp3"

        input_data = {
            "input_text": text,
            "service": "manual",
            "escena": self.audio_dir.name,
            "indice": self._contador,
        }
        audio_path = self.get_audio_basename(input_data) + ".mp3"
        destino = Path(cache_dir) / audio_path

        if origen.exists():
            shutil.copyfile(origen, destino)
        else:
            # Placeholder de silencio — permite renderizar aunque falten
            # tomas todavía no enviadas.
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=mono", "-t", "2",
                    str(destino),
                ],
                check=True,
            )

        return {
            "input_text": text,
            "input_data": input_data,
            "original_audio": str(audio_path),
        }
