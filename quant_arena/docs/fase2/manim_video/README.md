# Video quant-arena — proyecto Manim (guión v3, ~15:30)

27 escenas, una por cada fila de `../video_guion_15min_v3.md`, con narración
sincronizada vía `manim-voiceover`. Pensado para render técnico tipo
3Blue1Brown, no un avatar leyendo el guión — el registro que espera una
audiencia de un congreso bayesiano.

## 1. Instalación

```bash
cd quant_arena/docs/fase2/manim_video
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Manim necesita LaTeX (para las fórmulas MathTex) y ffmpeg como binarios de sistema:
#   Debian/Ubuntu: sudo apt install ffmpeg texlive-full
#   macOS:         brew install ffmpeg && brew install --cask mactex
```

Por defecto la narración usa **gTTS** (Google Text-to-Speech), gratis y sin
API key, para poder previsualizar todo de inmediato. La calidad de voz no
es la de una grabación final — para eso ver el paso 5.

## 2. Estructura

```
manim_video/
├── base.py              # clase EscenaBase compartida (voz, colores, helpers)
├── manim.cfg             # calidad/fps/fondo por defecto
├── requirements.txt
├── render_all.sh          # renderiza las 27 escenas en orden
├── concat_escenas.sh       # las une en un solo mp4
└── escenas/
    ├── escena_01_apertura.py
    ├── escena_02_split_sharpe.py
    ├── ...
    └── escena_27_logo.py
```

Cada archivo mapea 1:1 a una fila de la tabla del guión v3 — el timestamp
del guión está en el docstring del archivo. `escena_10_kalman.py` es la
escena nueva agregada en v3 (el filtro de Kalman antes del Juez TTT).

## 3. Previsualizar UNA escena mientras la ajustás

Esto es lo que vas a usar el 90% del tiempo mientras iterás:

```bash
manim -pql escenas/escena_07_gaussianas.py Gaussianas
```

- `-p` abre el video automáticamente al terminar.
- `-ql` = *quality low* (renderiza en segundos, ideal para iterar).
- El segundo argumento (`Gaussianas`) es el nombre de la clase `Scene`
  dentro del archivo — está siempre al final del docstring/en la primera
  línea `class ... (EscenaBase):`.

Cuando quede bien, la escena queda cacheada en `media/videos/.../480p15/`.

## 4. Renderizar TODO

```bash
chmod +x render_all.sh concat_escenas.sh

# Borrador rápido de las 27 escenas (para revisar timing y contenido global):
./render_all.sh draft

# Calidad final 1080p60 (esto tarda — cada escena con LaTeX/curvas puede
# llevar 1-3 min, contá con 30-60 min para las 27):
./render_all.sh final

# Unir las 27 en un solo archivo:
./concat_escenas.sh
# -> video_ttt_congreso.mp4
```

`render_all.sh` genera `lista_escenas.txt` con la ruta absoluta de cada
`.mp4` en el orden correcto (01 a 27) — es el input de `concat_escenas.sh`
vía `ffmpeg -f concat`, que no reencodea (rápido, sin pérdida de calidad).

## 5. Subir la calidad de voz para la grabación final

gTTS suena robótico. Para el congreso, cambiar a **ElevenLabs** (mucho
mejor prosodia en español):

```bash
pip install "manim-voiceover[elevenlabs]"
export ELEVENLABS_API_KEY="tu_key"
```

En `base.py`, reemplazar:

```python
# Antes:
from manim_voiceover.services.gtts import GTTSService
VOZ = GTTSService(lang="es", tld="com.mx")

# Después:
from manim_voiceover.services.elevenlabs import ElevenLabsService
VOZ = ElevenLabsService(voice_name="Nombre_de_tu_voz_ES")  # elegir voz en elevenlabs.io
```

Ese único cambio se propaga a las 27 escenas porque todas heredan de
`EscenaBase` y usan la misma variable `VOZ`. Después de cambiarlo, hay que
volver a renderizar (`./render_all.sh final`) — el audio queda embebido en
cada clip.

## 6. Qué NO está automatizado (y por qué)

- **Capturas de código real** (`escena_09`, `escena_10`, `escena_11`,
  `escena_15`, `escena_19`): el código está tipeado directo en cada
  escena con `manim.Code`, no es una captura de pantalla del editor real.
  Si preferís mostrar el archivo real con su resaltado de sintaxis nativo,
  reemplazá el bloque `Code(code_string=...)` por una imagen (screenshot
  del editor) importada con `ImageMobject("ruta.png")`.
- **Transiciones entre escenas y subtítulos**: `ffmpeg concat` solo pega
  los clips en seco. Si querés cortes con crossfade, lower-thirds o
  subtítulos quemados, importá `video_ttt_congreso.mp4` a DaVinci Resolve
  (gratis) y editá ahí — Manim no es un editor de timeline.
- **Ecuación LaTeX en las tablas de la escena 13/14**: usan `manim.Table`
  con `Text`, no LaTeX, para evitar dependencias de compilación pesadas en
  celdas; si tenés LaTeX instalado y preferís `MathTable`, es un cambio de
  una línea en esos dos archivos.

## 7. Discrepancias deliberadas frente al guión v3 (documentadas, no bugs)

- `escena_24_tabla_resumen.py` y `escena_25_limitaciones.py` usan **194/194
  tests** y marcan **crowding como implementado** en el texto en pantalla,
  aunque la locución textual del guión v3 (heredada de v1) dice "175/175"
  y presenta crowding como roadmap. Es la cifra y el estado reales al día
  de esta sesión — ver la nota de reconciliación en
  `../video_guion_15min_v3.md`. Si vas a grabar el audio de esas dos
  escenas, decí la cifra y el estado actuales, no los del guión original.

## 8. Checklist antes de la grabación final

- [ ] Revisar cada escena en `-pql` (rápido) para timing y contenido.
- [ ] Cambiar `VOZ` a ElevenLabs en `base.py`.
- [ ] `./render_all.sh final` (calidad 1080p60).
- [ ] `./concat_escenas.sh`.
- [ ] Pasar `video_ttt_congreso.mp4` por DaVinci Resolve para
      transiciones/subtítulos si el congreso los exige.
- [ ] Confirmar que la escena 24 dice 194/194 (no 175/175) al hablar en
      vivo o al grabar el audio final.
