#!/usr/bin/env bash
# Une las 27 escenas renderizadas (en el orden de lista_escenas.txt) en un
# solo archivo. Requiere haber corrido ./render_all.sh antes.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f lista_escenas.txt ]; then
    echo "No existe lista_escenas.txt — corré ./render_all.sh primero." >&2
    exit 1
fi

ffmpeg -y -f concat -safe 0 -i lista_escenas.txt -c copy video_ttt_congreso.mp4
echo "Listo: video_ttt_congreso.mp4"
