"""Extrae, en orden, el texto de cada narrar()/tarjeta_simple() de cada
escena, y genera audio_manual/GUION_GRABACION.md con la lista exacta de
qué grabar y cómo nombrar cada archivo.

Uso: python3 extraer_guion_audio.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

RAIZ = Path(__file__).parent
ESCENAS_DIR = RAIZ / "escenas"
SALIDA_DIR = RAIZ / "audio_manual"


def _texto_de_nodo(nodo: ast.AST) -> str | None:
    """Reconstruye un string literal (posiblemente concatenado con +
    implícito entre literales adyacentes, como se usa en todo el proyecto)."""
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return nodo.value
    if isinstance(nodo, ast.JoinedStr):
        return None  # f-strings no usadas para narración
    return None


def extraer_narraciones(archivo: Path) -> list[str]:
    codigo = archivo.read_text(encoding="utf-8")
    arbol = ast.parse(codigo)
    textos: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, nodo: ast.Call) -> None:
            nombre_metodo = None
            if isinstance(nodo.func, ast.Attribute):
                nombre_metodo = nodo.func.attr
            if nombre_metodo in ("narrar", "tarjeta_simple") and nodo.args:
                # narrar(texto, ...) -> primer arg; tarjeta_simple(titulo, texto, ...) -> segundo
                idx = 1 if nombre_metodo == "tarjeta_simple" else 0
                if idx < len(nodo.args):
                    t = _texto_de_nodo(nodo.args[idx])
                    if t:
                        textos.append((nodo.lineno, t))
            self.generic_visit(nodo)

    Visitor().visit(arbol)
    textos.sort(key=lambda x: x[0])
    return [t for _, t in textos]


def nombre_clase(archivo: Path) -> str | None:
    codigo = archivo.read_text(encoding="utf-8")
    m = re.search(r"^class (\w+)\(", codigo, re.MULTILINE)
    return m.group(1) if m else None


def main() -> None:
    SALIDA_DIR.mkdir(exist_ok=True)
    lineas = ["# Guión de grabación — audio manual\n",
              "Un archivo `NN.mp3` por cada bloque, en el orden exacto de esta lista.\n",
              "Guardalos en `audio_manual/<NombreEscena>/NN.mp3` (la carpeta ya está creada para cada una).\n"]

    total_bloques = 0
    for archivo in sorted(ESCENAS_DIR.glob("escena_*.py")):
        if archivo.name == "acto_card.py":
            continue
        clase = nombre_clase(archivo)
        if not clase:
            continue
        textos = extraer_narraciones(archivo)
        if not textos:
            continue
        carpeta_escena = SALIDA_DIR / clase
        carpeta_escena.mkdir(parents=True, exist_ok=True)

        lineas.append(f"\n## {clase}  (`audio_manual/{clase}/`)\n")
        for i, texto in enumerate(textos, start=1):
            total_bloques += 1
            lineas.append(f"**{i:02d}.mp3**\n> {texto}\n")

    # También los actos (tarjetas mudas, heredan de acto_card.py, sin narrar())
    lineas.append("\n## Tarjetas de ACTO (sin locución — no hace falta grabar nada)\n")
    lineas.append("Acto1Problema, Acto2Conflicto, Acto3Solucion, Acto4Aplicacion, "
                   "Acto5Conclusion: son transiciones mudas de ~10-12s, no llevan audio.\n")

    (SALIDA_DIR / "GUION_GRABACION.md").write_text("\n".join(lineas), encoding="utf-8")
    print(f"Listo: {total_bloques} bloques de audio a grabar, en {SALIDA_DIR / 'GUION_GRABACION.md'}")


if __name__ == "__main__":
    main()
