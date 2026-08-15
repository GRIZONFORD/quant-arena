# =============================================================================
# FILE: quant_arena/core/torch_compat.py
# Fallback mínimo para `torch.nn` cuando PyTorch no está instalado.
# =============================================================================
"""
Varias estrategias del Zoo (`tft_strategy`, `mamba_ssm_strategy`,
`stgnn_strategy`, `wavelet_lstm_strategy`, `neural_ff_strategy`) definen
arquitecturas que heredan de ``nn.Module`` A NIVEL DE MÓDULO — la
definición de clase (``class Bloque(nn.Module): ...``) se ejecuta durante
el *import*, no durante la instanciación.

Cada uno de esos archivos ya guarda ``torch`` detrás de un
``try/except ImportError`` (el patrón estándar de este repo para
dependencias ML opcionales, igual que ``hmmlearn``/``arch`` en
``hmm_garch_strategy.py``). Pero si el import falla, ``nn`` queda sin
definir, y la sentencia ``class Bloque(nn.Module)`` lanza ``NameError`` al
cargar el módulo — no un ``ImportError`` capturable, un fallo de carga que
propaga hacia arriba. Como ``zoo/estrategias/__init__.py`` importa TODOS
los submódulos del paquete de forma eager, ese ``NameError`` rompe
``RegistroZoo.autodescubrir()`` para TODO el Zoo, incluidas estrategias sin
ninguna relación con PyTorch (momentum, OLPS-RMR).

Este módulo NO reemplaza a PyTorch: solo provee una clase base inerte para
que la DEFINICIÓN de clase no falle. Nada aquí hace matemática de tensores.
Instanciar o usar una estrategia neuronal real sigue requiriendo PyTorch —
cada estrategia mantiene su propio flag ``_TORCH_OK`` y lanza
``ImportError`` explícito en su constructor si torch no está disponible
(igual que antes de esta corrección).

Uso en cada estrategia afectada:

    try:
        import torch
        import torch.nn as nn
        ...
        _TORCH_OK = True
    except ImportError:
        _TORCH_OK = False
        from quant_arena.core.torch_compat import nn  # type: ignore[assignment]

    class MiBloque(nn.Module):  # ya no lanza NameError sin torch
        ...
"""
from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn  # type: ignore[import-untyped]
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False

    class _ModuloInerte:
        """
        Sustituto inerte de ``torch.nn.Module``.

        Permite que ``class X(_ModuloInerte)`` se DEFINA sin error. Llamar
        al constructor no falla (para no romper ``super().__init__()`` en
        las subclases), pero cualquier intento de uso real —tensores,
        parámetros, forward— requiere PyTorch instalado y por tanto nunca
        llega a ejecutarse: cada estrategia valida ``_TORCH_OK`` antes de
        instanciar su arquitectura.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "PyTorch no está instalado. Esta estrategia requiere "
                "'pip install torch' (o 'pip install -e .[dl]')."
            )

    class _NNFallback:
        """Sustituto de facto del módulo ``torch.nn`` — solo expone ``Module``."""

        Module = _ModuloInerte

    nn = _NNFallback()  # type: ignore[assignment]
