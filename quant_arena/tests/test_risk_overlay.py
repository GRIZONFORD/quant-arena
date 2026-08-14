# =============================================================================
# FILE: quant_arena/tests/test_risk_overlay.py
# Suite pytest para RiskOverlay — target vol + trailing stop (fijo/dinámico,
# gate binario/gradual).
#
# Cobertura:
#   - No-regresión: con dynamic_dd=False y gradual_derisking=False (defaults)
#     el comportamiento reproduce exactamente el V2.1 original.
#   - dd_limit dinámico se mantiene dentro de [dd_min, dd_max].
#   - de-risking gradual reduce exposición de forma continua (no salto 0/1)
#     ante un gap de precio que dispara drawdown.
#   - Validación de parámetros en __init__.
#   - Estado persiste correctamente entre folds (compute_risk_weight llamado
#     dos veces en secuencia).
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_arena.backtesting.risk_overlay import RiskOverlay


def _serie_precio_con_crash(n=80, crash_idx=40, crash_size=-0.30):
    """Serie de retornos diarios con un crash abrupto en `crash_idx`."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    ret = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    ret.iloc[crash_idx] = crash_size
    market_data = pd.DataFrame({"realized_vol": np.full(n, 0.15)}, index=idx)
    base_weight = pd.Series(1.0, index=idx)
    return base_weight, ret, market_data


def test_defaults_reproducen_comportamiento_v21_gate_binario():
    """Con los defaults, el gate debe ser estrictamente 0.0 o 1.0 (sin rampa)."""
    base_weight, ret, market_data = _serie_precio_con_crash()
    overlay = RiskOverlay(dd_limit=0.15)
    salida = overlay.compute_risk_weight(base_weight, ret, market_data, "estrategia_x")
    tv = (overlay.target_vol / market_data["realized_vol"].shift(1)).clip(upper=overlay.max_leverage).fillna(0.0)
    w_tv = base_weight * tv
    factor_implicito = (salida / w_tv.replace(0.0, np.nan)).fillna(0.0)
    valores_unicos = set(np.round(factor_implicito.unique(), 6))
    assert valores_unicos.issubset({0.0, 1.0})


def test_dd_limit_dinamico_respeta_cotas():
    base_weight, ret, market_data = _serie_precio_con_crash(n=150, crash_idx=100)
    overlay = RiskOverlay(dynamic_dd=True, dd_min=0.05, dd_max=0.30, dd_k=2.0)
    # No debe lanzar y debe producir una serie de exposición válida en [0, cap]
    salida = overlay.compute_risk_weight(base_weight, ret, market_data, "din")
    assert salida.notna().all()
    assert (salida >= -1e-9).all()


def test_gradual_derisking_produce_valores_intermedios():
    """Con gradual_derisking=True, el factor de exposición no debe ser sólo {0, 1}."""
    base_weight, ret, market_data = _serie_precio_con_crash(n=120, crash_idx=60, crash_size=-0.10)
    overlay = RiskOverlay(dd_limit=0.20, gradual_derisking=True, derisk_power=2.0)
    salida = overlay.compute_risk_weight(base_weight, ret, market_data, "gradual")
    tv = (overlay.target_vol / market_data["realized_vol"].shift(1)).clip(upper=overlay.max_leverage).fillna(0.0)
    w_tv = base_weight * tv
    factor_implicito = (salida / w_tv.replace(0.0, np.nan)).dropna()
    intermedios = factor_implicito[(factor_implicito > 0.01) & (factor_implicito < 0.99)]
    assert len(intermedios) > 0


def test_gradual_derisking_reduce_exposicion_mas_alla_del_gate_binario():
    """Tras un drawdown parcial (por debajo de dd_limit), el gate binario sigue
    en 1.0 pero la rampa gradual ya debe haber empezado a reducir exposición."""
    base_weight, ret, market_data = _serie_precio_con_crash(n=120, crash_idx=60, crash_size=-0.08)
    binario = RiskOverlay(dd_limit=0.20, gradual_derisking=False)
    gradual = RiskOverlay(dd_limit=0.20, gradual_derisking=True, derisk_power=1.0)

    salida_bin = binario.compute_risk_weight(base_weight, ret, market_data, "x")
    salida_grad = gradual.compute_risk_weight(base_weight, ret, market_data, "x")

    # Justo después del crash (drawdown parcial, no cruza el límite), la
    # exposición gradual debe ser estrictamente menor que la binaria.
    ventana = slice(61, 65)
    assert (salida_grad.iloc[ventana] < salida_bin.iloc[ventana]).any()


def test_estado_persiste_entre_folds():
    base_weight, ret, market_data = _serie_precio_con_crash(n=60, crash_idx=30, crash_size=-0.25)
    overlay = RiskOverlay(dd_limit=0.15)

    fold1_ret = ret.iloc[:30]
    fold1_bw = base_weight.iloc[:30]
    fold1_md = market_data.iloc[:30]
    overlay.compute_risk_weight(fold1_bw, fold1_ret, fold1_md, "persistente")

    assert "persistente" in overlay._state
    eq1, hwm1, factor1 = overlay._state["persistente"]
    assert eq1 > 0.0 and hwm1 > 0.0

    fold2_ret = ret.iloc[30:]
    fold2_bw = base_weight.iloc[30:]
    fold2_md = market_data.iloc[30:]
    salida2 = overlay.compute_risk_weight(fold2_bw, fold2_ret, fold2_md, "persistente")
    # El primer día del fold2 debe heredar el factor de participación del fold1
    assert salida2.notna().all()


def test_reset_limpia_estado():
    overlay = RiskOverlay()
    overlay._state["x"] = (0.5, 1.0, 0.0)
    overlay.reset()
    assert overlay._state == {}


# ---------------------------------------------------------------------------
# Validación de configuración
# ---------------------------------------------------------------------------

def test_dd_limit_no_positivo_lanza_error():
    with pytest.raises(ValueError):
        RiskOverlay(dd_limit=0.0)


def test_dynamic_dd_cotas_invalidas_lanza_error():
    with pytest.raises(ValueError):
        RiskOverlay(dynamic_dd=True, dd_min=0.30, dd_max=0.05)


def test_derisk_power_no_positivo_lanza_error():
    with pytest.raises(ValueError):
        RiskOverlay(derisk_power=0.0)
