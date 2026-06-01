#!/usr/bin/env python3
"""
scripts/run_mvp_calibracion.py
MVP de validacion del modulo calibracion/ — quant_arena v1.1.0.
Verifica que L-BFGS-B converge correctamente y mejora la log-evidencia
sobre los valores iniciales de sigma/gamma del modelo TTT.
"""
import sys
import time
import warnings

import numpy as np

# Permite importar quant_arena desde el directorio raiz del proyecto
sys.path.insert(0, ".")

from quant_arena.core.abstracciones import MetricasResultado
from quant_arena.juez.ttt_juez import TTTJuez
from quant_arena.calibracion.optimizador import OptimizadorTTT

# ---------------------------------------------------------------------------
# Parametros de referencia (valores por defecto del paper ATP)
# ---------------------------------------------------------------------------
SIGMA_0 = 1.6
GAMMA_0 = 0.036

# ---------------------------------------------------------------------------
# Datos sinteticos: 10 epocas con tiempos secuenciales
#   Epocas  1-5 -> partidas 1v1 (2 jugadores)
#   Epocas 6-10 -> partidas 2v2 (4 jugadores, ranking individual)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

JUGADORES_1V1 = ["jugador_1", "jugador_2"]
JUGADORES_2V2 = ["jugador_1", "jugador_2", "jugador_3", "jugador_4"]

juez_base = TTTJuez(sigma=SIGMA_0, gamma=GAMMA_0)

for epoca in range(1, 11):
    participantes = JUGADORES_1V1 if epoca <= 5 else JUGADORES_2V2
    metricas_ep = {
        j: MetricasResultado(sharpe=float(rng.normal(0.5, 0.4)))
        for j in participantes
    }
    juez_base.registrar_periodo(metricas_ep, tiempo=float(epoca))

# ---------------------------------------------------------------------------
# Exportar historial — deepcopy garantiza independencia del estado del Juez
# ---------------------------------------------------------------------------
composition, times = juez_base.exportar_historial()
jugadores_unicos = len({p for game in composition for team in game for p in team})

# ---------------------------------------------------------------------------
# Log-evidencia inicial (punto de referencia para medir la mejora)
# ---------------------------------------------------------------------------
opt = OptimizadorTTT(sigma_inicial=SIGMA_0, gamma_inicial=GAMMA_0)

log_ev_inicial = opt._evaluar_log_evidencia(
    np.array([np.log(SIGMA_0), np.log(GAMMA_0)]),
    composition,
    times,
)

# ---------------------------------------------------------------------------
# Calibracion
# ---------------------------------------------------------------------------
SEP = "=" * 57
print(f"\n{SEP}")
print(f"  quant_arena v1.1.0 -- MVP Calibracion TTT")
print(SEP)
print(f"  Periodos registrados : {len(composition)}")
print(f"  Jugadores unicos     : {jugadores_unicos}")
print(f"  Log-ev. inicial      : {log_ev_inicial:.4f}  "
      f"(sigma={SIGMA_0}, gamma={GAMMA_0})")
print()

t0 = time.perf_counter()
try:
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        resultado = opt.calibrar(composition, times)
    t_total = time.perf_counter() - t0

    if capturados:
        for w in capturados:
            print(f"  [WARN] {w.category.__name__}: {w.message}")
        print()

    sigma_opt = resultado["sigma_optimo"]
    gamma_opt = resultado["gamma_optimo"]
    log_ev_max = resultado["log_evidencia_maxima"]
    mejora = log_ev_max - log_ev_inicial

    print(f"  {'Parametro':<10} {'Inicial':>10} {'Optimo':>10}  {'Delta':>10}")
    print(f"  {'-'*44}")
    print(f"  {'sigma':<10} {SIGMA_0:>10.4f} {sigma_opt:>10.4f}  "
          f"{sigma_opt - SIGMA_0:>+10.4f}")
    print(f"  {'gamma':<10} {GAMMA_0:>10.4f} {gamma_opt:>10.4f}  "
          f"{gamma_opt - GAMMA_0:>+10.4f}")
    print()
    print(f"  Log-ev. maxima       : {log_ev_max:.4f}")
    print(f"  Mejora log-evidencia : {mejora:>+.4f}")
    print(f"  Tiempo optimizacion  : {t_total:.3f} s")

    convergido = (
        sigma_opt > 0.0
        and gamma_opt > 0.0
        and np.isfinite(log_ev_max)
        and mejora >= -0.01   # tolerancia para precision numerica
    )
    estado = "[OK] CONVERGENCIA EXITOSA" if convergido else "[!] REVISAR RESULTADO"
    print(f"\n  Estado : {estado}")

except Exception as exc:
    t_total = time.perf_counter() - t0
    print(f"\n  [ERROR] Calibracion fallo en {t_total:.3f} s")
    print(f"  Detalle: {exc}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"{SEP}\n")
