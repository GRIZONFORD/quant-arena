# Hallazgos Fase 2 — quant_arena (v1)

> **Nota de procedencia:** este documento consolida los hallazgos numéricos
> reales medidos durante la Fase 2 (riesgo, Kelly, diagnósticos, Juez TTT,
> crowding y el primer backtest conjunto de las 11 estrategias del Zoo).
> Todas las cifras provienen de ejecuciones reales contra
> `quant_arena/data/sp500_daily_1997_to_today.parquet`, no de estimaciones.
> Terminología obligatoria: **TrueSkill Through Time** (Landfried,
> [arXiv:2209.00092](https://arxiv.org/abs/2209.00092)) — nunca
> "Test-Time Training" ni "Landfield".

## 1. Resumen ejecutivo

La Fase 2 implementó y verificó de punta a punta cinco piezas del roadmap
(§1.1–§1.5): de-risking dinámico y take-profit por ATR, Kelly bayesiano
acoplado a la incertidumbre del posterior TTT, validación de supuestos
estadísticos con selección automática de K para el HMM-GARCH, un Juez TTT
con causalidad estricta y calibración de hiperparámetros, y un modelo de
*crowding* entre estrategias. Todo quedó integrado en `BacktestEngine` como
componentes opcionales (`None` por defecto) para no romper la no-regresión,
con 194/194 tests pasando.

La verificación contra datos reales — no solo contra unit tests — encontró
2 bugs reales que los tests no habían detectado, y confirmó que la mejora
de Sharpe del RiskOverlay, aunque real, **no es estadísticamente
significativa** con la muestra actual (n_trials=2).

## 2. Hallazgos clave

### 2.1 RiskOverlay (§1.1) — Momentum + OLPS-RMR, S&P500, 2015–2020

| Métrica | Sin overlay | Con overlay |
|---|---|---|
| Sharpe | 0.60 | 0.73 |
| Max Drawdown | −34% | −20% |
| Deflated Sharpe Ratio (DSR) | 0.909 | 0.949 |

- PBO (Probability of Backtest Overfitting) = 0.42.
- Con `n_trials=2` el DSR no alcanza el umbral de significancia al 95%.
  **La mejora de Sharpe es real pero no debe reportarse sin este matiz** —
  hace falta un grid de hiperparámetros más amplio (pendiente, ver §4) para
  un PBO con más columnas y mayor poder estadístico.

### 2.2 Kelly bayesiano (§1.2)

`KellyBayesianSizer` dimensiona posiciones con
`μ / (σ²_ret + κ·σ²_skill_TTT)`, sin forzar que la exposición absoluta sume 1.

- Con `κ=1.0` (default) el sizing es **muy conservador**: la escala nativa
  de la incertidumbre TTT (~1.6) domina sobre la varianza de retornos
  (~1e-4), aplastando el numerador. Esto está documentado como necesidad
  de calibración de `κ`, no como un bug.

### 2.3 Diagnósticos de supuestos estadísticos (§1.5)

Sobre S&P500 real (7,394 observaciones):

| Test | Resultado |
|---|---|
| Normalidad | Rechazada (p ≈ 0) |
| ARCH-LM (heterocedasticidad condicional) | Rechazada (p ≈ 6e-155) |
| Estacionariedad | No rechazada |
| Independencia (lag=20) | Rechazada |

Consecuencia directa: `HMMGARCHStrategy` ahora usa `dist='t'` en el GARCH
cuando se rechaza la normalidad (opt-in), en vez de asumir gaussianidad por
defecto.

### 2.4 Selección de K para el HMM (§1.5)

BIC, AIC y CV **coinciden en K=4** como número óptimo de regímenes, frente
al K=3 hardcodeado en la versión original del HMM-GARCH.

### 2.5 Bugs reales encontrados en verificación end-to-end

No estaban en el código original; aparecieron solo al correr contra datos
reales, no contra los unit tests:

1. **RiskOverlay sin lookback de volatilidad suficiente por fold** —
   el overlay subestimaba la vol en folds cortos al inicio de la ventana
   walk-forward.
2. **`_ensamblar_meta` renormalizando los pesos de Kelly a suma=1** —
   anulaba el propósito explícito de Kelly de *no* forzar exposición total
   fija.

Ambos corregidos y cubiertos por tests de regresión.

### 2.6 Primer backtest conjunto de las 11 estrategias del Zoo

Ejecutado en esta sesión (2026-08-15/16), período 2018-01-01 a 2020-12-31,
rebalanceo trimestral (63 días hábiles) — el run mensual completo
2015–2020 se estimó en ~7-8h por el costo de reajustar HMM-GARCH y los
modelos DL en cada fold y se pospuso.

| Estrategia | Sharpe | Sortino | MaxDD | Calmar |
|---|---|---|---|---|
| **META_PORTFOLIO** | **0.23** | 0.25 | **−12.45%** | 0.28 |
| tft_trend | 0.52 | 0.81 | −7.6% | 0.89 |
| neural_ff | 0.41 | 0.50 | −32.7% | 0.22 |
| momentum_126d | 0.38 | 0.34 | −33.9% | 0.25 |
| olps_rmr | 0.37 | 0.42 | −40.2% | 0.20 |
| ppo_rl | 0.36 | 0.37 | −33.9% | 0.23 |
| stgnn_alpha | −0.20 | −0.32 | −7.6% | 0.08 |
| mamba_ssm | −0.30 | −0.42 | −12.4% | −0.01 |
| xgboost_trend | −1.20 | −1.48 | −24.3% | −0.37 |
| llm_sentiment | −1.52 | −1.67 | −67.0% | −0.44 |
| hmm_garch / wavelet_lstm | −3.39 | −13.48 | 0.0% | — |

Hallazgo puntual (no es un bug): `hmm_garch` y `wavelet_lstm` dan métricas
idénticas porque, en este régimen, su señal casi nunca cruza el umbral de
decisión — quedan en cash casi todo el período salvo el peso inicial
equitativo (9.1%) del primer rebalanceo. El Juez TTT las bajó a 0% de peso
desde el segundo rebalanceo en adelante, que es el comportamiento esperado
de un ranking bayesiano que penaliza desempeño real pobre.

Resultados completos (CSV + tear-sheet PDF) en
`results/backtest_zoo_completo/`, commit `e679bd9` en
`claude/quant-arena-ttt-risk-w50b6w`.

## 3. Limitaciones documentadas (no son bugs)

- **ADV dinámica en `crowding.py`**: hoy es un snapshot estático de
  volumen promedio diario; no se actualiza intra-simulación.
- **Kelly con κ=1.0**: conservador por diseño hasta que se calibre contra
  datos reales (ver §4).

## 4. Pendiente / no empezado

- Grid real de hiperparámetros de RiskOverlay/Kelly/crowding sobre el
  dataset real, para un PBO con más de 2 columnas.
- Backtest mensual completo 2015–2020 con las 11 estrategias (el de esta
  sesión fue trimestral 2018–2020 por costo computacional).
- ADV dinámica en `crowding.py`.

## 5. Historial de cambios

| Versión | Fecha | Cambios |
|---|---|---|
| v1 | 2026-08-16 | Creación inicial. Consolida hallazgos de §1.1–§1.5, los 2 bugs reales encontrados en verificación end-to-end, y los resultados del primer backtest conjunto de las 11 estrategias del Zoo (2018–2020, trimestral). |
