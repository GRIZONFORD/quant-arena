# TrueSkill Through Time aplicado a quant_arena (v1)

> **Nota de procedencia:** esta explicación se redacta de nuevo en esta
> sesión a partir de los hallazgos reales del proyecto; no es una
> transcripción de una explicación anterior (esa versión se dio en un chat
> de otra sesión cuyo texto no está disponible aquí). Si existe una versión
> previa con matices distintos, pegarla para reconciliar como v2.

## 1. Qué es TrueSkill Through Time (y qué no es)

**TrueSkill Through Time (TTT)** es el método bayesiano de Gustavo
Landfried y Esteban Mocskos
([arXiv:2209.00092](https://arxiv.org/abs/2209.00092)) para estimar la
habilidad de jugadores en competencias por parejas/equipos a lo largo del
tiempo, con dos propiedades que lo distinguen de TrueSkill "clásico":

1. **Suavizado temporal bidireccional** (forward-backward, tipo
   Rauch-Tung-Striebel): la habilidad estimada en el período *t* usa
   información de *todos* los períodos, no solo de los anteriores.
2. **Drift explícito de habilidad**: entre períodos, la habilidad de un
   jugador se difumina con un proceso de Wiener (parámetro `gamma`),
   modelando que la habilidad cambia con el tiempo.

En `quant_arena` cada "jugador" es una estrategia del Zoo, y cada
"partida" es un período de rebalanceo donde las estrategias compiten por
Sharpe (u otra métrica configurable) sobre la ventana rolling reciente.

**Nunca usar los términos "Test-Time Training" ni "Landfield"** — son
confusiones frecuentes con el nombre del método y del autor,
respectivamente, y aparecen incorrectos en borradores previos que hay que
evitar reproducir.

## 2. Por qué TTT y no un ranking simple de Sharpe

Un ranking de Sharpe rolling puro:
- Es ruidoso período a período (ventanas cortas, colas gordas).
- No modela que la habilidad de una estrategia puede degradarse con el
  tiempo (régimen de mercado, decaimiento de un edge).
- No captura incertidumbre — dos estrategias con Sharpe similar pero
  distinta variabilidad histórica no deberían tratarse igual.

TTT resuelve los tres puntos con:
- Un **filtro de Kalman** que suaviza las métricas rolling ruidosas antes
  de alimentarlas al TTT (ver `metricas/filtros.py`).
- El parámetro `gamma` (drift) que permite que la habilidad "olvide"
  desempeño viejo a una tasa calibrable.
- Una distribución posterior completa (media `mu`, desviación `sigma`) por
  estrategia y período, no solo un punto estimado.

## 3. Cómo se usa el posterior en el resto del sistema

- **Ranking / pesos del meta-portafolio**: el método de asignación
  (`metodo='mu_sobre_sigma'` o `'kelly_bayes'`) convierte el posterior TTT
  en pesos de cartera.
- **Kelly bayesiano (§1.2)**: `KellyBayesianSizer` usa `sigma` del
  posterior TTT como término de penalización adicional en el denominador
  (`σ²_ret + κ·σ²_skill_TTT`), así que una estrategia con alta incertidumbre
  de habilidad recibe menos exposición aunque su Sharpe puntual sea bueno.
- **Causalidad estricta**: el Juez (`juez/ttt_juez.py`) exige que ninguna
  actualización de habilidad use información del futuro
  (`CausalidadVioladaError` si se viola), condición necesaria para que el
  backtest walk-forward sea válido.

## 4. Hiperparámetros y su calibración

| Parámetro | Rol | Default en el proyecto |
|---|---|---|
| `sigma` | Incertidumbre inicial (prior) de habilidad | 1.6 |
| `gamma` | Velocidad de drift de habilidad entre períodos | 0.036 |

`TTTJuez.calibrar_hiperparametros()` ajusta ambos vía `OptimizadorTTT`
(L-BFGS-B) en vez de fijarlos a mano. Es opt-in (`calibrar=True` en el
pipeline) porque la calibración añade costo computacional y no cambia el
comportamiento por defecto si no se pide explícitamente.

## 5. Resultado observado en el backtest real

En el primer backtest conjunto de las 11 estrategias (2018–2020,
trimestral — ver `hallazgos_fase_2_v1.md` §2.6), el comportamiento del TTT
fue el esperado: asignó peso equitativo inicial (9.1% a cada una de las 11
estrategias) en el primer rebalanceo, y tras observar el desempeño real
del segundo período bajó a 0% las estrategias con Sharpe muy negativo
(`hmm_garch`, `wavelet_lstm`, entre otras), concentrando peso en las que
mostraban señal de habilidad positiva y estable (`tft_trend`, `neural_ff`,
`momentum_126d`).

## 6. Historial de cambios

| Versión | Fecha | Cambios |
|---|---|---|
| v1 | 2026-08-16 | Creación inicial, redactada desde cero en esta sesión (sin acceso al texto de una explicación previa entregada en otra sesión). |
