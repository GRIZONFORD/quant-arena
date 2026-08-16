# Defensa Q&A — quant-arena (v1)

> **Nota de procedencia:** esta sesión no tiene acceso al texto de una
> defensa Q&A entregada en un chat anterior — solo al resumen compacto de
> estado y al guión de 15 min que el usuario acaba de pegar. Este documento
> es un borrador nuevo (v1), construido a partir de esas dos fuentes y de
> los hallazgos reales del proyecto. Si existe una versión previa con
> preguntas o respuestas distintas, pegarla para reconciliar como v2 en vez
> de descartar esta.

Formato: pregunta anticipada de tribunal/evaluador → respuesta directa,
con la cifra o archivo que la respalda.

## 1. Terminología y método

**P: ¿Por qué "TrueSkill Through Time" y no "Test-Time Training"? ¿No es
lo mismo con otro nombre?**
R: No, son campos distintos. TrueSkill Through Time (Landfried,
[arXiv:2209.00092](https://arxiv.org/abs/2209.00092)) es inferencia
bayesiana de habilidad latente vía Expectation Propagation sobre un grafo
factorial — se usa para *rankear* estrategias en el tiempo. Test-Time
Training es una técnica de adaptación de pesos de un modelo *durante la
inferencia*, de un campo completamente distinto (deep learning online).
Confundirlos sería citar mal el método central del proyecto, no un error
cosmético — de ahí que el proyecto tenga una regla explícita de
terminología obligatoria en todo el código y la documentación.

**P: ¿Por qué el suavizado bidireccional ("Through Time") no introduce
look-ahead bias en el backtest?**
R: Porque `TTTJuez` exige causalidad estricta: cualquier actualización de
habilidad que use información del futuro dispara
`CausalidadVioladaError` (`juez/ttt_juez.py`). El suavizado
forward-backward del algoritmo de EP opera *dentro* de la historia ya
observada hasta la fecha de corte de cada rebalanceo — nunca sobre datos
posteriores a esa fecha. Es una garantía verificada por tests, no solo
una intención de diseño.

## 2. Significancia estadística

**P: El Sharpe sube de 0.60 a 0.73 con el RiskOverlay — ¿eso es
significativo o podría ser ruido?**
R: Con la evidencia actual, **no podemos afirmar que sea significativo al
95%**. El Deflated Sharpe Ratio pasa de 0.909 a 0.949 con solo
`n_trials=2`, y el PBO (Probability of Backtest Overfitting) es 0.42 —
alto. La mejora es real en la muestra medida, pero el número de
configuraciones probadas es insuficiente para descartar sobreajuste con
confianza. Este matiz se reporta siempre junto con la cifra de Sharpe, no
por separado (ver `hallazgos_fase_2_v1.md` §2.1).

**P: ¿Entonces por qué reportan la mejora si no es significativa?**
R: Porque el objetivo de esta fase era demostrar que la arquitectura
(inyección de `RiskOverlay` sin romper no-regresión) funciona
extremo-a-extremo contra datos reales, no probar una ventaja estadística
definitiva. Esa prueba requiere el grid de hiperparámetros más amplio que
está listado como pendiente — con más columnas, el PBO gana poder
estadístico.

## 3. Kelly bayesiano y κ

**P: ¿Por qué el Kelly bayesiano con κ=1.0 apenas invierte capital?**
R: Porque `σ²_skill_TTT` vive en la escala interna de TTT (~1.6), varios
órdenes de magnitud por encima de `σ²_retornos` (~1e-4). Con κ=1.0 ese
término domina el denominador y aplasta el tamaño de la apuesta. Es un
resultado esperado de la fórmula, no un bug — está documentado
explícitamente como necesidad de calibración empírica de κ, pendiente.

**P: ¿Por qué no fijaron un κ calibrado antes de reportar resultados?**
R: Porque calibrar κ sin un grid de validación adecuado sería ajustar un
hiperparámetro al mismo dataset donde se reporta el resultado — el mismo
riesgo de overfitting que ya señala el PBO=0.42 del RiskOverlay. Se
prefirió dejarlo documentado como limitación abierta a reportar un número
optimista sin la validación correspondiente.

## 4. Diagnósticos estadísticos

**P: ¿Por qué el HMM-GARCH original usaba K=3 si el óptimo es K=4?**
R: K=3 estaba hardcodeado sin ningún criterio de selección. Al correr
`diagnostics/model_selection.py` con BIC, AIC y validación cruzada
temporal sobre el historial completo del S&P 500, las tres métricas
coinciden en K=4. El fix es opt-in: la estrategia puede seguir usando K=3
si así se configura, pero por defecto ahora selecciona K automáticamente.

**P: Rechazan normalidad y homocedasticidad — ¿eso invalida los resultados
previos que asumían lo contrario?**
R: No los invalida retroactivamente (esos resultados siguen siendo lo que
midieron), pero sí motiva el cambio a `dist='t'` en el GARCH cuando se
rechaza normalidad, disponible como opción verificada. El punto central es
que ahora se *mide* el supuesto en vez de asumirlo silenciosamente.

## 5. Bugs encontrados en verificación

**P: ¿Cómo encontraron bugs que los 194 tests no detectaron?**
R: Corriendo el pipeline completo contra el parquet real de S&P 500
(1997–2026, 7,394 observaciones), no solo contra fixtures sintéticos de
los unit tests. Ahí aparecieron dos bugs reales: (1) el RiskOverlay sin
suficiente lookback de volatilidad en folds cortos al inicio de la
ventana walk-forward, y (2) `_ensamblar_meta` renormalizando los pesos de
Kelly a suma=1, lo cual anulaba el propósito explícito de Kelly de no
forzar exposición total fija. Ambos se corrigieron y quedaron cubiertos
por tests de regresión antes de reportar cualquier resultado final.

## 6. Crowding (§1.4) y su estado real

**P: El guión presenta crowding como diseño, pero el estado del proyecto
dice que `crowding.py` y `ArenaCrowding` ya están implementados. ¿Cuál es
la verdad?**
R: Ambas cosas son ciertas en momentos distintos: el guión de 15 min se
escribió *antes* de implementar §1.4 (por eso lo presenta explícitamente
como roadmap, no como resultado — mismo estándar de honestidad del resto
del proyecto). Después de esa grabación se completó `CrowdingModel`
(decaimiento de Berk & Green) y `ArenaCrowding` (demanda agregada +
solapamiento coseno entre estrategias), con tests pasando. Al defender el
proyecto hoy, la cifra correcta es: **implementado**, no solo diseñado.
El guión queda como registro histórico de un estado anterior del
proyecto, no se reescribe.

**P: ¿Qué limitación real tiene el crowding implementado?**
R: La demanda de volumen diario promedio (ADV) es un snapshot estático —
no se actualiza intra-simulación. Documentado, no es un bug.

## 7. Alcance del Zoo y del backtest conjunto

**P: ¿Corrieron las 11 estrategias juntas contra datos reales?**
R: Sí, en un backtest walk-forward 2018–2020 con rebalanceo trimestral
(commit `e679bd9`, rama `claude/quant-arena-ttt-risk-w50b6w`). El run
mensual 2015–2020 completo se estimó en 7-8 horas por el costo de
reajustar HMM-GARCH y los modelos de deep learning en cada fold, y se
pospuso a favor de un run trimestral más corto que sí terminó. Resultado:
Sharpe del meta-portafolio 0.23, MaxDD −12.45%, con el Juez TTT bajando a
0% de peso las estrategias de peor desempeño real desde el segundo
rebalanceo — comportamiento esperado del ranking bayesiano, no un fallo.

**P: ¿Por qué `hmm_garch` y `wavelet_lstm` dan exactamente el mismo
resultado?**
R: No es un bug compartido ni una excepción silenciada — se verificó
directamente en el código de ambas estrategias. En este régimen real,
ninguna de las dos cruza su umbral de decisión en casi ningún rebalanceo,
así que ambas quedan en cash todo el período salvo el peso inicial
equitativo (9.1%) del primer rebalanceo, que es idéntico para cualquier
estrategia con el mismo universo. Es una coincidencia de comportamiento
plano, no un error de instrumentación — verificado ejecutando
`generar_señales()` de ambas clases directamente sobre los datos reales.

## 8. Arquitectura

**P: ¿Por qué todo (`risk_overlay`, `position_sizer`, `tp_rule`,
`crowding`) es opcional (`None` por defecto) en `BacktestEngine`?**
R: Para que ninguna pieza nueva rompa la no-regresión de resultados ya
reportados. Cada componente se activa explícitamente por quien construye
el engine; el comportamiento original del motor queda intacto si no se
inyecta nada. Es la misma razón por la que `TTTJuez` es un Adapter sobre
`AbstractJuez` — sustituir el motor de ranking no debería requerir tocar
`motor.py`.

## Historial de cambios

| Versión | Fecha | Cambios |
|---|---|---|
| v1 | 2026-08-16 | Creación inicial. Borrador nuevo construido a partir del guión de 15 min y los hallazgos reales de la Fase 2; sin acceso al texto de una defensa Q&A entregada en una sesión anterior. |
