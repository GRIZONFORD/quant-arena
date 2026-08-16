# Guión de Video Explicativo — quant-arena (v5, ~16:30, storytelling + audiencia universitaria)

**Nota de alcance (heredada desde v1):** este guión se apoya únicamente en
resultados medidos durante la implementación (§1.1, §1.2, §1.3, §1.5,
corridas reales sobre el parquet de S&P 500).

> **Nota de reconciliación (v1):** al hablar de la tabla resumen, usar
> **194/194 pruebas automatizadas**, no "175/175".

> **Nota sobre v5 — motivo del cambio:** el pedido fue aplicar técnicas de
> marketing/storytelling a la estructura, usando como referencia un
> framework de: gancho inicial → arco Problema/Conflicto/Solución →
> visuales dinámicos → marca académica (la cita del paper, mostrada como
> credencial) → CTA de cierre. v4 ya tenía casi todo el *contenido*
> correcto para eso — lo que faltaba era la *arquitectura narrativa*
> explícita. v5 no cambia las explicaciones en criollo de v4 (Sharpe, HMM,
> GARCH, etc. — esas se mantienen intactas), agrega:
>
> 1. **El gancho pasa a ser la primera escena**, antes del glosario — en
>    marketing nunca se abre con definiciones, se abre con la tensión.
>    Se reescribe además como pregunta directa a la audiencia, en vez de
>    afirmación declarativa.
> 2. **Cinco tarjetas de "acto"** (Problema / Conflicto / Solución
>    bayesiana / Aplicación práctica / Conclusión) que actúan como
>    señalización de capítulos — igual que un video institucional marca
>    sus secciones — para que alguien que se distrae 10 segundos pueda
>    reengancharse sabiendo en qué parte de la historia está.
> 3. **CTA de cierre real**: v3/v4 terminaban en una reflexión ("la
>    respuesta que construimos..."); v5 le agrega una invitación
>    explícita a la audiencia, como pide el framework.
> 4. La cita de Landfried se mantiene destacada visualmente (ya estaba en
>    v1) — es la "marca académica" del framework: la credencial que
>    sostiene la seriedad del proyecto.
>
> Esto agrega ~30-40s más de duración (las 5 tarjetas de acto, ~6-8s cada
> una) — el guión pasa de ~16:00 a ~16:30. Mismo trade-off de siempre: se
> prioriza la claridad estructural sobre la duración exacta.
>
> **Nota de fidelidad guión↔video:** esta tabla comprime/omite dos filas
> técnicas de v4 (la nota sobre la escala de κ sin calibrar, y la
> transición del ajedrez hacia el pool de capital compartido) para que la
> narrativa de storytelling no se sienta cortada por asides técnicos. El
> proyecto Manim real **sí las mantiene** (`escena_20_kappa_nota.py`,
> `escena_21_ajedrez_pool.py`) — cortar una limitación real del código
> por ritmo narrativo iría contra la disciplina de honestidad de todo el
> proyecto. El video termina siendo ~15-20s más largo que esta tabla por
> esa razón; es una discrepancia deliberada, no un error de sincronía.

| Tiempo / Escena Visual | Guión de Locución |
| :--- | :--- |
| **[00:00 – 00:35]** *(Reescrita en v5 — ahora es la primera escena, antes del glosario)* Plano de apertura: curva de equity de una sola estrategia, con un recuadro rojo marcando un tramo donde el Sharpe rodante colapsa sin aviso. | "¿Qué pasaría si el modelo que usamos para invertir no solo midiera resultados pasados, sino que aprendiera en tiempo real de su propia incertidumbre? Toda estrategia cuantitativa tiene una ventana de vigencia. El problema no es que las estrategias fallen — es que fallan *en momentos distintos*, y ninguna medida estática de qué tan bien le fue nos dice cuándo." |
| **[00:35 – 01:05]** *(Antes escena 00 de v4)* Glosario exprés: cuatro tarjetas — SHARPE, HABILIDAD LATENTE, INCERTIDUMBRE, TTT. | "Antes de seguir, cuatro palabras que van a aparecer todo el video. Sharpe: rendimiento dividido por qué tan parejo fue lograrlo. Habilidad latente: algo real que no vemos directo, solo inferimos por resultados — como la inteligencia de una persona. Incertidumbre: cuánto confiamos en una estimación. Y TTT, el modelo que junta las tres. Con eso alcanza para seguir todo lo que viene." |
| **[01:05 – 01:15]** *(Tarjeta de acto nueva en v5)* Texto pantalla completa: "ACTO I — EL PROBLEMA". | *(sin locución — pausa visual de 10s, transición de capítulo)* |
| **[01:15 – 01:45]** Split screen: dos curvas de Sharpe rodante de estrategias distintas, cruzándose varias veces a lo largo de 10 años. | "Los modelos financieros tradicionales asumen una estabilidad que no existe. Si la habilidad no es estable en el tiempo, medirla con un promedio histórico congelado es, en el mejor caso, ruido; en el peor, una falsa sensación de control." |
| **[01:45 – 02:05]** *(Tarjeta de acto nueva en v5)* Texto pantalla completa: "ACTO II — EL CONFLICTO". | *(sin locución — pausa visual de 10s)* |
| **[02:05 – 02:35]** Animación: tabla de ranking simple reordenándose violentamente mes a mes. | "Acá está el conflicto: la solución ingenua es rankear por el Sharpe más reciente y asignar el dinero a la que va primera. Es simple — y es frágil por tres razones concretas." |
| **[02:35 – 03:05]** Tres viñetas: suerte vs. habilidad, sin incertidumbre, sin memoria. | "Uno: en ventanas cortas, un buen resultado puede ser suerte, no ventaja real. Dos: el ranking simple no distingue una estimación precisa de una ruidosa. Tres: no tiene memoria — un trimestre malo puede borrar tres años de evidencia. Los indicadores estáticos, literalmente, engañan." |
| **[03:05 – 03:20]** *(Tarjeta de acto nueva en v5)* Texto pantalla completa: "ACTO III — LA SOLUCIÓN BAYESIANA". | *(sin locución — pausa visual de 12s)* |
| **[03:20 – 04:00]** Corte a texto: "TrueSkill Through Time — Gustavo Landfried, arXiv:2209.00092" destacado como credencial científica. Tachar "Test-Time Training" y "Landfield". | "La estadística bayesiana ofrece una salida: en vez de fingir que la habilidad es un número fijo, la modela como una variable latente que evoluciona. Este proyecto usa TrueSkill Through Time, de Gustavo Landfried — un método bayesiano que actualiza una creencia con cada nuevo resultado, en vez de recalcular todo desde cero. No es Test-Time Training, un campo completamente distinto — confundirlos es citar mal el método central del proyecto." |
| **[04:00 – 04:30]** Diagrama de ajedrez: jugadores = estrategias, partidas = rebalanceos. | "La analogía es un sistema de ranking de ajedrez. Cada estrategia es un jugador, cada rebalanceo una partida. Lo que interesa no es el resultado de una partida aislada, sino la habilidad que revela." |
| **[04:30 – 05:20]** *(Tarjeta de acto nueva en v5, breve, dentro de la misma respiración)* Texto en pantalla: "HACIÉNDOLO TANGIBLE". Animación: curva gaussiana $\mathcal{N}(\mu,\sigma^2)$ angostándose con más partidas. | "Hagámoslo tangible. TTT mantiene, para cada estrategia, una creencia con forma de campana. El centro, mu, es la habilidad estimada; qué tan ancha, sigma, es la incertidumbre. Con pocas partidas, ancha — el modelo admite que no sabe. Con más evidencia, se angosta." |
| **[05:20 – 06:00]** Flecha de una partida reciente ajustando retroactivamente una vieja (Expectation Propagation). | "Y acá está la parte elegante: el algoritmo propaga mensajes hacia adelante *y hacia atrás* sobre todo el historial — 'Through Time'. Un resultado reciente puede refinar lo que creíamos saber hace un año. Ese suavizado bidireccional es la fuente de su poder — y, como vamos a ver, también exige blindarse contra ver el futuro por error." |
| **[06:00 – 06:40]** Código: `pesos_asignacion(metodo='mu_sobre_sigma')`. | "De esa creencia se deriva cuánto capital recibe cada estrategia: mu sobre sigma. Alta habilidad *y* alta certeza reciben más capital que la misma habilidad con mucha incertidumbre. La incertidumbre ya no es ruido descartado — es información para decidir cuánto arriesgar." |
| **[06:40 – 07:10]** Filtro de Kalman antes del Juez TTT (código `motor.py` / `ttt_juez.py`). | "Un detalle que suele pasar desapercibido: antes de competir, el Sharpe de cada estrategia pasa por un filtro de Kalman causal, porque TTT acá no ve el número — ve el orden. El Kalman no le da más precisión — le da un ranking más estable antes de que se vuelva evidencia permanente." |
| **[07:10 – 07:50]** Código: `class TTTJuez(AbstractJuez)`, patrón Adapter. | "Arquitectónicamente, TTTJuez es un Adapter — una pieza que traduce entre el motor y la librería externa. Sustituir TTT por otro sistema no requiere tocar el motor. Es la razón de que el proyecto pueda evolucionar sin reescrituras." |
| **[07:50 – 08:35]** Texto: "§1.5 — ¿Y si el modelo mismo viola sus propios supuestos?" + explicación en criollo de HMM y GARCH. | "Ahora la pregunta que casi ningún proyecto responde: los modelos internos —en particular uno que combina un HMM, que asume 'estados de ánimo' ocultos del mercado, con un GARCH, que predice volatilidad mirando la reciente— hacen supuestos. ¿Se cumplen en los datos reales?" |
| **[08:35 – 09:20]** Tabla en vivo: normalidad (Jarque-Bera, se rechaza) y homocedasticidad (ARCH-LM, se rechaza) sobre S&P 500 real. | "Corrimos la batería completa. Normalidad: se rechaza — los retornos tienen eventos extremos mucho más frecuentes que una campana perfecta. Homocedasticidad: también se rechaza — hay rachas de volatilidad. Ninguno sorprende en finanzas. Lo que importa es que no lo asumimos: lo medimos, y actuamos." |
| **[09:20 – 10:00]** Estacionariedad/independencia + comparación K=3 vs. K=4 (BIC/AIC/CV temporal). | "La estacionariedad no se rechaza. Y encontramos algo más: el número de 'estados de ánimo' del HMM estaba fijo en 3, a mano, sin criterio. Con tres métricas distintas, el óptimo real es 4." |
| **[10:00 – 10:30]** Código: `GARCHModeler(dist='t')` condicional. | "La acción, no solo el diagnóstico: cuando se rechaza normalidad, el modelo cambia automáticamente a una distribución con colas más gordas. Opcional, no rompe nada — pero verificado, no es una promesa." |
| **[10:30 – 10:45]** *(Tarjeta de acto nueva en v5)* Texto pantalla completa: "ACTO IV — APLICACIÓN PRÁCTICA". | *(sin locución — pausa visual de 10s)* |
| **[10:45 – 11:15]** Texto: el backtest original corría sin ninguna gestión de riesgo. | "Acá conectamos la teoría con la realidad. Encontramos el hallazgo más grave del proyecto: existía un control de riesgo completo en el código, pero el motor principal nunca lo usaba. El sistema invertía el cien por ciento del capital, siempre, sin protección." |
| **[11:15 – 12:00]** Curvas de equity reales: Sharpe 0.60→0.73, drawdown −34%→−20%. | "Lo conectamos. El resultado sobre datos reales: Sharpe sube de 0.60 a 0.73, la peor caída se reduce de menos 34 a menos 20 por ciento. Misma estrategia, mismo período, con y sin ese control aplicado a su propio historial real." |
| **[12:00 – 12:45]** Ecuación de Kelly bayesiano con $\sigma^2_{skill,TTT}$ en el denominador. | "Segundo hallazgo: el dinero se repartía sumando cien por ciento sin importar la convicción real del modelo. La corrección usa el criterio de Kelly — cuánto arriesgar dado cuánta ventaja y cuánta certeza tenés de esa ventaja. Cuando el Juez duda, esa duda penaliza directamente la apuesta." |
| **[12:45 – 13:15]** Bug real encontrado y corregido (`_ensamblar_meta`). | "Honestidad de proceso: al verificar con datos reales encontramos que el motor anulaba en la práctica esa exposición reducida. Lo detectamos porque insistimos en correr el backtest real, no solo pruebas automatizadas — y lo corregimos antes de reportar." |
| **[13:15 – 13:30]** *(Tarjeta de acto nueva en v5)* Texto pantalla completa: "ACTO V — CONCLUSIÓN". | *(sin locución — pausa visual de 10s)* |
| **[13:30 – 14:10]** Ecuación de crowding: $\alpha_{efectivo}(t) = \alpha_{bruto}(t) \cdot e^{-\kappa w_i(t)\text{AUM}/\text{ADV}_i}$. | "Todavía falta cerrar algo: nada impide concentrar todo el capital en una sola estrategia. El mecanismo de amontonamiento hace que la ventaja real decaiga con cuánto capital persigue esa misma ventaja, relativa al volumen diario del mercado." |
| **[14:10 – 14:50]** Lazo de realimentación: asigna → decae → reasigna → equilibrio multi-estrategia. | "Eso cierra un lazo genuino: el Juez asigna, la ventaja se erosiona con esa asignación, el resultado del siguiente período lo refleja, y el Juez reasigna. Emerge un equilibrio — la respuesta a por qué no concentrar todo en la mejor del ranking: porque hacerlo degradaría exactamente la ventaja que la hizo ganadora." |
| **[14:50 – 15:30]** Tabla resumen: 194/194 pruebas, Sharpe real, K óptimo real. | "En resumen: control de riesgo con mejora medida y real; Kelly acoplado a la incertidumbre del Juez; supuestos estadísticos validados, no asumidos; y un Juez que no puede ver el futuro por construcción. Ciento noventa y cuatro de ciento noventa y cuatro pruebas, sin ninguna regresión." |
| **[15:30 – 16:00]** Limitaciones abiertas, sin maquillaje. | "Y sin maquillaje: el factor que conecta Kelly con TTT necesita calibración empírica. El amontonamiento asume un volumen diario fijo, cuando cambia día a día. Y nueve de once estrategias todavía no están auditadas en profundidad." |
| **[16:00 – 16:20]** Cierre: la curva de equity inicial, ahora con el control de riesgo aplicado. | "La pregunta con la que abrimos era si la habilidad es estable en el tiempo. La respuesta que construimos no finge que sí — modela explícitamente la incertidumbre, y actúa en consecuencia con ella en cada capa: ranking, tamaño de apuesta, riesgo tolerado." |
| **[16:20 – 16:35]** *(CTA nueva en v5)* Texto en pantalla: "La incertidumbre no se ignora: se modela, y se usa." | "La estadística bayesiana no es solo teoría: es una herramienta que puede redefinir cómo gestionamos riesgo y capital en mercados reales. Los invito a mirar el código, cuestionar los supuestos, y decidir ustedes mismos dónde está el límite entre lo que ya funciona y lo que todavía falta calibrar." |
| **[16:35 – 16:50]** Fade a logo: "TrueSkill Through Time — G. Landfried, arXiv:2209.00092. quant-arena." | "TrueSkill Through Time, de Gustavo Landfried. quant-arena." |

## Historial de cambios

| Versión | Fecha | Cambios |
|---|---|---|
| v1 | 2026-08-16 | Creación inicial, guión tal como fue entregado por el usuario. |
| v2 | 2026-08-16 | Nota técnica del filtro de Kalman, como escena opcional. |
| v3 | 2026-08-16 | Escena del Kalman integrada a la línea de tiempo formal. |
| v4 | 2026-08-16 | Reescritura para audiencia universitaria general: glosas en criollo de cada sigla técnica, dichas en voz alta; escena de glosario exprés. |
| v5 | 2026-08-16 | Pedido explícito: aplicar técnicas de marketing/storytelling (gancho, arco Problema-Conflicto-Solución, marca académica, CTA). El gancho pasa a ser la primera escena (antes iba después del glosario); se agregan 5 tarjetas de "acto" como señalización de capítulos; se agrega un CTA de cierre real que invita a la audiencia a actuar, no solo a reflexionar. El contenido en criollo de v4 se mantiene intacto — v5 es una capa narrativa sobre v4, no un reemplazo de su accesibilidad. ~16:00 → ~16:30. |
