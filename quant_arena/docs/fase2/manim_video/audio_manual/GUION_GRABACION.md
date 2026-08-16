# Guión de grabación — audio manual

Un archivo `NN.mp3` por cada bloque, en el orden exacto de esta lista.

Guardalos en `audio_manual/<NombreEscena>/NN.mp3` (la carpeta ya está creada para cada una).


## Glosario  (`audio_manual/Glosario/`)

**01.mp3**
> Antes de arrancar, cuatro palabras que van a aparecer todo el video. Sharpe: una forma de medir si a una inversión le fue bien, dividiendo cuánto ganó por cuánto se sacudió para lograrlo — más alto es mejor.

**02.mp3**
> Habilidad latente: algo que existe pero no se puede medir directamente, como la inteligencia de una persona — solo la inferimos observando resultados.

**03.mp3**
> Incertidumbre: qué tan seguros estamos de una estimación, no solo cuál es esa estimación.

**04.mp3**
> Y TTT, TrueSkill Through Time, el modelo que junta las tres ideas anteriores para armar un ranking honesto. Con eso alcanza para seguir todo lo que sigue.


## Apertura  (`audio_manual/Apertura/`)

**01.mp3**
> ¿Qué pasaría si el modelo que usamos para invertir no solo midiera resultados pasados, sino que aprendiera en tiempo real de su propia incertidumbre?

**02.mp3**
> Toda estrategia cuantitativa tiene una ventana de vigencia. El problema no es que las estrategias fallen — es que fallan en momentos distintos, y ninguna medida estática de qué tan bien le fue nos dice cuándo.


## SplitSharpe  (`audio_manual/SplitSharpe/`)

**01.mp3**
> quant-arena parte de esa premisa: si la habilidad no es estacionaria, medirla con un promedio histórico congelado es, en el mejor caso, ruido; en el peor, una falsa sensación de control. Necesitábamos un modelo que tratara la habilidad como lo que es — una variable latente que cambia con el tiempo.


## RankingTabla  (`audio_manual/RankingTabla/`)

**01.mp3**
> La solución ingenua es rankear por Sharpe rodante y asignar capital al líder. Es simple, y es frágil por tres razones concretas, no genéricas.


## TresProblemas  (`audio_manual/TresProblemas/`)

**01.mp3**
> Uno: en ventanas cortas, un Sharpe alto puede ser tres apuestas ganadoras seguidas, no una ventaja real.

**02.mp3**
> Dos: el ranking simple no distingue una estimación precisa de una ruidosa — trata igual un Sharpe de 0.8 con veinte observaciones que con doscientas.

**03.mp3**
> Tres: no tiene memoria estructurada — un trimestre malo puede borrar tres años de evidencia acumulada de que una estrategia es genuinamente buena.


## TTTTitulo  (`audio_manual/TTTTitulo/`)

**01.mp3**
> Aquí es importante ser precisos, porque es el tipo de error que un tribunal ataca primero: este proyecto usa TrueSkill Through Time, de Gustavo Landfried — un método bayesiano, es decir, que no calcula la habilidad de cero cada vez, sino que actualiza una creencia previa con cada nuevo resultado, usando un algoritmo llamado Expectation Propagation.

**02.mp3**
> No es Test-Time Training, que es un campo completamente distinto de inteligencia artificial. Confundirlos no es un detalle cosmético, es citar mal el método central del proyecto.


## Ajedrez  (`audio_manual/Ajedrez/`)

**01.mp3**
> La analogía correcta es un sistema de ranking de ajedrez. Cada estrategia es un jugador. Cada período de rebalanceo es una partida.

**02.mp3**
> El Sharpe rodante decide el resultado de esa partida — quién quedó primero, quién último. Y como en el ajedrez, lo que nos interesa no es el resultado de una partida aislada, sino la habilidad que ese resultado revela.


## Gaussianas  (`audio_manual/Gaussianas/`)

**01.mp3**
> TrueSkill Through Time mantiene, para cada estrategia, una creencia — no un número, una distribución de probabilidad con forma de campana. El centro de la campana, mu, es la habilidad estimada; qué tan ancha es, sigma, es cuánta incertidumbre tenemos sobre esa estimación.

**02.mp3**
> Con pocas partidas, sigma es grande — el modelo admite que no sabe. Con más evidencia, se angosta. Eso ya resuelve el problema dos del ranking ingenuo: la incertidumbre es explícita, no implícita.


## EPFlecha  (`audio_manual/EPFlecha/`)

**01.mp3**
> Y resuelve el problema tres, la memoria: el algoritmo de Expectation Propagation propaga mensajes hacia adelante y hacia atrás sobre todo el historial de partidas — 'Through Time'.

**02.mp3**
> Un resultado reciente puede refinar retroactivamente lo que creíamos saber sobre la habilidad de esa estrategia hace un año. Ese suavizado bidireccional es la fuente de su poder — y, como veremos en el minuto 8, también la fuente del riesgo de look-ahead que tuvimos que blindar explícitamente.


## MuSobreSigma  (`audio_manual/MuSobreSigma/`)

**01.mp3**
> De esa creencia se deriva directamente cuánto capital recibe cada estrategia: el cociente mu sobre sigma — la habilidad estimada dividida por la incertidumbre.

**02.mp3**
> Estrategias con alta habilidad estimada y alta certeza reciben más capital que estrategias con la misma habilidad estimada pero mucha incertidumbre. La incertidumbre ya no es ruido que se descarta — es información para decidir cuánto arriesgar.


## Kalman  (`audio_manual/Kalman/`)

**01.mp3**
> Un detalle que suele pasar desapercibido: antes de que el Sharpe de cada estrategia entre a competir, pasa por un filtro de Kalman causal.

**02.mp3**
> ¿Por qué? Porque TrueSkill Through Time, tal como lo usamos acá, no ve el número — ve el orden: sabe que una estrategia quedó tercera de once, no que sacó cero coma setenta y tres de Sharpe.

**03.mp3**
> Un ranking basado en una ventana corta y ruidosa se voltea todo el tiempo por pura casualidad estadística. El Kalman no le da más precisión a TTT — le da un ranking más estable antes de que ese ranking se vuelva evidencia permanente en el historial de partidas.


## Adapter  (`audio_manual/Adapter/`)

**01.mp3**
> Arquitectónicamente, TTTJuez es un Adapter sobre el paquete trueskillthroughtime — el motor de backtesting depende de la interfaz AbstractJuez, no de la librería externa.

**02.mp3**
> Eso significa que sustituir TTT por, digamos, un sistema Elo, no requiere tocar una sola línea de motor.py. Es la razón de que este proyecto pueda evolucionar sin reescrituras.


## PreguntaSupuestos  (`audio_manual/PreguntaSupuestos/`)

**01.mp3**
> Ahora, la pregunta que casi ningún proyecto de este tipo responde: los modelos internos de las estrategias —en particular la que combina un HMM de régimen con GARCH— asumen normalidad, estacionariedad, homocedasticidad. ¿Se cumplen esos supuestos en los datos reales que estamos usando?


## HMMGarchGlosa  (`audio_manual/HMMGarchGlosa/`)

**01.mp3**
> Un momento para explicar dos términos que van a aparecer seguido. Un HMM, o modelo de Markov oculto, asume que el mercado pasa por distintos 'estados de ánimo' — calma, pánico, tendencia — que no observamos directamente, pero podemos inferir mirando cómo se comportan los precios.

**02.mp3**
> Un GARCH es un modelo que predice cuánto va a variar el precio mañana, mirando cuánto varió en los días recientes — la idea de que la volatilidad viene en rachas, no es constante.


## TablaNormalidad  (`audio_manual/TablaNormalidad/`)

**01.mp3**
> Corrimos la batería completa de contrastes estadísticos sobre los retornos diarios reales del S&P 500.

**02.mp3**
> Normalidad, o sea si los datos siguen esa forma de campana que mencionamos antes: se rechaza, con un test de Jarque-Bera cuyo resultado numérico es treinta y un mil — cuanto más grande ese número, más lejos está de parecerse a una campana perfecta. En criollo: los retornos financieros tienen eventos extremos mucho más frecuentes de lo que una campana predeciría.

**03.mp3**
> Homocedasticidad, o sea que la variabilidad del mercado sea pareja en el tiempo: también se rechaza, con una probabilidad de que sea casualidad de prácticamente cero — hay 'rachas' de volatilidad, períodos tranquilos y violentos que se agrupan. Ninguno de los dos es sorprendente en finanzas. Lo que importa es que no lo asumimos: lo medimos, y actuamos en consecuencia.


## TablaKOptimo  (`audio_manual/TablaKOptimo/`)

**01.mp3**
> La estacionariedad —que las propiedades estadísticas no cambien con el tiempo— no se rechaza, coherente con que ya trabajamos con retornos en vez de precios crudos.

**02.mp3**
> Y encontramos algo más interesante: el número de 'estados de ánimo' ocultos que el HMM asume estaba fijo en 3 en el código original, sin ningún criterio de selección — alguien lo puso a mano.

**03.mp3**
> Corrimos un barrido probando distintos valores, evaluados con tres criterios que castigan a un modelo por ser innecesariamente complicado — las tres métricas coinciden en que el número óptimo sobre el historial completo es 4, no 3.


## GarchDistT  (`audio_manual/GarchDistT/`)

**01.mp3**
> La acción, no solo el diagnóstico: cuando se rechaza normalidad, el modelo ahora cambia automáticamente a emisiones t de Student en el GARCH, en vez de asumir gaussianidad por defecto.

**02.mp3**
> Es opt-in — no rompe el comportamiento original — pero está disponible y verificado, no es una promesa.


## TextoRiesgo  (`audio_manual/TextoRiesgo/`)

**01.mp3**
> Con los supuestos auditados, encontramos un fallo real en el código — el hallazgo más grave de todo el proyecto: existía una clase completa para controlar el riesgo, con reducción de exposición y corte de pérdidas, pero el motor principal nunca la usaba. El sistema invertía el cien por ciento del capital, siempre, sin ninguna protección activa. Fue justamente por encontrar ese fallo que hicimos la modificación que van a ver ahora.


## EquityRiskOverlay  (`audio_manual/EquityRiskOverlay/`)

**01.mp3**
> La conectamos, de forma opcional, para no romper ningún resultado previo.

**02.mp3**
> El resultado sobre datos reales: el Sharpe anualizado sube de 0.60 a 0.73, y la peor caída — desde el punto más alto hasta el más bajo antes de recuperarse, lo que llamamos drawdown — se reduce de menos 34 por ciento a menos 20 por ciento. No es un backtest sintético — es la misma estrategia, el mismo período, con y sin ese control de riesgo aplicado a su propio historial real.


## KellyEcuacion  (`audio_manual/KellyEcuacion/`)

**01.mp3**
> El segundo hallazgo era más sutil: el dinero siempre se repartía sumando cien por ciento entre las estrategias, sin importar cuánta convicción tuviera realmente el modelo.

**02.mp3**
> Un ejemplo simple antes de ir a la fórmula real: imaginen una apuesta que ganan el sesenta por ciento de las veces, duplicando lo que arriesgan en cada una.

**03.mp3**
> El criterio de Kelly dice: arriesguen dos veces esa probabilidad, menos uno — acá, veinte por ciento del capital en cada apuesta. Ni todo, porque tarde o temprano se arruinan; ni poco, porque dejan plata sobre la mesa.

**04.mp3**
> Así lo aplicamos acá: la fracción óptima de capital es la ventaja estimada, dividida por qué tan riesgosa es esa ventaja. Pero esta vez el riesgo no es solo la variabilidad de los retornos — sumamos también la incertidumbre del posterior de habilidad que TTT ya calcula.

**05.mp3**
> Cuando el Juez tiene poca convicción en una estrategia, esa incertidumbre penaliza directamente cuánto se le arriesga.


## BugFix  (`audio_manual/BugFix/`)

**01.mp3**
> Y aquí una honestidad de proceso, no de resultado: al verificar esto con datos reales encontramos que el motor renormalizaba los pesos a suma uno incondicionalmente, lo cual anulaba en la práctica la exposición reducida de Kelly.

**02.mp3**
> Lo detectamos precisamente porque insistimos en correr el backtest real, no solo los tests unitarios — y lo corregimos antes de reportar el resultado final.


## KappaNota  (`audio_manual/KappaNota/`)

**01.mp3**
> Con el kappa por defecto, la penalización es agresiva — el modelo se vuelve muy conservador porque la escala de incertidumbre de TTT no está calibrada contra la escala de los retornos. Lo documentamos como limitación abierta, no como resultado final.


## AjedrezPool  (`audio_manual/AjedrezPool/`)

**01.mp3**
> Todo lo anterior corrige cómo medimos y cómo dimensionamos. Pero hasta acá, el ranking sigue siendo eso: un ranking.

**02.mp3**
> Las once estrategias no interactúan entre sí — TTT las puntúa, pero no hay ninguna fuerza económica que impida concentrar todo el capital en una sola. El siguiente hito, todavía en diseño, cierra esa brecha.


## CrowdingEcuacion  (`audio_manual/CrowdingEcuacion/`)

**01.mp3**
> El mecanismo propuesto es lo que se llama crowding — amontonamiento. Piensen en un recital que vende mil boletos por día. Si alguien intenta comprar diez mil de una sola vez, el precio se dispara antes de que termine de comprar — se estorba a sí mismo.

**02.mp3**
> Eso mismo le pasa a una estrategia cuantitativa: si le meten más plata de la que el mercado puede absorber en un día, la propia operación mueve el precio en contra, y la ventaja que tenían se evapora.

**03.mp3**
> Formalizado, la ventaja real de una estrategia decae exponencialmente con la fracción de capital que recibe, relativa al volumen diario del activo que opera.

**04.mp3**
> AUM es cuánta plata metieron; ADV es cuántos 'boletos' vende el mercado por día. Cuando AUM se acerca a ADV, la ventaja efectiva se derrumba — es el mismo principio de rendimientos decrecientes a escala documentado por Berk y Green.


## FeedbackLoop  (`audio_manual/FeedbackLoop/`)

**01.mp3**
> Eso cierra un lazo de realimentación genuino: el Juez asigna, el alpha se erosiona con la asignación, el KPI del siguiente período refleja esa erosión, y el Juez reasigna.

**02.mp3**
> Ya no hay una sola estrategia ganadora — emerge un equilibrio. Y es la respuesta directa a la pregunta obvia: '¿por qué no concentran todo el capital en la mejor estrategia del ranking?' — porque hacerlo, en un mercado real, degradaría exactamente la ventaja que la hizo ganadora.


## TablaResumen  (`audio_manual/TablaResumen/`)

**01.mp3**
> En resumen, lo que está construido y verificado hoy: control de riesgo conectado, con mejora medida en Sharpe y en la peor caída, sobre datos reales; el criterio de Kelly acoplado a la incertidumbre del Juez, con su limitación de calibración documentada.

**02.mp3**
> Validación de los supuestos estadísticos con resultados reales sobre el S&P 500, no simulados; y un Juez TTT que no puede ver el futuro por construcción, con costo computacional acotado y ajuste de sus parámetros basado en evidencia. Ciento noventa y cuatro de ciento noventa y cuatro pruebas automatizadas pasando, sin ninguna regresión sobre el comportamiento original.


## Limitaciones  (`audio_manual/Limitaciones/`)

**01.mp3**
> Y las limitaciones, sin maquillaje: el factor kappa que conecta la incertidumbre de TTT con la varianza de retornos necesita calibración empírica que todavía no hicimos.

**02.mp3**
> El mecanismo de crowding ya está implementado y con tests propios, pero la demanda de volumen diario sigue siendo un snapshot estático, no dinámico.

**03.mp3**
> Y de las once estrategias del Zoo, auditamos en profundidad tres — momentum, OLPS-RMR y HMM-GARCH. Las otras ocho, sobre todo las que dependen de librerías de deep learning pesadas, corrieron todas juntas en un backtest real, pero no recibieron la misma revisión línea por línea — quedan para el siguiente ciclo.


## Cierre  (`audio_manual/Cierre/`)

**01.mp3**
> La pregunta con la que abrimos era si la habilidad de una estrategia es estacionaria. La respuesta que construimos no es un modelo que finge que sí lo es — es un sistema que modela explícitamente la incertidumbre sobre esa habilidad,

**02.mp3**
> y que ahora, además, actúa en consecuencia con esa incertidumbre en cada capa: en el ranking, en el tamaño de la apuesta, y en el riesgo que tolera.


## CTA  (`audio_manual/CTA/`)

**01.mp3**
> La estadística bayesiana no es solo teoría: es una herramienta que puede redefinir cómo gestionamos riesgo y capital en mercados reales.

**02.mp3**
> Los invito a mirar el código, cuestionar los supuestos, y decidir ustedes mismos dónde está el límite entre lo que ya funciona y lo que todavía falta calibrar.


## Logo  (`audio_manual/Logo/`)

**01.mp3**
> TrueSkill Through Time, de Gustavo Landfried. quant-arena.


## Tarjetas de ACTO (sin locución — no hace falta grabar nada)

Acto1Problema, Acto2Conflicto, Acto3Solucion, Acto4Aplicacion, Acto5Conclusion: son transiciones mudas de ~10-12s, no llevan audio.
