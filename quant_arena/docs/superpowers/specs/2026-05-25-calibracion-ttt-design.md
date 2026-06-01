# Design Spec — Módulo de Calibración Empírica de Bayes (v1.1.0)

**Fecha:** 2026-05-25
**Autor:** quant_arena engineering
**Estado:** Aprobado e implementado

---

## Problema

Los hiperparámetros `sigma` (prior de habilidad) y `gamma` (drift de Wiener) del modelo
TrueSkill Through Time son actualmente valores fijos copiados de los datos ATP
(Landfried, 2024). Para el dominio de estrategias cuantitativas con dinámicas de régimen
distintas, estos valores pueden ser subóptimos y deben calibrarse con los datos reales
del backtest.

---

## Solución: Empirical Bayes via maximización de log-evidencia

$$(\sigma^*, \gamma^*) = \arg\max_{\sigma,\,\gamma > 0} \log p(\mathcal{D} \mid \sigma, \gamma)$$

donde $\log p(\mathcal{D} \mid \sigma, \gamma)$ es la log-evidencia marginal del grafo
factorial TTT, aproximada via Expectation Propagation (EP).

---

## Decisiones de diseño

### D1: Fuente de datos — `exportar_historial()`

**Decisión:** `OptimizadorTTT` recibe `(composition, times)` directamente,
obtenidos mediante el nuevo método `TTTJuez.exportar_historial()`.

**Alternativas descartadas:**
- *DataFrame de métricas*: duplicaría la lógica de ranking→composition que ya existe
  en `TTTJuez.registrar_periodo()` (violación de DRY).
- *Tupla pre-construida manual*: impone al llamador conocer el formato interno de TTT.

**Consecuencia:** `AbstractJuez` no se modifica. `exportar_historial()` es un método
concreto adicional en `TTTJuez`.

### D2: Copia defensiva profunda en `exportar_historial()`

```python
return copy.deepcopy(self._composition), list(self._times)
```

- `_composition` → `deepcopy`: `List[List[List[str]]]` tiene contenedores mutables
  en todos los niveles. Una copia superficial dejaría las sub-listas compartidas entre
  el original y el consumidor.
- `_times` → `list()`: suficiente porque `float` es inmutable en Python.

### D3: Optimización en espacio logarítmico

Paramétrizamos $\theta = (\log \sigma, \log \gamma) \in \mathbb{R}^2$ y minimizamos
$f(\theta) = -\log p(\mathcal{D} \mid e^{\theta_0}, e^{\theta_1})$.

**Beneficios:**
1. Garantiza $\sigma, \gamma > 0$ sin restricciones de desigualdad activas.
2. Convierte el dominio semi-infinito $(0, \infty)^2$ en $\mathbb{R}^2$,
   donde L-BFGS-B opera con mayor estabilidad numérica.
3. Previene evaluaciones donde la matriz de covarianza del proceso de Wiener colapsa.

**Método:** `scipy.optimize.minimize` con `L-BFGS-B`, bounds transformados al espacio log.

### D4: Tolerancia EP durante la optimización

`epsilon_ep=0.1`, `max_iter_ep=5` (vs. `epsilon=0.01`, `max_iter=30` en TTTJuez).

Justificación: el costo por evaluación de función domina el tiempo total de optimización.
Una tolerancia más suelta reduce ~6× el costo sin afectar significativamente la
dirección del gradiente durante la búsqueda.

### D5: Manejo de fallos

| Escenario | Comportamiento |
|---|---|
| `composition` con < 2 períodos | `RuntimeWarning` + retorna `(sigma_inicial, gamma_inicial)` |
| EP diverge en un punto | `_evaluar_log_evidencia` captura excepción → retorna `_LOG_EV_FALLBACK = -1e9` |
| Todos los puntos fallan | `RuntimeWarning` (log_ev ≈ -1e9 detectado post-minimización) + retorna iniciales |
| `scipy.minimize` no converge | `RuntimeWarning` + retorna el mejor punto visitado |
| `scipy.minimize` lanza excepción | `RuntimeWarning` + retorna defaults |

---

## Árbol de archivos

```
quant_arena/
├── juez/ttt_juez.py          ← +exportar_historial() con copy.deepcopy
├── calibracion/
│   ├── __init__.py            ← exporta OptimizadorTTT
│   └── optimizador.py         ← OptimizadorTTT.calibrar()
└── tests/
    └── test_optimizador.py    ← 16 tests
```

---

## Flujo de uso canónico

```python
# 1. Ejecutar backtest (genera el historial de competencias)
resultado = engine.ejecutar_walk_forward(...)

# 2. Exportar historial del Juez (ya tiene los périodos registrados)
composition, times = juez.exportar_historial()

# 3. Calibrar hiperparámetros
opt = OptimizadorTTT()
params = opt.calibrar(composition, times)
# → {'sigma_optimo': 1.73, 'gamma_optimo': 0.041, 'log_evidencia_maxima': -312.4}

# 4. Re-instanciar TTTJuez con parámetros óptimos para el siguiente backtest
juez_calibrado = TTTJuez(sigma=params['sigma_optimo'], gamma=params['gamma_optimo'])
```
