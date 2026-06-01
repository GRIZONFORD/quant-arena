# Correcciones de API — TrueSkill Through Time (Landfried & Mocskos, 2025)

Análisis de desviaciones entre los módulos implementados y la API oficial
documentada en `logistic.py`, `example.py` y el paper JSS v112i06.

---

## Firma real de `History()` (verificada contra la librería)

```python
History(
    composition,          # requerido
    results    = [],      # opcional: [[1.,0.],[0.,1.]] — ORDINAL por defecto
    times      = [],      # opcional: floats en días
    priors     = {},      # opcional: Dict[str, Player]
    mu         = 0.0,
    sigma      = 6.0,     # DEFAULT es 6.0, NO 1.0
    beta       = 1.0,
    gamma      = 0.03,
    p_draw     = 0.0,
    weights    = []
)
```

**Parámetros que NO existen:**
- `obs` → NO EXISTE. Causaba `TypeError` silencioso en try/except.
- El resultado "Continuous" del paper de Guo et al. (2012) **no está implementado** en la librería actual.

---

## Firma real de `convergence()` (verificada)

```python
h.convergence(
    epsilon    = 1e-6,   # DEFAULT es 1e-6, NO 1e-3
    iterations = 30,     # DEFAULT es 30, NO 6
    verbose    = True
)
# RETORNA: ((step_mu, step_sigma), n_iter)
# NO retorna: (step_scalar, n_iter)
```

---

## Métodos públicos reales de `History` (verificados)

```
convergence, learning_curves, log_evidence,
agents, batches, gamma, iteration, mu, p_draw,
sigma, size, time, trueskill
```

**Métodos que NO existen:**
- `geometric_mean()` → NO EXISTE. El paper lo menciona como concepto pero
  el método real es `log_evidence()`.
- `add_fold()` → NO EXISTE. Es solo una convención del motor propio.

---

## Patrón canónico de `learning_curves()` (de logistic.py líneas 23-24)

```python
# OFICIAL — exactamente como Landfried lo usa:
mu    = [tp[1].mu    for tp in h.learning_curves()["a"]]
sigma = [tp[1].sigma for tp in h.learning_curves()["a"]]
# tp[0] → float (time_days)
# tp[1] → Gaussian con .mu y .sigma
```

---

## Regla sobre `results` y `p_draw`

| results               | p_draw   | Comportamiento                          |
|----------------------|----------|-----------------------------------------|
| `[1.,0.]` / `[0.,1.]`| 0.0      | ✅ Patrón canónico de Landfried          |
| `[1.,0.]` / `[0.,1.]`| 0.25     | ✅ Con empates permitidos                |
| `[0.55, 0.45]` (tied)| 0.0      | ❌ ValueError (scores iguales sin draw) |
| `[0.55, 0.45]`       | 0.10     | ✅ Funciona pero NO es el modelo Guo    |

**Conclusión:** para resultados continuos, usar `[1.,0.]/[0.,1.]` según quién
tenga mayor score, y configurar `p_draw > 0` solo si hay empates posibles.

---

## Tabla de correcciones por módulo

### `walk_forward_engine_con_TTT.py` — `TTTAccumulator.fit_ttt()`

```python
# ANTES (incorrecto):
h = History(
    composition=self.composition,
    results=self.results,
    times=self.times,
    mu=0.0,
    sigma=self.ttt_sigma,
    gamma=self.ttt_gamma,
    obs=self.obs,           # ← NO EXISTE
)
step, n_iter = h.convergence(
    iterations=iterations, epsilon=epsilon, verbose=verbose
)
if step > epsilon:          # ← step es una tupla, no escalar

# DESPUÉS (correcto):
h = History(
    composition = self.composition,
    results     = self.results,
    times       = self.times,
    mu          = 0.0,
    sigma       = self.ttt_sigma,
    gamma       = self.ttt_gamma,
    p_draw      = 0.0,      # ← parámetro correcto
)
(step_mu, step_sigma), n_iter = h.convergence(
    iterations=iterations, epsilon=epsilon, verbose=verbose
)
step = max(step_mu, step_sigma)   # ← desempaquetar tupla primero
if step > epsilon:
```

### `TTTAccumulator.calibrate_ttt_parameters()` — `geometric_mean()` no existe

```python
# ANTES (incorrecto — geometric_mean no existe):
score: float = h_calib.geometric_mean()

# DESPUÉS (correcto — usar log_evidence):
score: float = h_calib.log_evidence()
# log_evidence() es la log-verosimilitud marginal: mayor = mejor ajuste
# Equivale conceptualmente a log(geometric_mean) pero es el método real.
```

### `TTTAccumulator.add_fold()` — resultados ordinales

```python
# ANTES (incorrecto — retornos compuestos como scores continuos):
self.results.append([strat_ret, bench_ret])
self.obs.append("Continuous")   # obs no existe

# DESPUÉS (correcto — patrón canónico de Landfried):
if strat_ret >= bench_ret:
    self.results.append([1., 0.])   # estrategia gana
else:
    self.results.append([0., 1.])   # benchmark gana
# Sin obs — no existe ese parámetro
```

> **Alternativa avanzada:** Para preservar la magnitud del diferencial
> (alpha cuantitativo), normalizar los scores a [0,1] y usar `p_draw > 0`:
> ```python
> # Normalización suave: sigmoid del diferencial
> import math
> diff = strat_ret - bench_ret
> score_a = 1.0 / (1.0 + math.exp(-diff * 10))
> score_b = 1.0 - score_a
> self.results.append([score_a, score_b])
> # + p_draw=0.05 en History() para manejar scores similares
> ```

### `simulation_engine.py` — `TTTSimulator._fit_ttt()`

```python
# ANTES (incorrecto):
h = History(
    composition = self._composition,
    results     = self._results,
    times       = self._times,
    mu          = self.ttt_mu,
    sigma       = self.ttt_sigma,
    gamma       = self.ttt_gamma,
    obs         = self._obs,    # ← NO EXISTE
)
step, n_iter = h.convergence(...)
if step > epsilon:

# DESPUÉS (correcto):
h = History(
    composition = self._composition,
    results     = self._results,
    times       = self._times,
    mu          = self.ttt_mu,
    sigma       = self.ttt_sigma,
    gamma       = self.ttt_gamma,
    p_draw      = self._p_draw,  # configurable, default 0.0
)
(step_mu, step_sigma), n_iter = h.convergence(
    iterations=iterations, epsilon=epsilon, verbose=verbose
)
step = max(step_mu, step_sigma)
```

### `analyzer.py` — `compute_ttt_geometric_mean()`

```python
# ANTES (incorrecto — geometric_mean no existe):
def compute_ttt_geometric_mean(history_obj) -> float:
    return float(history_obj.geometric_mean())

# DESPUÉS (correcto — log_evidence es el método real):
def compute_ttt_log_evidence(history_obj) -> float:
    """
    Log-evidencia marginal del modelo ajustado.
    history_obj.log_evidence() = sum(log P(result_t | history_{<t}))
    Mayor valor → mejor ajuste del modelo a los datos.
    Ref: Landfried & Mocskos (2025), JSS v112i06, Eq. 9.
    """
    if history_obj is None:
        return float("nan")
    try:
        return float(history_obj.log_evidence())
    except Exception as exc:
        logger.error(f"log_evidence error: {exc}")
        return float("nan")
```

---

## Parámetros por defecto correctos para el motor walk-forward

Según el paper (Figura 4 y Section 2):

| Parámetro | Default oficial | Recomendado ATP | Recomendado trading |
|-----------|----------------|-----------------|---------------------|
| `mu`      | 0.0            | 0.0             | 0.0                 |
| `sigma`   | **6.0**        | 1.6             | 1.0                 |
| `beta`    | 1.0            | 1.0             | 1.0                 |
| `gamma`   | 0.03           | 0.036           | 0.02–0.05           |
| `p_draw`  | 0.0            | 0.0             | 0.0–0.10            |
| `epsilon` | **1e-6**       | 0.01            | 1e-3                |
| `iterations` | **30**      | 10              | 6                   |

> **Nota crítica:** El motor usaba `sigma=1.0` como default, pero el default
> real de la librería es `sigma=6.0`. Para el contexto de trading donde los
> alpha son pequeños, `sigma=1.0` es correcto conceptualmente (prior más
> informativo), pero debe ser explícito.

---

## Patrón de uso correcto completo (basado en logistic.py oficial)

```python
from trueskillthroughtime import History, Player, Gaussian

# 1. Preparar datos — patrón exacto de Landfried
composition = [[["estrategia_A"], [str(i)]] for i in range(N)]
results     = [[1., 0.] if score_A > score_B else [0., 1.]
               for score_A, score_B in zip(scores_A, scores_B)]
times       = [fold_start.timestamp() / 86_400 for fold_start in fold_starts]
priors      = {}  # vacío = usar defaults de History

# 2. Instanciar — sin obs, sin geometric_mean
h = History(
    composition,
    results,
    times,
    priors,        # {} si no hay priors personalizados
    mu    = 0.0,
    sigma = 1.0,   # explícito, no el default 6.0
    gamma = 0.03,
    p_draw = 0.0,
)

# 3. Convergencia — desempaquetar tupla correctamente
(step_mu, step_sigma), n_iter = h.convergence(
    epsilon    = 1e-3,
    iterations = 6,
    verbose    = False,
)
step = max(step_mu, step_sigma)

# 4. Extraer curvas — patrón exacto de logistic.py
lc = h.learning_curves()
mu_curve    = [tp[1].mu    for tp in lc["estrategia_A"]]
sigma_curve = [tp[1].sigma for tp in lc["estrategia_A"]]

# 5. Evaluar ajuste del modelo — log_evidence, NO geometric_mean
log_ev = h.log_evidence()
```
