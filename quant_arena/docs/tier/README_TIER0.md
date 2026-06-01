# Proyecto Paraguay — Tier 0: Fricciones y Validación Estadística Robusta

> **Ubicación de los módulos (tras la consolidación en `quant_arena/`):**
> `TransactionCostModel` → `quant_arena/backtesting/transaction_costs.py` ·
> `StatisticalSignificanceValidator` → `quant_arena/metricas/statistical_validation.py`.
> Import de paquete, p.ej. `from quant_arena.metricas import StatisticalSignificanceValidator`.
> Los ejemplos con imports planos (`from transaction_costs import ...`) corresponden
> al desarrollo original, hoy congelado en `legacy/TTT/`.

**Bitácora y pacto técnico** de los cambios del Tier 0 sobre el motor de
backtesting. Objetivo: eliminar el sesgo de *backtest sin fricciones* y añadir
puertas de significancia que distingan **alpha real de suerte/sobreajuste**.

> Paradigma: POO estricto, tipado, vectorizado. Cada responsabilidad vive en su
> propia clase, inyectada por constructor en el motor (Dependency Injection).

---

## 1. Resumen de cambios

| # | Cambio | Archivo | Estado |
|---|---|---|---|
| 1 | `TransactionCostModel` (comisión + slippage por turnover) | `transaction_costs.py` *(nuevo)* | ✅ |
| 2 | `StatisticalSignificanceValidator` (PSR, **DSR**, PBO) | `statistical_validation.py` *(nuevo)* | ✅ |
| 3 | `BacktestEngine` consume costos → retornos **NETOS** al Juez | `simulation_engine.py` | ✅ |
| 4 | `TTTSimulator` cablea costos por defecto (`apply_costs=True`) | `simulation_engine.py` | ✅ |
| 5 | `BacktestResult` extendido (`gross_returns`, `costs`, `turnover`) | `simulation_engine.py` | ✅ |

**Pendiente del Tier 0 (no cubierto aquí):** las 3 estrategias degeneradas
(`TrendFollow`, `MomXover`, `VolBreakout`) ya están **blindadas** en el Juez vía
`np.nan_to_num` (parche previo), pero la *causa raíz* de su señal plana es un bug
de generación de señal del adapter — se aborda por separado.

---

## 2. Arquitectura POO

```
                 ┌──────────────────────────────┐
                 │        TTTSimulator           │
                 │  apply_costs / apply_overlay  │
                 └───────────────┬──────────────┘
                                 │ inyecta (DI)
                 ┌───────────────▼──────────────┐
                 │        BacktestEngine         │
                 │  .risk_overlay  .cost_model   │
                 │        run_backtest()         │
                 └───────┬───────────────┬───────┘
          weight (post   │               │  gross_returns, weight
          overlay)       ▼               ▼
              ┌────────────────┐  ┌──────────────────────┐
              │  RiskOverlay   │  │ TransactionCostModel  │
              │ (Target Vol +  │  │  apply() → CostReport │
              │  Stop global)  │  │  (net_returns, costs) │
              └────────────────┘  └──────────────────────┘

   Análisis OOS (offline, sobre el track stitcheado):
              ┌───────────────────────────────────────┐
              │   StatisticalSignificanceValidator     │
              │   PSR · DSR · PBO · SignificanceReport  │
              └───────────────────────────────────────┘
```

Principios aplicados:
- **Single Responsibility:** costos, riesgo y validación son clases independientes.
- **Dependency Injection:** `BacktestEngine` recibe `cost_model` y `risk_overlay`
  por constructor; `None` = comportamiento legacy (sin fricciones / desnudo).
- **Inmutabilidad de resultados:** `CostReport` y `SignificanceReport` son
  `@dataclass(frozen=True)`.
- **Sin dependencias frágiles:** la estadística usa `statistics.NormalDist` (stdlib).

---

## 3. `TransactionCostModel` — `transaction_costs.py`

Modelo de fricciones lineal en *turnover*, 100% vectorizado.

### Interfaz

| Miembro | Firma | Descripción |
|---|---|---|
| `__init__` | `(commission_bps=0.5, slippage_bps=1.0, name="linear_bps")` | Costos one-way en bps. |
| `cost_per_turnover` | `-> float` (property) | `(commission+slippage)/1e4`. |
| `compute_turnover` | `(weight: Series) -> Series` | `|Δ exposición|`, vectorizado; día 0 = `|w₀|`. |
| `compute_costs` | `(weight: Series) -> Series` | `turnover · cost_per_turnover`. |
| `apply` | `(gross_returns, weight) -> CostReport` | Devuelve retornos netos + diagnósticos. |

`CostReport` *(frozen)*: `gross_returns`, `net_returns`, `costs`, `turnover`, y
las propiedades `total_cost`, `avg_turnover`, `annualized_cost_drag`.

### Modelo matemático (vectorizado, sin bucles)

```
turnover(t) = |w(t) − w(t−1)|                       # w = exposición efectiva post-overlay
cost(t)     = turnover(t) · (commission_bps + slippage_bps) / 1e4
net_ret(t)  = gross_ret(t) − cost(t)
```

`w(t)` es la exposición fraccional **durante** el día *t* (= `positions.shift(1)/capital`
ya escalada por el Risk Overlay), por lo que el turnover captura tanto los cambios
de señal como el redimensionamiento del overlay. Default = **1.5 bps** por unidad
de turnover one-way (razonable para ETFs líquidos como SPY).

---

## 4. `StatisticalSignificanceValidator` — `statistical_validation.py`

### Interfaz

| Miembro | Firma | Descripción |
|---|---|---|
| `sharpe_periodic` | `(returns) -> float` *(static)* | Sharpe por periodo (sin anualizar). |
| `sharpe_annual` | `(returns) -> float` | `SR · √252`. |
| `probabilistic_sharpe_ratio` | `(returns, sr_benchmark=0.0) -> float` | **PSR**. |
| `expected_max_sharpe` | `(sr_variance, n_trials) -> float` *(static)* | Umbral **SR₀**. |
| `deflated_sharpe_ratio` | `(returns, sr_variance, n_trials) -> float` | **DSR** = PSR(SR₀). |
| `evaluate` | `(returns, sr_variance, n_trials) -> SignificanceReport` | Bundle completo. |
| `evaluate_from_trials` | `(trials: DataFrame, target: str) -> SignificanceReport` | Deriva `V` y `N` de la matriz de ensayos. |
| `probability_of_backtest_overfitting` | `(trials, n_partitions=16) -> float` | **PBO** vía CSCV. |

### 4.1 Fundamento matemático del Deflated Sharpe Ratio (DSR)

El Sharpe muestral `ŜR` sobreestima la habilidad por tres razones; el DSR las
corrige todas:

**(a) No-normalidad y longitud finita — Probabilistic Sharpe Ratio (PSR):**

```
              ⎡        (ŜR − SR*) · √(n − 1)            ⎤
PSR(SR*) = Φ  ⎢ ─────────────────────────────────────── ⎥
              ⎣  √( 1 − γ₃·ŜR + ((γ₄ − 1)/4)·ŜR² )       ⎦
```

- `ŜR` = Sharpe muestral **periódico** (media/desv. estándar de los retornos).
- `γ₃` = asimetría (skew) de los retornos.
- `γ₄` = curtosis **no-excesiva** (= `pandas.kurt()` + 3; vale 3 para la normal).
- `n` = número de observaciones; `Φ` = CDF normal estándar.
- Retornos con cola izquierda (γ₃<0) o leptocúrticos (γ₄ alto) **reducen** el PSR:
  el mismo Sharpe es menos creíble bajo no-normalidad.

**(b) Múltiples ensayos — umbral de deflación SR₀:**

Probar `N` configuraciones y quedarse con la mejor infla el máximo esperado bajo
la hipótesis nula (todas con habilidad cero). Ese máximo esperado es:

```
SR₀ = √V · [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]
```

- `V` = **varianza de los Sharpe periódicos** de los `N` ensayos.
- `γ` = constante de Euler-Mascheroni (≈ 0.5772).
- `Z⁻¹` = CDF normal inversa (`NormalDist.inv_cdf`); `e` = constante de Euler.
- Más ensayos (`N↑`) o más dispersión entre ellos (`V↑`) ⇒ `SR₀↑` ⇒ se exige más.

**(c) Deflated Sharpe Ratio:**

```
DSR = PSR(SR₀)
```

Probabilidad de que el Sharpe observado supere el **máximo esperado por puro azar**
dado el número de ensayos. **Regla:** `DSR ≥ 0.95` ⇒ significativo al 95%.

`SignificanceReport` *(frozen)* expone: `n_obs`, `sharpe_periodic`,
`sharpe_annual`, `skew`, `kurtosis`, `psr_zero`, `sr0_deflation`, `dsr`,
`n_trials` y la propiedad `is_significant`.

### 4.2 Probability of Backtest Overfitting (PBO) — CSCV

Combinatorially Symmetric Cross-Validation (Bailey et al., 2017): se parte el
track en `S` bloques; para cada combinación de `S/2` como *in-sample* se elige la
estrategia óptima IS y se mide su **rango relativo** ω en el *out-of-sample*
complementario. Con el logit `λ = ln(ω/(1−ω))`:

```
PBO = P(λ ≤ 0) = fracción de combinaciones donde la mejor IS cae bajo la mediana OOS
```

`PBO > 0.5` ⇒ el desempeño IS no se sostiene OOS (sobreajuste). El recorrido de
combinaciones es algorítmicamente necesario (no es iteración fila-a-fila); el
Sharpe por bloque se computa vectorizado con `DataFrame.apply`.

---

## 5. Integración con el motor existente

### `BacktestResult` (extendido)
Nuevos campos: `gross_returns` (antes de costos), `costs`, `turnover`.
**`returns` ahora es NETO de costos** → el Juez TTT (`score_metric="sharpe"`)
puntúa sobre retornos netos.

### `BacktestEngine.__init__`
```python
BacktestEngine(benchmark_symbol="SPY", simulator=None,
               risk_overlay=None, cost_model=None)   # cost_model=None ⇒ sin fricciones
```
`run_backtest` aplica el `cost_model` justo después de calcular los retornos brutos
(`gross_ret = weight · price_ret`) y antes de la curva de equity.

### `TTTSimulator.__init__`
```python
TTTSimulator(..., apply_costs=True, cost_model=None)
# apply_costs=True y cost_model=None ⇒ instancia TransactionCostModel() por defecto
# apply_costs=False                  ⇒ backtest sin fricciones (legacy)
```
Verificado en log de ejecución:
```
TTTSimulator: 10 algoritmos | ... | metric=sharpe | RiskOverlay=TargetVol=15% | cap=1.0x |
TrailingStop GLOBAL@−15% | Costs=TransactionCostModel(commission_bps=0.5, slippage_bps=1.0, per_turnover=1.50bps)
```

---

## 6. Guía de uso

**Backtest con costos (por defecto):**
```python
from simulation_engine import TTTSimulator
sim = TTTSimulator(algorithms=zoo, score_metric="sharpe")   # costos + overlay ON
sim.run_tournament(market_data=df, n_folds=200)
```

**Costos personalizados / desactivados:**
```python
from transaction_costs import TransactionCostModel
sim = TTTSimulator(algorithms=zoo, cost_model=TransactionCostModel(commission_bps=1.0, slippage_bps=2.0))
sim = TTTSimulator(algorithms=zoo, apply_costs=False)        # sin fricciones (comparación)
```

**Puerta de significancia sobre el track OOS stitcheado:**
```python
from statistical_validation import StatisticalSignificanceValidator
val = StatisticalSignificanceValidator(periods_per_year=252)

# trials: DataFrame (filas=retornos OOS, columnas=las N estrategias ensayadas)
report = val.evaluate_from_trials(trials, target="LowVol_21d_tv10")
print(report.sharpe_annual, report.dsr, report.is_significant)

pbo = val.probability_of_backtest_overfitting(trials, n_partitions=16)
```

---

## 7. Validación realizada

- **Smoke test de clases:** `TransactionCostModel` (turnover/costo vectorizados),
  `PSR`, `DSR` (con `SR₀` por N ensayos) y `PBO` (CSCV) computan sin error sobre
  series sintéticas.
- **Motor end-to-end:** torneo de 10 algoritmos corre con `Costs=...` activo;
  los retornos netos alimentan al Juez Sharpe.

### 7.1 Auditoría OOS NETA + puerta DSR/PBO (ejecutada — `_oos_audit_tier0.py`)

Track OOS **2008-04-25 → 2024-12-31** (4.200 días), neto de **1.5 bps/turnover**,
overlay global, 7 ensayos activos (3 degeneradas excluidas). `V = 1.03e-3`,
umbral de deflación **SR₀ ≈ 0.706 anualizado**:

| Estrategia | Sharpe (neto) | MaxDD | CAGR | PSR(0) | **DSR** | Signif. 95% |
|---|---|---|---|---|---|---|
| LowVol_21d_tv10 | **0.962** | −12.15% | 8.21% | 1.000 | **0.848** | no |
| BuyAndHold | 0.602 | −24.38% | 6.79% | 0.992 | 0.337 | no |
| XGBoostTrend_d5 | 0.307 | −33.06% | 2.37% | 0.894 | 0.052 | no |
| RSI_14_30_70 | 0.011 | −14.32% | −0.02% | 0.518 | 0.002 | no |
| MeanRevBB_20d | −0.048 | −3.57% | −0.05% | 0.421 | 0.001 | no |
| DonchianBreakout_20d | −0.346 | −31.23% | −1.91% | 0.078 | 0.000 | no |
| FadeExtremes_63d | −0.451 | −0.28% | −0.02% | 0.000 | 0.000 | no |

**`PBO (CSCV, S=16) = 0.012` → aceptable (sin sobreajuste de selección).**

**Conclusiones verificadas (honestas):**
- **Ninguna estrategia pasa la puerta DSR ≥ 0.95.** `LowVol` es la única candidata
  real (DSR 0.848, ~85% de confianza) pero **no alcanza el umbral institucional
  del 95%** tras deflactar por los 7 ensayos.
- **Los costos sí muerden:** `XGBoost` cae de Sharpe 0.417 (bruto) → **0.307 (neto)**
  por su mayor rotación; `LowVol` apenas baja (0.984 → 0.962) por su bajo turnover.
- **PBO bajo (0.012) + DSR bajo es coherente y honesto:** el proceso **no está
  sobreajustando la selección** (el ranking IS→OOS es estable), pero las
  estrategias son genuinamente mediocres, no "afortunadas". No hay alpha oculto
  que el sobreajuste estuviera disfrazando.

---

## 8. Limitaciones honestas

- El modelo de costos es **lineal en turnover** (no modela impacto de mercado no
  lineal ni spread dependiente de la liquidez intradía). Suficiente para ETFs
  líquidos; insuficiente para small-caps o tamaños grandes.
- El DSR asume ensayos aproximadamente independientes; con estrategias muy
  correlacionadas, `N` efectivo < `N` nominal (el DSR es entonces *conservador*).
- PBO/CSCV requiere un track OOS suficientemente largo (`len ≥ 2·n_partitions`).

---

*Referencias:* Bailey, D. & López de Prado, M. (2012) *The Sharpe Ratio Efficient
Frontier*, J. of Risk; (2014) *The Deflated Sharpe Ratio*, J. of Portfolio Mgmt;
Bailey, Borwein, López de Prado & Zhu (2017) *The Probability of Backtest
Overfitting*, J. of Computational Finance.
