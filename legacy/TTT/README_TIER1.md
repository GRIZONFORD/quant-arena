# Proyecto Paraguay — Tier 1: Motor Cross-Sectional de Panel

**Contrato de diseño** del motor de portafolio long/short de sección cruzada.
Migra la investigación de un solo activo (SPY) a un **universo** (ETFs sectoriales)
manteniendo intactos los componentes Tier 0 (`TransactionCostModel`,
`StatisticalSignificanceValidator`) mediante **inyección de dependencias**.

---

## 1. Motivación

El Tier 0 demostró (DSR/PBO) que **no hay alpha significativo** cronometrando un
único activo con features colineales. El alpha de sección cruzada —rankear activos
*entre sí* y apostar a los extremos— es estructuralmente más explotable. El Tier 1
provee la infraestructura de **datos de panel** y **construcción de portafolio con
restricciones** para investigarlo con el mismo rigor de costos y significancia.

---

## 2. Arquitectura POO y flujo de dependencias

```
        PanelDataManager                         (datos)
   MultiIndex [date, ticker] OHLCV
        │  .wide(field) → Date×Ticker
        │  .simple_returns() / .forward_returns(h) / .realized_vol()
        ▼
   scores (Date×Ticker)   asset_returns (Date×Ticker)   vol (Date×Ticker)
        │                        │                          │
        └──────────────┬─────────┴──────────────────────────┘
                       ▼
              CrossSectionalEngine                 (portafolio)
        compute_weights() → ranking + restricciones
        run() → weights.shift(1)·returns − costos
                       │  inyecta (DI)
                       ▼
              TransactionCostModel  ──► apply_with_turnover(gross, turnover_panel)
                       │
                       ▼
                 PortfolioResult  (net_returns: Series agregada)
                       │
                       ▼
        StatisticalSignificanceValidator  ──► PSR · DSR · PBO
```

Principios: **Single Responsibility** (datos / portafolio / costos / validación
separados), **Dependency Injection** (el modelo de costos entra por constructor),
**resultados inmutables** (`PortfolioResult` es `@dataclass(frozen=True)`),
**vectorización total** (cero bucles sobre fechas; todo `rank(axis=1)`, `.unstack()`,
`.diff()`, `.sum(axis=1)`).

---

## 3. `PanelDataManager` — `panel_data_manager.py`

Contenedor de panel `[date, ticker]` con garantías de integridad temporal.

| Método | Firma | Descripción |
|---|---|---|
| `load` | `(start, end, source="yahoo") -> self` | Descarga y normaliza el panel del universo. |
| `from_panel` | `(panel: DataFrame) -> PanelDataManager` *(classmethod)* | Inyecta panel externo (universo deslistado-inclusivo). |
| `wide` | `(field="close") -> DataFrame` | Campo como matriz Date×Ticker (`.unstack`). |
| `availability_mask` | `() -> DataFrame[bool]` | True donde el activo cotiza (anti-survivorship). |
| `simple_returns` / `log_returns` | `(price_col="close") -> DataFrame` | Retornos contemporáneos por ticker. |
| `forward_returns` | `(horizon=1, price_col="close") -> DataFrame` | **TARGET**: `close(t+h)/close(t)−1` (`shift(-h)` por ticker). |
| `realized_vol` | `(window=21) -> DataFrame` | Vol anualizada por ticker (inverse-vol). |

**Anti-bias:**
- *Look-ahead*: `forward_returns` usa `shift(-h)` y es exclusivamente objetivo; los
  features/scores son contemporáneos.
- *Survivorship*: días sin cotización quedan `NaN` y se excluyen de la sección
  cruzada vía `availability_mask` (no se rellenan hacia atrás).
- ⚠️ Limitación: yfinance solo entrega supervivientes; para un estudio sin sesgo
  inyectar un panel deslistado-inclusivo con `from_panel()`.

---

## 4. `CrossSectionalEngine` — `cross_sectional_engine.py`

### Interfaz

| Método | Firma | Descripción |
|---|---|---|
| `__init__` | `(cost_model, long_pct=0.2, short_pct=0.2, leverage=2.0, dollar_neutral=True, weighting="equal")` | DI del modelo de costos + parámetros de portafolio. |
| `compute_weights` | `(scores, vol=None) -> DataFrame` | Ranking percentil → pesos con restricciones. |
| `run` | `(scores, asset_returns, vol=None) -> PortfolioResult` | Backtest de panel neto de costos. |

### 4.1 Construcción de pesos (vectorizada)

Ranking percentil **por fecha** (robusto a universo de tamaño variable y NaN):

```
ranks(t,i)   = pctrank_fila( scores(t,·) )           # rank(axis=1, pct=True)
long_mask    = ranks > 1 − long_pct                  # top q%
short_mask   = ranks ≤ short_pct                     # bottom q%
basis(t,i)   = 1            (equal-weight)
             | 1/vol(t,i)   (inverse-vol)
```

**Dollar-Neutral + apalancamiento** (`half = leverage/2`):

```
w_longᵢ  = +half · basisᵢ·long_maskᵢ  / Σⱼ(basisⱼ·long_maskⱼ)
w_shortᵢ = −half · basisᵢ·short_maskᵢ / Σⱼ(basisⱼ·short_maskⱼ)
wᵢ       = w_longᵢ + w_shortᵢ
```

Garantiza por construcción:  **Σᵢ wᵢ = 0**  (neutralidad de dólar)  y
**Σᵢ |wᵢ| = leverage**  (apalancamiento bruto). En modo `dollar_neutral=False`
opera long-only sobre el top cuantil escalado a `leverage`.

### 4.2 Turnover de panel y costos

El turnover del portafolio en cada rebalanceo es la suma de los cambios absolutos
de peso de **todos** los activos:

```
turnover(t) = Σᵢ |wᵢ(t) − wᵢ(t−1)|        # weights.diff().abs().sum(axis=1)
```

Se delega la conversión a costo al modelo inyectado:
`cost_model.apply_with_turnover(gross, turnover)` → `cost(t) = turnover(t)·(comm+slip)/1e4`.

### 4.3 Retornos (anti look-ahead)

```
r_p(t) = Σᵢ wᵢ(t−1) · rᵢ(t)          # weights.shift(1) · asset_returns
net(t) = r_p(t) − cost(t)
```

`scores(t)` (conocidos al cierre de t) fijan `weights(t)`, que se materializan en
`t+1` vía `shift(1)` → sin uso de información futura.

### 4.4 `PortfolioResult` (frozen)

`weights`, `gross_returns`, `net_returns`, `costs`, `turnover`; propiedades
`avg_turnover`, `avg_gross_leverage`, `avg_net_exposure`. `net_returns` es una
Serie agregada → **input directo** de `StatisticalSignificanceValidator`.

---

## 5. Integración con Tier 0

- **Costos:** `CrossSectionalEngine` recibe `TransactionCostModel` por constructor;
  se añadió a éste el método nativo `apply_with_turnover(gross, turnover)` para
  aceptar el turnover de panel ya agregado (la conversión bps→costo sigue siendo
  responsabilidad única del modelo de costos).
- **Validación:** `PortfolioResult.net_returns` se pasa tal cual a
  `StatisticalSignificanceValidator.evaluate(...)` / `probabilistic_sharpe_ratio(...)`.
  Para DSR multi-config, construir una matriz de ensayos (columnas = variantes del
  portafolio) y usar `evaluate_from_trials` / `probability_of_backtest_overfitting`.

---

## 6. Guía de uso

```python
from panel_data_manager import PanelDataManager
from cross_sectional_engine import CrossSectionalEngine
from transaction_costs import TransactionCostModel
from statistical_validation import StatisticalSignificanceValidator

pdm = PanelDataManager(["XLK","XLF","XLV","XLE","XLU","XLI","XLP","XLY","XLB"])
pdm.load("2005-01-01", "2024-12-31")

scores        = pdm.wide("close").pct_change(252)   # momentum 12-1 (placeholder)
asset_returns = pdm.simple_returns()
vol           = pdm.realized_vol(21)

engine = CrossSectionalEngine(
    cost_model=TransactionCostModel(commission_bps=0.5, slippage_bps=1.0),
    long_pct=0.2, short_pct=0.2, leverage=2.0, weighting="inverse_vol",
)
result = engine.run(scores, asset_returns, vol)

val = StatisticalSignificanceValidator(periods_per_year=252)
print(val.probabilistic_sharpe_ratio(result.net_returns))
```

---

## 7. Validación realizada (`_tier1_smoke.py`)

Universo: 9 ETFs sectoriales, 2005-2024. Score placeholder = momentum 252d.

| Config | Sharpe | MaxDD | Turnover | Σ\|w\| | Σw |
|---|---|---|---|---|---|
| L/S equal 20/20 lev2.0 | 0.061 | −60.75% | 0.249 | 1.90 | +0.000 |
| L/S inv-vol 20/20 lev2.0 | 0.049 | −61.10% | 0.279 | 1.90 | −0.000 |
| L/S equal 30/30 lev1.0 | −0.075 | −41.79% | 0.104 | 0.95 | −0.000 |

**Restricciones verificadas (fechas con universo pleno):** `Σw = 0` exacto,
`Σ|w| = 2.0000` exacto. ✅

> El score momentum es un **placeholder** para validar la tubería, no una señal de
> alpha. Su Sharpe ~0.06 y los `Σ|w|=1.90` en cuantiles parciales reflejan que 9
> activos con cuantil 20% (≈2 por pata) son demasiado concentrados. La arquitectura
> queda lista para (a) universos más amplios y (b) scores reales del Tier 2.

---

## 8. Limitaciones y siguiente paso

- Universo pequeño (9 sectores) ⇒ secciones cruzadas concentradas. Ampliar a
  sectores+factores o large-caps reduce el riesgo idiosincrático.
- Sin neutralización por sector/beta todavía (el dollar-neutral no implica
  market-neutral si las patas tienen betas dispares).
- **Siguiente (Tier 2):** alimentar `scores` con un modelo real (Triple-Barrier +
  Meta-Labeling, ranking ML cross-sectional) y pasar las variantes por la puerta
  DSR/PBO para certificar significancia neta de costos.

---

## 9. Tier 1.5 — Neutralización de Beta (`beta_neutralizer.py`)

Corrige el *beta leak*: el portafolio dollar-neutral (Σw=0) **no** es market-neutral
si las betas de las patas difieren. Se añade doble restricción **Σw=0 ∧ Σwβ=0**.

### 9.1 `BetaNeutralizer` — interfaz

| Miembro | Firma | Descripción |
|---|---|---|
| `__init__` | `(window=126, min_periods=60, lag=1, ridge=1e-12, det_tol=1e-12)` | Ventana OLS rodante; `lag=1` ⇒ sin look-ahead. |
| `compute_rolling_betas` | `(returns: DataFrame, market_returns: Series) -> DataFrame` | `βᵢ=Cov_w(Rᵢ,Rₘ)/Var_w(Rₘ)`, vectorizado vía momentos rodantes, `.shift(lag)`. |
| `neutralize_weights` | `(weights, betas, leverage=None) -> DataFrame` | Proyección ortogonal de doble restricción (ver §9.2). |
| `residualize` | `(returns, market_returns, betas=None) -> DataFrame` | `εᵢ(t)=Rᵢ(t)−βᵢ(t)·Rₘ(t)` (alpha residual para scores). |
| `portfolio_beta` | `(weights, betas) -> Series` *(static)* | `Σᵢ wᵢ(t)·βᵢ(t)` por fecha (≈0 si neutral). |

**Betas rodantes sin bucle por activo:** una sola pasada matricial sobre la matriz
wide `E_w[Rᵢ·Rₘ] − E_w[Rᵢ]·E_w[Rₘ]` dividida por `Var_w(Rₘ)`. Robusto a NaN vía
`min_periods`.

### 9.2 Proyección ortogonal de doble restricción (`neutralize_weights`)

`w = w₀ − Cᵀ(CCᵀ)⁻¹C w₀`, analítica y vectorizada sobre fechas — **no usa
`scipy.optimize`** (evita un loop por fecha):

```
C        = [m ; m·β]               (𝟙 restringido a la cesta activa m = |w₀|>0)
C w₀     = [Σw₀, Σw₀·β]            (vector 2×1 por fecha)
(CCᵀ)    = [[n, Σβ], [Σβ, Σβ²]]    (2×2 por fecha; n=Σm)
(CCᵀ)⁻¹  = (1/det)·[[Σβ², −Σβ], [−Σβ, n]],  det = n·Σβ² − (Σβ)²
λ        = (CCᵀ)⁻¹ (C w₀)
w        = w₀ − m·(λ₁ + λ₂·β)      → Σw=0 ∧ Σwβ=0
```

Restringir `C` a la cesta activa evita asignar peso a activos fuera del cuantil.
Tras proyectar se reescala a `Σ|w|=leverage` (`leverage=None` conserva el `Σ|w₀|`
original); el escalado escalar preserva ambas restricciones =0. Filas con
`|det|<det_tol` (degeneradas) → flat.

**Inyección en `CrossSectionalEngine`:** el constructor recibe
`beta_neutralizer: Optional[BetaNeutralizer]` (Dependency Injection) y
`compute_weights/run` aceptan `beta`. El engine **no reimplementa** la proyección
—delega en `beta_neutralizer.neutralize_weights(weights, beta, leverage=self.leverage)`—
y **no altera** las firmas de `TransactionCostModel` ni de
`StatisticalSignificanceValidator`.

```python
from beta_neutralizer import BetaNeutralizer
bn  = BetaNeutralizer(window=126, lag=1)
beta = bn.compute_rolling_betas(asset_returns, market_returns)   # SPY
eng = CrossSectionalEngine(cost_model=cm, weighting="inverse_vol", beta_neutralizer=bn)
res = eng.run(scores, asset_returns, vol=vol, beta=beta)         # Σw=0 ∧ Σwβ=0
```

**Verificado numéricamente** (caso sintético 3 activos × 2 fechas):
`w₀=[1,0.5,−1.5]`, `β=[1.2,1.0,0.8]` → proyección `[−0.75,1.5,−0.75]`
(Σw=0, Σwβ=0); reescalado a leverage 2 → `[−0.5,1.0,−0.5]`.

### 9.3 Validación (`_tier1_5_smoke.py`, 30 large-caps + SPY, 2005-2024)

| Motor | Sharpe | MaxDD | β_port (realizada) | Σwβ (ex-ante) | Σw | Σ\|w\| |
|---|---|---|---|---|---|---|
| Dollar-Neutral (Tier 1) | −0.540 | −88.2% | **−0.1669** | 0.226 | ~0 | 2.0 |
| Beta-Neutral (Tier 1.5) | −0.647 | −89.9% | **−0.0774** | **5e-16** | ~0 | 2.0 |

- **Restricción ex-ante exacta:** `Σwβ` cae a precisión de máquina (5e-16);
  `Σw`, `Σ|w|=2.0000` preservados.
- **Reducción real del leak:** la beta realizada OOS baja de −0.167 → −0.077
  (>50%). No llega a 0 exacto porque las betas rodantes rezagadas derivan ex-post
  — limitación honesta, no bug.
- El Sharpe negativo refleja que el score residual-momentum es **placeholder**;
  neutralizar beta aísla el residual pero no crea alpha donde no lo hay.

### 9.4 Limitaciones Tier 1.5
- Neutralidad **ex-ante**, no ex-post (drift de betas). Para neutralidad realizada
  más estricta: betas más reactivas o re-hedge intradía.
- Modelo de un factor (solo mercado). Multi-factor (size/value/momentum estilo
  Barra) requeriría proyectar sobre `span{𝟙, B}` con `B` la matriz de exposiciones
  N×K — la misma proyección generaliza con `(BᵀB)⁻¹` K×K.
