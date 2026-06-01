#  Replication of the results of the manuscript "TrueSkill Through Time: reliable initial skill estimates and historical comparability with Julia, Python and R"

Author: Gustavo Landfried
Author: Esteban Mocskos

1. Setup
    1. Julia Setup
    2. Python Setup
    3. R Setup
2. make Examples
3. make Figures
4. make Table1

---------------------------------------

## 1. Setup

Before running the code, install the packages used by each programming languages.
We specify the details of each one in the next subsections.

### 1.1 Julia SETUP

In `Julia` (at least 1.5, tested on 1.10.3) we use the packages `TrueSkillThroughTime`, `CSV`, `Dates`, `DataFrames`.
To install them, open the Julia console and execute the following code: 

```
using Pkg
Pkg.add("TrueSkillThroughTime")
Pkg.add("CSV")
Pkg.add("Dates")
Pkg.add("DataFrames")
```

### 1.2 Python SETUP

In Python3 (tested on 3.10.4, 3.10.12), we use the packages `trueskillthroughtime`, `pandas`, `numpy`, `python-dateutil`.
To install them, in a terminal execute the following code (some adjustments may be necessary depending on the operating system):

```
pip install trueskillthroughtime
pip install pandas
pip install numpy
pip install python-dateutil
```

### 1.3 R SETUP

In R (tested on 4.1.2), we use the packages `TrueSkillThroughTime`, `microbenchmark` and `hash`.}
To install them, open the R console and execute the following code:

```
install.packages("TrueSkillThroughTime")
install.packages("microbenchmark")
install.packages("hash")
```

---------------------------------------
## 2. make Examples

Executing `make Examples` in the root will execute the examples shown in the illustrations section.

- `TrueSkillThroughTime.jl/example.jl`
- `TrueSkillThroughTime.py/example.py`
- `TrueSkillThroughTime.R/example.R`

Also, the execution time reported in section Computational Details will be stored at:

- `./output/runtimes.csv`

Each column of this files summarizes one run time for each table (tables 2a, 2b, 3, 4 and 5).

---------------------------------------
## 3. make Figures

Executing `make Figures` in the root will trigger the skill estimates performed in Illustrations section: skill evolution, ATP and ATP with ground.

- `TrueSkillThroughTime.py/logistic.py`
- `TrueSkillThroughTime.jl/atp.jl`
- `TrueSkillThroughTime.jl/atp_ground.jl`

The results will be stored in the folder:

- `./output/logistic.csv`
- `./output/logistics_mu.csv`
- `./output/atp_learning_curves.csv`
- `./output/atp_ground_learning_curves.csv`

And this process will also create the Figures 5, 6 and 7.

- `./Code/logistic.pdf`
- `./Code/atp.pdf`
- `./Code/atp_ground.pdf`

---------------------------------------
## 4. make Table1

**This procedure takes several hours. We precomputed this file to save this time, if you still want to re-generate it, rename or remove table1.csv**

Executing `make Table1` in the root will trigger the generation of Table 1.

- `TrueSkillThroughTime.jl/table1.jl`

The results will be stored in the folder:

- `./output/table1.csv`

The meaning of the columns are:

- `rate_train`: the geometric mean of the prior prediction of the training set.
- `N_TRAIN`: the size of the training set.
- `rate_test`: the geometric mean of the prior prediction of the testing set.
- `N_TEST`: the size of the testing set.
- `sigma`: the optimal uncertainty of the prior skill distributions.
- `gamma`: the optimal dynamic uncertainty.
- `p_draw`: the optimal draw probability.

---------------------------------------
## 5. Proyecto Paraguay V2 — Arquitectura de Frecuencia Diaria (1D)

> **FRECUENCIA: DIARIA (1D) — EXCLUSIVAMENTE.** Una fila = un día de mercado.
> Se descartó por completo la vía intradía de 5 minutos: es inviable adquirir
> datos 5-min reales y está **PROHIBIDO sintetizarlos o interpolarlos** (destruye
> la validez del backtest institucional e induce ilusión estadística). No existe
> ninguna conversión "días → barras"; todo el sistema razona en días naturales.

El sobreajuste de los modelos ML (XGBoost, etc.) se agrava cuando el dataset solo
contiene OHLCV crudo. Para mitigarlo, los datos **diarios** del S&P 500 (SPY /
`^GSPC`, 1997→hoy, descargados de yfinance) se enriquecen con una batería de
indicadores técnicos antes de entrar al motor de validación CPCV/walk-forward.

### 5.1 Componentes (POO)

| Componente | Archivo | Responsabilidad |
|---|---|---|
| `FeatureEngineer` | `feature_engineer.py` | Calcula ~32 indicadores **diarios 100% vectorizados** (sin bucles `for` sobre filas). Sin look-ahead bias: todo es `rolling`/`ewm`/`shift` *trailing*. |
| `FeatureConfig` | `feature_engineer.py` | Dataclass con las ventanas **en días** y `trading_days=252` para anualizar. |
| `DailyFeaturePipeline` | `build_features.py` | Descarga SPY/^GSPC **diario** (yfinance) → `FeatureEngineer` → recorta warm-up (200 días) → guarda `sp500_daily_1997_to_today_features.parquet`. |
| `DataManager` | `data_manager.py` | Motor diario (`trading_days_per_year=252`). Expone los features vía `add_technical_features=True` en `load_data()` y `engineer_features(df)`. Splits de purga/embargo medidos en **días**. |

Generación del dataset diario enriquecido:

```
python build_features.py                 # SPY diario 1997→hoy
python build_features.py ^GSPC           # índice S&P 500 contado
python build_features.py SPY 2005-01-01  # ticker + fecha inicio
```

Consumo desde el motor de backtesting:

```python
dm = DataManager(trading_days_per_year=252, add_technical_features=True)
df = dm.load_data("SPY", "1997-01-01", "2026-01-01", source="yahoo")
# df incluye OHLCV diario + log_return + los indicadores técnicos
```

### 5.2 Diccionario de datos (features generados — ventanas en DÍAS)

Todas las ventanas se expresan en **días de mercado**. Los retornos rezagados
usan `shift(k>0)` → exclusivamente pasado (sin fuga).

| Feature | Familia | Definición (diaria) |
|---|---|---|
| `log_return` | Retornos | Retorno log de cierre: `ln(Cₜ / Cₜ₋₁)` |
| `ret_lag_1..10` | Retornos | `log_return` rezagado 1, 2, 3, 5, 10 días (sin fuga) |
| `rsi` | Momentum | Relative Strength Index de Wilder (14 días), vía EWM |
| `roc` | Momentum | Rate of Change del cierre (10 días): `Cₜ/Cₜ₋₁₀ − 1` |
| `momentum` | Momentum | `Cₜ − Cₜ₋₁₀` (10 días) |
| `macd` | Momentum | EMA(12d) − EMA(26d) del cierre |
| `macd_signal` | Momentum | EMA(9d) del MACD |
| `macd_hist` | Momentum | `macd − macd_signal` |
| `stoch_k` | Momentum | Oscilador estocástico %K (14 días) |
| `stoch_d` | Momentum | Media móvil (3 días) de %K |
| `sma_ratio_20/50/200` | Tendencia | Distancia relativa del cierre a su SMA (20/50/**200** días): `Cₜ/SMA − 1` |
| `ema_ratio_12/26` | Tendencia | Distancia relativa del cierre a su EMA (12/26 días): `Cₜ/EMA − 1` |
| `sma_slope` | Tendencia | Pendiente normalizada de la SMA(50 días): `pct_change` |
| `volatility` | Volatilidad | Desv. estándar rolling (20 días) de `log_return` |
| `realized_vol` | Volatilidad | Vol. realizada (21 días) **anualizada con √252** |
| `atr` | Volatilidad | Average True Range (14 días), True Range suavizado por EWM |
| `bb_width` | Volatilidad | Ancho normalizado de Bandas de Bollinger (20 días, 2σ) |
| `bb_pctb` | Volatilidad | %B: posición del cierre dentro de las bandas |
| `parkinson_vol` | Volatilidad | Vol. de Parkinson (20 días) basada en rango high/low |
| `volume_zscore` | Volumen | Z-score del volumen sobre ventana (20 días) |
| `volume_roc` | Volumen | Rate of Change del volumen (20 días) |
| `obv` | Volumen | On-Balance Volume: `Σ sign(ΔC)·V` (acumulado) |
| `vwap_dev` | Volumen | Desviación del cierre vs. VWAP rolling (20 días): `Cₜ/VWAP − 1` |

### 5.3 Manejo de NaN y look-ahead bias

- **Warm-up**: las primeras `max_lookback` (=**200 días**, por la SMA de 200) se
  recortan por posición con `trim_warmup()`. Es seguro: solo descarta el inicio,
  nunca días interiores.
- **Resultado verificado** (SPY diario, 1997-10-16 → hoy): **7.198 filas, 37
  columnas, 0 NaN residuales** tras el recorte de warm-up.
- **Sin relleno hacia atrás**: si apareciese un NaN interior (denominador nulo en
  un día plano), se deja como `NaN` — rellenarlo introduciría look-ahead bias.
  El motor CPCV/walk-forward (`DataManager`) lo gestiona por split.

### 5.4 Integración ML, consumo de features y Juez TTT por Sharpe

Tras la auditoría del CIO (Feature Engineering desconectado, Juez premiando beta
vía `hit_rate`), se cerró el circuito de extremo a extremo:

**(a) El pipeline consume los 32 features.** `run_ttt_analysis.py::_load_market_data`
ahora carga `sp500_daily_1997_to_today_features.parquet` como `market_data` (OHLCV
+ features), con fallback a yfinance crudo y, solo en último recurso, sintético.
Las ventanas IS/OOS heredan las columnas de features y se entregan a las estrategias.

**(b) Estrategia ML con re-fit Walk-Forward (anti-fuga).** Nueva
`XGBoostTrendStrategy` (`strategy_zoo.py`), décima del zoo:

| Estrategia | Tipo | Señal |
|---|---|---|
| BuyAndHold | passive | Largo permanente (benchmark) |
| TrendFollow_EMA12_26 | momentum | Cruce de EMAs + filtro ATR |
| MeanRevBB_20d | mean_reversion | Reversión a banda de Bollinger |
| RSI_14_30_70 | mean_reversion | Sobreventa/sobrecompra RSI |
| MomXover_20_50 | momentum | Cruce de medias 20/50 |
| VolBreakout_20d | volatility_breakout | Ruptura por pico de volatilidad |
| LowVol_21d_tv10 | low_volatility | Targeting de volatilidad |
| FadeExtremes_63d | contrarian | Fade de movimientos extremos (z) |
| DonchianBreakout_20d | breakout | Ruptura de canal Donchian |
| **XGBoostTrend_d5** | **ml_trend** | **Gradient boosting sobre los 32 features** |

Contrato anti-fuga de `XGBoostTrendStrategy`:
- `fit(is_data)` se invoca **una vez por fold** con la ventana IN-SAMPLE estricta
  (anterior al OOS, ya con purging+embargo). El modelo **solo ve `is_data`**.
- Target = dirección del retorno **forward** a `h=5` días; las últimas `h` filas
  del IS se descartan para que X e Y no miren fuera de la ventana.
- `generate_signals(oos_data)` predice con features contemporáneos; el backtest
  aplica `positions.shift(1)` → la posición de `t` se materializa en `t+1`.
- Hiperparámetros robustos: `max_depth=3, learning_rate=0.05, n_estimators=100,
  subsample=0.8`. Si el motor recibe OHLCV crudo (sin features), la estrategia
  queda **plana** (`side=0`) en vez de fallar.

**(c) El Juez TTT puntúa por Sharpe OOS, no `hit_rate`.** `TTTSimulator` acepta
`score_metric ∈ {hit_rate, skill_score, mse_diff, sharpe, sortino}`. El default del
orquestador pasó a **`sharpe`**: `_evaluate_algorithm` deriva el Sharpe anualizado
de los retornos OOS (`result.returns`, ya con `shift(1)`) vía `_fold_risk_score`.
Esto fuerza al algoritmo bayesiano a premiar **retorno ajustado por riesgo**, no
aciertos direccionales ni beta. (`sortino` disponible para penalizar solo la cola
inferior.)

**Flujo completo:**

```
build_features.py → sp500_daily_1997_to_today_features.parquet (OHLCV + 32 features)
        │
run_ttt_analysis.py (_load_market_data lee el parquet)
        │  por cada fold Walk-Forward:
        │     IS estricto ──fit()──► XGBoostTrend (re-entrena solo con IS)
        │     OOS        ──predict──► señales ──► Sharpe OOS ──► Juez TTT
        ▼
ttt_skill_summary.csv / ttt_dashboard.html  (ranking por habilidad riesgo-ajustada)
```

> Verificado en ejecución: 10 algoritmos, `metric=sharpe`, parquet enriquecido de
> 37 columnas cargado, XGBoost re-entrenado por fold (mayor reducción de σ del zoo).
> Estar conectado **no implica** ser rentable: en ventanas OOS cortas XGBoost aún no
> bate al benchmark — por eso se añade la capa de gestión de riesgo (§5.5).

### 5.5 Risk Overlay — Target Volatility + Trailing Stop (V2.1)

Las estrategias emitían exposición **"desnuda"** (100% invertidas en su señal
direccional), heredando el −55% de drawdown de la beta de mercado en crisis. La
clase `RiskOverlay` (`risk_overlay.py`) escala el vector de pesos **antes** de
computar los retornos, en el único cuello de botella señal→retorno
(`BacktestEngine.run_backtest`), de modo **uniforme para las 10 estrategias** y
afectando directamente al Sharpe que evalúa el Juez TTT.

**Mecanismo (vectorizado dentro del fold; ESTADO GLOBAL entre folds):**

| Capa | Fórmula | Default |
|---|---|---|
| Target Volatility | `risk_w = clip(target_vol / realized_vol_ann(t−1), 0, cap)` | `target_vol=15%`, **`cap=1.0x`** |
| Trailing Stop GLOBAL | `dd = equity_tv/hwm_global − 1`; `gate = (dd > −dd_limit).shift(1)` | `dd_limit=15%` |

- **⚠️ FIX V2.1 — Trailing Stop Global y Persistente.** Antes el `cummax()` se
  reiniciaba dentro de cada fold de 21 días (stop *fold-local*), permitiendo
  sangrados multi-mes (Donchian −65% pese al límite −15%). Ahora `RiskOverlay`
  mantiene **estado por estrategia** `_state[name] = (equity_tv, hwm_global,
  in_market)`: el equity vol-targeted se acumula a través de TODOS los folds y el
  drawdown se mide contra el **High-Water Mark global**, no contra el inicio del
  fold. El estado se actualiza al cierre de cada fold y se consume al inicio del
  siguiente; `RiskOverlay.reset()` (invocado por `run_tournament`) lo limpia entre
  torneos para no contaminar ejecuciones independientes.
- **Cap reducido a 1.0x** (sin apalancamiento; el 1.5x previo amplificaba
  estrategias malas). El overlay ahora SOLO reduce exposición.
- **Consistencia de datos:** usa el feature `realized_vol` (anualizado √252) del
  `FeatureEngineer` si está en `market_data`; si no, calcula σ₂₁d·√252 al vuelo.
- **Anti look-ahead:** vol con `.shift(1)`; gate del stop con `.shift(1)`; el
  estado heredado proviene solo de folds OOS anteriores (pasado), nunca del futuro.
- **Peso final aplicado:** `weight = positions.shift(1)/capital · risk_w_tv · stop_gate`.

**Activación (por defecto ON):**

```python
# TTTSimulator activa el overlay automáticamente:
sim = TTTSimulator(algorithms=..., score_metric="sharpe")          # overlay ON (15%/1.0x/−15% GLOBAL)
sim = TTTSimulator(algorithms=..., apply_risk_overlay=False)        # exposición desnuda
from risk_overlay import RiskOverlay
sim = TTTSimulator(algorithms=..., risk_overlay=RiskOverlay(target_vol=0.10, max_leverage=1.0))
```

> Verificado en ejecución: log `RiskOverlay=TargetVol=15% | cap=1.0x |
> TrailingStop GLOBAL@−15%`. El stop global **erradicó el sangrado multi-fold**
> (Donchian MaxDD −65.13% → −29.79%). Es una **segunda capa de portafolio** sobre
> el Meta-Labeling sizing interno de cada estrategia.

### 5.6 Auditoría OOS 2005-2024 (resultados verificados)

Backtest walk-forward de 200 folds (OOS contiguo **2008-04-25 → 2024-12-31**,
4.200 días), métrica Sharpe, **Risk Overlay V2.1 (Stop GLOBAL + cap 1.0x)**.
Métricas de **equity** reales (`_oos_metrics.py`, mismo `BacktestEngine`+`RiskOverlay`+`fit()`):

| Estrategia | Sharpe | Sortino | MaxDD | CAGR | DD 2008 | DD 2020 |
|---|---|---|---|---|---|---|
| **LowVol_21d_tv10** | **0.984** | 1.269 | **−12.08%** | 8.42% | −11.36% | −7.55% |
| BuyAndHold (overlay) | 0.618 | 0.752 | −24.05% | 7.01% | −16.94% | −19.56% |
| **XGBoostTrend_d5** | 0.417 | 0.482 | −28.70% | 3.39% | −9.05% | −8.26% |
| RSI_14_30_70 | 0.084 | 0.079 | −13.64% | 0.23% | −1.64% | −2.94% |
| MeanRevBB_20d | −0.019 | −0.003 | −3.55% | −0.02% | −0.10% | −0.85% |
| DonchianBreakout_20d | −0.319 | −0.242 | −29.79% | −1.78% | −5.49% | 0.00% |
| FadeExtremes_63d | −0.449 | −0.028 | −0.28% | −0.02% | 0.00% | −0.04% |
| _SPY B&H (desnudo)_ | _0.534_ | _0.647_ | _−52.39%_ | _9.05%_ | _−52.39%_ | _−34.10%_ |

> `TrendFollow_EMA12_26`, `MomXover_20_50`, `VolBreakout_20d` quedan **planas**
> (Sharpe=nan, 0 retornos) — artefacto degenerado del adapter. El Juez TTT está
> blindado con `np.nan_to_num` → un retorno todo-cero puntúa 0.0 (neutro), no NaN.

**Conclusiones verificadas (honestas):**
- **✅ El fix del Stop Global funcionó en su objetivo:** erradicó el sangrado
  multi-fold. `DonchianBreakout` cayó de **−65.13% → −29.79%** y SPY gestionado
  de −52% → −24% (DD 2008 de −29.76% → −16.94%). El `cummax()` ahora trabaja
  sobre el HWM global, no por fold.
- **⚠️ Pero el −15% NO es un techo duro.** `BuyAndHold` (−24%), `Donchian`
  (−30%) y `XGBoost` (−29%) aún lo cruzan: el stop reacciona con lag de
  `.shift(1)` y granularidad diaria, así que un gap-down de crisis sobrepasa el
  umbral antes de cortar. Un techo estricto exigiría stop intradía o `dd_limit`
  más ajustado. Solo `LowVol` (−12%) y `RSI` (−14%) quedan dentro.
- **⚠️ El overlay PERJUDICA a XGBoost:** con overlay Sharpe 0.417 / MaxDD −28.70%
  vs **sin** overlay 0.512 / −20.68%. El stop persistente lo saca cerca de mínimos
  y reentra tarde. Para XGBoost el overlay resta valor.
- **XGBoost sigue SIN alpha:** Sharpe OOS **0.417 < 0.8** objetivo, por debajo del
  SPY desnudo (0.534) y de `LowVol` (0.984).
- **El mejor desempeño riesgo-ajustado sigue siendo `LowVol` (smart-beta), no ML.**

### 5.7 Tier 0 — Fricciones y validación estadística

Implementado el **Tier 0** de la hoja de ruta: modelo de costos de transacción
(`TransactionCostModel`) que descuenta comisión+slippage por turnover de forma
vectorizada — el Juez TTT ahora puntúa el Sharpe sobre retornos **NETOS** — y un
validador de significancia (`StatisticalSignificanceValidator`) con PSR, **Deflated
Sharpe Ratio** y **PBO** (CSCV). Bitácora técnica completa, arquitectura POO y
fundamento matemático del DSR en **[`README_TIER0.md`](README_TIER0.md)**.

**Auditoría OOS neta + DSR/PBO (ejecutada):** ninguna estrategia pasa la puerta
DSR ≥ 0.95. `LowVol` es la única candidata (Sharpe neto 0.962, DSR 0.848 — ~85%,
por debajo del 95%). Costos: `XGBoost` cae de Sharpe 0.417 → 0.307 neto. **PBO =
0.012** (sin sobreajuste de selección) — el proceso es sano, pero **no hay alpha
estadísticamente significativo** en el zoo actual. Confirma la tesis: el salto
real exige el **Tier 1 (sección cruzada + datos ortogonales)**.

### 5.8 Tier 1 / 1.5 — Motor Cross-Sectional + Neutralización de Beta

Implementada la infraestructura de **panel** (`PanelDataManager`, MultiIndex
`[date,ticker]`, forward returns sin look-ahead, máscara anti-survivorship) y el
**motor long/short** (`CrossSectionalEngine`): ranking percentil por fecha,
dollar-neutral (Σw=0) y apalancamiento (Σ|w|=leverage), con el
`TransactionCostModel` inyectado para el turnover de panel `Σᵢ|Δwᵢ|`.

**Tier 1.5 — `BetaNeutralizer`** (`beta_neutralizer.py`): corrige el *beta leak*
(dollar-neutral ≠ market-neutral). Interfaz final:
- `compute_rolling_betas(returns, market_returns)` → betas OLS rodantes
  vectorizadas (`Cov_w/Var_w`, sin bucle por activo) con `.shift(lag=1)` estricto.
- `neutralize_weights(weights, betas, leverage=None)` → **proyección ortogonal
  analítica** (sin `scipy`, vectorizada sobre fechas): `w = w₀ − Cᵀ(CCᵀ)⁻¹C w₀`
  con `C=[𝟙;β]` restringida a la cesta activa e inversa 2×2 en forma cerrada;
  reescala a `Σ|w|=leverage`. Garantiza **Σw=0 ∧ Σwβ=0**.
- `residualize(...)` (εᵢ=Rᵢ−βᵢRₘ) y `portfolio_beta(...)` auxiliares.

`CrossSectionalEngine` lo recibe por **inyección de dependencias** (parámetro
`beta_neutralizer`) y delega la proyección (no la reimplementa); firmas de
`TransactionCostModel` y `StatisticalSignificanceValidator` intactas.

**Verificado:** caso sintético 3×2 (`w₀=[1,0.5,−1.5]`, `β=[1.2,1.0,0.8]` →
`[−0.5,1.0,−0.5]` con lev 2); universo real 30 large-caps + SPY (2005-2024):
`Σwβ`→5e-16 ex-ante, `Σ|w|`=2.0000, beta realizada OOS **−0.167 → −0.077**
(>50% del *leak* eliminado). Contrato de diseño, interfaces y matemática completa
en **[`README_TIER1.md`](README_TIER1.md)** (§9).

> El score residual-momentum usado es placeholder (Sharpe negativo): la
> neutralización aísla el residual, no crea alpha. Listo para scores reales del Tier 2.

---

## 6. Mapa de archivos del proyecto

| Archivo | Rol | Tier |
|---|---|---|
| `feature_engineer.py` | `FeatureEngineer` + `FeatureConfig` (32 features diarios) | base |
| `build_features.py` | `DailyFeaturePipeline` (SPY diario → parquet enriquecido) | base |
| `data_manager.py` | `DataManager` (carga, purging/embargo, CPCV) | base |
| `strategy_zoo.py` | `BaseStrategy` + 10 estrategias (incl. `XGBoostTrendStrategy`) | base |
| `simulation_engine.py` | `TTTSimulator` + `BacktestEngine` (Juez Sharpe, costos, overlay) | 0 |
| `risk_overlay.py` | `RiskOverlay` (Target Vol + Trailing Stop global) | — |
| `transaction_costs.py` | `TransactionCostModel` + `CostReport` | 0 |
| `statistical_validation.py` | `StatisticalSignificanceValidator` (PSR/DSR/PBO) | 0 |
| `panel_data_manager.py` | `PanelDataManager` (panel MultiIndex) | 1 |
| `cross_sectional_engine.py` | `CrossSectionalEngine` + `PortfolioResult` | 1 |
| `beta_neutralizer.py` | `BetaNeutralizer` (betas rodantes + proyección) | 1.5 |
| `README_TIER0.md` / `README_TIER1.md` | Bitácoras técnicas Tier 0 / Tier 1-1.5 | — |

