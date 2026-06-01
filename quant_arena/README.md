# quant_arena

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Version](https://img.shields.io/badge/version-1.0.0-0D3B8C)
![Tests](https://img.shields.io/badge/tests-54%2F54%20passed-2E7D32)
![License](https://img.shields.io/badge/license-MIT-757575)
![Status](https://img.shields.io/badge/status-research%20prototype-E65100)

---

## Abstract

Los modelos de asignación táctica de capital tradicionales asumen que la habilidad de una estrategia es estacionaria en el tiempo, lo cual es empíricamente refutable: los cambios de régimen macroeconómico, la reversión de factores y la saturación de señales producen perfiles de retorno no estacionarios que invalidan los supuestos de ventanas rodantes fijas. **quant_arena** plantea una solución basada en el principio de *arbitraje de habilidad latente*: en lugar de optimizar pesos sobre retornos pasados, el sistema infiere la distribución posterior de competencia de cada estrategia —modelada como un proceso de Wiener gaussiano— mediante el algoritmo de Expectation Propagation sobre grafos factoriales (TrueSkill Through Time, Landfried 2024). La señal de entrada al modelo bayesiano es previamente suavizada por un Filtro de Kalman escalar causal, eliminando el ruido de alta frecuencia inherente a las métricas de performance en ventanas cortas. El resultado es un meta-portafolio cuya asignación dinámica es proporcional al cociente señal/ruido $\mu / \sigma$ de las habilidades latentes, un análogo bayesiano del Sharpe Ratio que pondera simultáneamente la calidad de la señal y la incertidumbre del estimador.

---

## Arquitectura del Sistema

### Árbol de Directorios

```
quant_arena/
├── core/
│   └── abstracciones.py          # Interfaces ABC: AbstractStrategy, AbstractJuez, AbstractMetricas
├── metricas/
│   ├── performance_metrics.py    # PerformanceMetrics: Sharpe, Sortino, MDD, Calmar, IR, α t-stat
│   └── filtros.py                # KalmanSignalFilter: filtro escalar causal, NaN-safe
├── juez/
│   └── ttt_juez.py               # TTTJuez: adaptador TrueSkill Through Time (EP inference)
├── zoo/
│   ├── base_estrategia.py        # RegistroZoo, ZooManager, ConfigEstrategia
│   └── estrategias/
│       └── ejemplo_momentum.py   # MomentumStrategy: señal 126d, top-N, 3 esquemas de pesos
├── backtesting/
│   └── motor.py                  # BacktestEngine: walk-forward estrictamente causal
├── resultados/
│   └── visualizador.py           # ReporteCuantitativo: tear sheet PDF multi-página
└── tests/
    ├── conftest.py                # Stub de trueskillthroughtime para CI sin dependencias reales
    ├── test_metricas.py           # 31 tests: casos límite, invariantes y flujo normal
    └── test_juez.py               # 23 tests: normalización de pesos, causalidad, fallback
```

### Módulos y Responsabilidades

| Módulo | Clase Principal | Responsabilidad |
|---|---|---|
| `core` | `MetricasResultado`, `AbstractMetricas`, `AbstractJuez`, `AbstractStrategy` | Contratos de interfaz (DIP). Tipado estricto. |
| `metricas` | `PerformanceMetrics`, `KalmanSignalFilter` | Cálculo vectorizado de KPIs institucionales y suavizado de señal. |
| `juez` | `TTTJuez` | Inferencia EP sobre habilidades latentes. Genera pesos de asignación táctica. |
| `zoo` | `ZooManager`, `RegistroZoo`, `MomentumStrategy` | Registro y orquestación de sub-estrategias. Generación de señales con corte causal duro. |
| `backtesting` | `BacktestEngine`, `ResultadoBacktest` | Loop walk-forward con garantías de causalidad. Integra todos los módulos. |
| `resultados` | `ReporteCuantitativo` | Visualización institucional: equity curve, curvas TTT, asignación dinámica, tear sheet PDF. |

### Flujo de Datos Causal (DAG)

El sistema garantiza **ausencia de look-ahead bias** mediante un DAG estrictamente causal. En cada fecha de rebalanceo $t_k$:

```
datos[0 : t_{k-1}]
    └─→ ZooManager.generar_señales_todas()          ← señales generadas CON datos hasta t_{k-1}
            └─→ retornos_periodo(t_{k-1}, t_k]       ← aplicadas DURANTE (t_{k-1}, t_k]
                    └─→ PerformanceMetrics.calcular_rolling()   ← métricas acumuladas hasta t_k
                            └─→ KalmanSignalFilter.filtrar()     ← suavizado causal elemento a elemento
                                    └─→ TTTJuez.registrar_periodo(tiempo=t_k)
                                            └─→ TTTJuez.actualizar()       ← EP con datos hasta t_k
                                                    └─→ pesos_asignacion() ← usados en (t_k, t_{k+1}]
```

---

## Fundamentos Matemáticos

### 1. Filtro de Kalman Escalar (Suavizado de Métricas)

Las métricas de performance calculadas en ventanas cortas son observaciones ruidosas del estado latente de una estrategia. Se modela el proceso como:

$$x_t = x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

$$z_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

donde $x_t$ es la habilidad latente (estado oculto), $z_t$ es la métrica observada (e.g., Sharpe rolling), $Q$ es la varianza del proceso (velocidad de cambio de régimen) y $R$ es la varianza de la observación (ruido de estimación). Las ecuaciones de actualización son:

$$\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}, \quad P_{t|t-1} = P_{t-1|t-1} + Q$$

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}, \quad \hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (z_t - \hat{x}_{t|t-1})$$

En régimen estacionario, la ganancia de Kalman converge a:

$$K^* = \frac{u}{u + R}, \quad u = \frac{Q}{2} + \sqrt{\frac{Q^2}{4} + QR}$$

`KalmanSignalFilter` implementa el algoritmo secuencial exacto con soporte para valores faltantes (`NaN = predict-only step`), garantizando que el filtro sea estrictamente causal para su uso dentro del loop de backtesting.

### 2. TrueSkill Through Time — Inferencia sobre Habilidades Latentes

**quant_arena** adopta el modelo TTT de Landfried (2024) para estimar la evolución temporal de la competencia relativa entre estrategias. Cada estrategia $i$ tiene una habilidad latente que evoluciona como un proceso de Wiener:

$$s_i(t) = s_i(t-1) + \delta_i, \quad \delta_i \sim \mathcal{N}(0, \gamma^2)$$

En cada período de rebalanceo, el ranking de estrategias por KPI define un evento de competencia multi-equipo. La probabilidad de que la estrategia $i$ supere a la estrategia $j$ se modela mediante:

$$P(\text{strategy}_i > \text{strategy}_j) = \Phi\!\left(\frac{s_i - s_j}{\sqrt{2}\,\beta}\right)$$

donde $\Phi$ es la CDF normal estándar y $\beta$ controla la variabilidad del desempeño. La inferencia del posterior conjunto $p(s_1, \ldots, s_N \mid \text{historial})$ se realiza mediante **Expectation Propagation** sobre el grafo factorial completo:

$$p(\mathbf{s} \mid \mathcal{D}) \propto \prod_{t} \prod_{(i,j) \in \text{ranking}_t} \Phi\!\left(\frac{s_i(t) - s_j(t)}{\sqrt{2}\,\beta}\right) \cdot \prod_i \mathcal{N}(s_i(0); \mu_0, \sigma_0^2)$$

El posterior marginal de cada estrategia es aproximado como $s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$. Los pesos de asignación táctica se derivan del cociente señal/ruido:

$$w_i = \frac{\max(\mu_i / \sigma_i,\; 0)}{\sum_j \max(\mu_j / \sigma_j,\; 0)}$$

Este estimador es el análogo bayesiano del Sharpe Ratio: pondera la señal de habilidad $\mu_i$ e inversamente la incertidumbre $\sigma_i$, con fallback a pesos iguales $w_i = 1/N$ cuando todos los scores son no-positivos.

> **Referencia:** Landfried, G. (2024). *TrueSkill Through Time: Revisiting the History of Chess*. arXiv:2209.00092. Implementación: [`trueskillthroughtime`](https://github.com/glandfried/TrueSkillThroughTime.py).

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/quant_arena.git
cd quant_arena

# Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas scipy matplotlib seaborn trueskillthroughtime
```

### Dependencias

| Paquete | Versión mínima | Rol |
|---|---|---|
| `numpy` | 1.26 | Álgebra lineal vectorizada |
| `pandas` | 2.2 | Series temporales y DataFrames |
| `scipy` | 1.13 | Reservado para bootstrap (roadmap) |
| `matplotlib` | 3.8 | Visualización base |
| `seaborn` | 0.13 | Estilos y heatmaps |
| `trueskillthroughtime` | 0.1.0 | Motor de inferencia EP (Landfried, 2024) |

---

## Quickstart

```python
import pandas as pd
import numpy as np

from quant_arena.metricas.performance_metrics import PerformanceMetrics
from quant_arena.metricas.filtros import KalmanSignalFilter
from quant_arena.juez.ttt_juez import TTTJuez
from quant_arena.zoo.base_estrategia import ZooManager
from quant_arena.zoo.estrategias.ejemplo_momentum import MomentumStrategy
from quant_arena.backtesting.motor import BacktestEngine
from quant_arena.resultados.visualizador import ReporteCuantitativo

# ------------------------------------------------------------------
# 1. Datos de precios (ejemplo sintético — reemplazar con S&P 500)
# ------------------------------------------------------------------
fechas = pd.bdate_range("2010-01-01", "2023-12-31")
rng = np.random.default_rng(42)
n_activos, n_dias = 50, len(fechas)
precios = pd.DataFrame(
    (1 + rng.normal(0.0004, 0.012, (n_dias, n_activos))).cumprod(axis=0),
    index=fechas,
    columns=[f"TICKER_{i:02d}" for i in range(n_activos)],
)
benchmark = pd.Series(
    (1 + rng.normal(0.0003, 0.010, n_dias)).cumprod(),
    index=fechas,
    name="SPX",
).pct_change().dropna()

# ------------------------------------------------------------------
# 2. Instanciar componentes
# ------------------------------------------------------------------
zoo = ZooManager()
zoo.agregar(MomentumStrategy(ventana=126, top_n=10, esquema_pesos="equal"))

metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.04)

juez = TTTJuez(
    sigma=1.6,      # prior de habilidad (calibrado en datos ATP, Landfried 2024)
    gamma=0.036,    # drift de Wiener inter-período
    p_draw=0.0,     # sin empates en rankings de KPI
)

kalman_config = {"Q": 0.01, "R": 0.1}  # proceso lento, observaciones ruidosas

# ------------------------------------------------------------------
# 3. Construir y ejecutar el motor walk-forward
# ------------------------------------------------------------------
engine = BacktestEngine(
    zoo=zoo,
    metricas=metricas,
    juez=juez,
    datos=precios,
    benchmark=benchmark,
    kalman_config=kalman_config,
    ventana_metricas=63,         # ventana rodante de 1 trimestre
    metrica_ranking="sharpe",
)

resultado = engine.ejecutar_walk_forward(
    inicio="2011-01-01",
    fin="2023-12-31",
    frecuencia="ME",             # rebalanceo mensual (Month End)
)

# ------------------------------------------------------------------
# 4. Visualizar y exportar tear sheet
# ------------------------------------------------------------------
reporte = ReporteCuantitativo(resultado, benchmark=benchmark, juez=juez)
reporte.generar_tearsheet("output/tearsheet_v1.pdf", log_scale=False)

# Resumen de KPIs finales
print(resultado.resumen_estadistico(metricas, benchmark))
```

### Salidas Esperadas

`generar_tearsheet()` produce un PDF con cuatro páginas:

| Página | Contenido |
|---|---|
| 1 | Equity curve del meta-portafolio vs. benchmark + panel de drawdown + caja de KPIs (CAGR, Vol, Sharpe, MDD) |
| 2 | Curvas de aprendizaje TTT: trayectoria $\mu_i(t) \pm \sigma_i(t)$ por estrategia |
| 3 | Área apilada de asignación dinámica de capital $w_i(t)$ |
| 4 | Señal bruta vs. Kalman-filtrada por estrategia ($K^*$ visible como suavizado) |

---

## Ejecutar los Tests

```bash
# Desde el directorio raíz del proyecto
python -m pytest quant_arena/tests/ -v

# Con reporte de cobertura (requiere pytest-cov)
python -m pytest quant_arena/tests/ --cov=quant_arena --cov-report=term-missing
```

La suite es completamente autónoma: `conftest.py` inyecta un stub de `trueskillthroughtime` en `sys.modules` cuando el paquete no está instalado, permitiendo la ejecución en entornos de CI sin dependencias opcionales.

```
========================= 54 passed in 2.21s =========================
```

---

## Roadmap

### v1.1.0 — Optimización de Hiperparámetros TTT

La selección de $(\sigma, \gamma)$ es actualmente manual. El siguiente paso es explotar `TTTJuez.log_evidencia()` —que expone `History.log_evidence()` de Landfried— como función objetivo para búsqueda grid o Bayesiana:

$$(\hat{\sigma}, \hat{\gamma}) = \arg\max_{\sigma, \gamma} \log p(\mathcal{D} \mid \sigma, \gamma)$$

Esto permitirá calibración automática y comparación rigurosa de especificaciones del modelo bajo el mismo criterio marginal.

### v1.2.0 — Intervalos de Confianza Bootstrap

Los KPIs reportados en `resumen_estadistico()` son estimadores puntuales. La integración de bootstrap por bloques (*stationary bootstrap*, Politis & Romano 1994) en el módulo `resultados/` producirá intervalos de confianza para Sharpe, Sortino y Calmar, cuantificando la incertidumbre de estimación en función de la longitud de la muestra.

### v1.3.0 — Expansión del Zoo de Estrategias

Incorporación de cuatro familias de factores de renta variable adicionales al `RegistroZoo`:

| Estrategia | Señal | Referencia |
|---|---|---|
| `value_ep` | E/P ratio (earnings yield) | Fama & French (1992) |
| `low_vol` | Volatilidad realizada inversa (63d) | Ang et al. (2006) |
| `quality_roe` | Return on Equity normalizado | Novy-Marx (2013) |
| `carry_div` | Dividend yield trailing 12m | Koijen et al. (2018) |

La diversificación de factores en el Zoo permitirá que TTT arbitre regímenes donde el momentum revierte pero el value o el quality mantienen su habilidad, incrementando la robustez del meta-portafolio.

### v2.0.0 — Datos Reales S&P 500 (1997–2026)

Integración del universo histórico completo del S&P 500 en `data/`, incluyendo ajuste por dividendos, gestión de entradas/salidas del índice (survivorship bias), y alineación con el calendario NASDAQ. Esto permitirá validación empírica completa del sistema sobre múltiples ciclos macroeconómicos.

---

## Estructura del Módulo `core` — Interfaces y Contratos

El módulo `core/abstracciones.py` define el contrato arquitectónico mediante clases base abstractas (ABCs), garantizando el Principio de Inversión de Dependencias (DIP):

```python
from quant_arena.core.abstracciones import (
    MetricasResultado,   # Dataclass DTO con 7 KPIs: sharpe, sortino, max_drawdown,
                         #   calmar, information_ratio, alpha_tstat, turnover
    AbstractMetricas,    # Interfaz para calculadoras de KPIs (stateless)
    AbstractStrategy,    # Interfaz para sub-estrategias del Zoo
    AbstractJuez,        # Interfaz para el árbitro de habilidad latente
)
```

`BacktestEngine` depende exclusivamente de `AbstractJuez`, `AbstractMetricas` y `ZooManager`; nunca de `TTTJuez` o `PerformanceMetrics` directamente. Esto permite sustituir cualquier componente —e.g., reemplazar TTT por un modelo de Elo estándar— sin modificar el motor.

---

## Licencia

MIT License — ver [`LICENSE`](LICENSE) para los términos completos.

---

## Citación

Si este framework es utilizado en investigación académica, por favor citar:

```bibtex
@software{quant_arena_2026,
  title   = {quant\_arena: Dynamic Capital Allocation via Latent Skill Arbitrage},
  year    = {2026},
  version = {1.0.0},
  note    = {Framework cuantitativo para asignación táctica de capital usando
             TrueSkill Through Time (Landfried, 2024) y Filtro de Kalman causal.}
}
```

> Landfried, G. (2024). *TrueSkill Through Time: Revisiting the History of Chess*. Advances in Neural Information Processing Systems. [arXiv:2209.00092](https://arxiv.org/abs/2209.00092)
