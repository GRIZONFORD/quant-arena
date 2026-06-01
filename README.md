# quant_arena

Framework de backtesting cuantitativo con ranking bayesiano para el índice S&P 500.
Combina un zoo de estrategias ML/DL con el algoritmo **TrueSkill Through Time (TTT)**
para asignar capital de forma adaptativa en cada período de rebalanceo.

---

## Stack tecnológico

| Componente | Versión mínima | Rol |
| :--- | :--- | :--- |
| Python | 3.12 | Runtime principal (confirmado por `cpython-312` en `__pycache__`) |
| NumPy | ≥ 1.26 | Álgebra vectorizada |
| Pandas | ≥ 2.2 | Series temporales y DataFrames |
| SciPy | ≥ 1.13 | Optimización L-BFGS-B en `calibracion/` |
| trueskillthroughtime | ≥ 0.1.0 | Motor de inferencia bayesiana del Juez TTT |
| pyarrow | ≥ 14.0 | Lectura del dataset `.parquet` |
| matplotlib / seaborn | ≥ 3.8 / 0.13 | Tear-sheet visual institucional |
| pytest | ≥ 9.0 | Suite de pruebas |

**Dependencias opcionales por capa:**

| Capa | Paquetes | Estrategias que las requieren |
| :--- | :--- | :--- |
| `[ml]` | xgboost, hmmlearn, arch, PyWavelets | `xgboost_trend`, `hmm_garch`, `wavelet_lstm` |
| `[dl]` | torch, gymnasium, stable-baselines3 | `tft_trend`, `mamba_ssm`, `stgnn_alpha`, `ppo_rl`, `neural_ff`, `wavelet_lstm` |
| `[nlp]` | transformers, torch | `llm_sentiment` |

---

## Arquitectura del sistema

```
quant_arena/
│
├── core/
│   └── abstracciones.py       Interfaces base: AbstractStrategy, AbstractJuez,
│                               AbstractMetricas, MetricasResultado (DTO)
│
├── zoo/
│   ├── base_estrategia.py     RegistroZoo (singleton), ZooManager, normalizar_pesos()
│   └── estrategias/           11 implementaciones concretas de AbstractStrategy
│       ├── ejemplo_momentum.py   momentum_126d  — baseline de referencia
│       ├── xgboost_strategy.py   xgboost_trend  — XGBoost Regressor [ml]
│       ├── hmm_garch_strategy.py hmm_garch      — HMM + GARCH(1,1) [ml]
│       ├── olps_rmr_strategy.py  olps_rmr       — Robust Median Reversion (sin ML)
│       ├── tft_strategy.py       tft_trend      — Temporal Fusion Transformer [dl]
│       ├── mamba_ssm_strategy.py mamba_ssm      — Mamba State Space Model [dl]
│       ├── stgnn_strategy.py     stgnn_alpha    — Spatio-Temporal GNN [dl]
│       ├── ppo_rl_strategy.py    ppo_rl         — PPO (RL) + Gymnasium [dl]
│       ├── neural_ff_strategy.py neural_ff      — Red neuronal feed-forward [dl]
│       ├── wavelet_lstm_strategy.py wavelet_lstm — DWT + LSTM [ml+dl]
│       └── llm_sentiment_strategy.py llm_sentiment — FinBERT / fallback técnico [nlp]
│
├── metricas/
│   ├── performance_metrics.py  Sharpe, Sortino, MDD, Calmar, IR, α-tstat, Turnover
│   └── filtros.py              KalmanSignalFilter — suavizado de métricas ruidosas
│
├── juez/
│   └── ttt_juez.py             TTTJuez — adaptador del paquete trueskillthroughtime
│                                Convierte KPIs → competencias → habilidades latentes
│
├── calibracion/
│   └── optimizador.py          OptimizadorTTT — ajusta sigma/gamma via L-BFGS-B
│
├── backtesting/
│   └── motor.py                BacktestEngine — loop walk-forward con garantía causal
│                                ResultadoBacktest — contenedor de resultados
│
├── resultados/
│   └── visualizador.py         ReporteCuantitativo — tear-sheet PDF institucional
│
├── data/
│   └── sp500_daily_1997_to_today.parquet   Dataset histórico OHLCV del S&P 500
│
├── scripts/
│   ├── run_backtest.py         ORQUESTADOR MAESTRO (ver sección de ejecución)
│   ├── run_mvp_calibracion.py  Smoke test del módulo de calibración TTT
│   └── fetch_sp500.py          Descarga/actualiza el dataset Parquet
│
└── tests/
    ├── conftest.py             Stub de trueskillthroughtime para CI sin la librería real
    ├── test_juez.py            22 tests del TTTJuez
    ├── test_metricas.py        Suite completa de PerformanceMetrics
    └── test_optimizador.py     Tests del OptimizadorTTT
```

### Flujo de datos en el loop walk-forward

```
                  ┌─────────────────────────────────────────────┐
                  │  BacktestEngine.ejecutar_walk_forward()      │
                  │                                              │
  datos.parquet ──► DataLoader ──► datos [fecha×Close]          │
                  │                     │                        │
                  │          ┌──────────▼──────────┐            │
                  │          │   ZooManager         │            │
                  │          │  .generar_señales()  │  pesos     │
                  │          │  (cada estrategia)   ├──────────► │
                  │          └──────────────────────┘           │
                  │                                              │
                  │  retornos ──► PerformanceMetrics             │
                  │               .calcular_rolling()            │
                  │                     │ métrica_bruta          │
                  │               KalmanSignalFilter             │
                  │                     │ métrica_filtrada       │
                  │               TTTJuez                        │
                  │               .registrar_periodo()           │
                  │               .actualizar()        EP (TTT)  │
                  │               .pesos_asignacion() ──────────►│
                  │                                    pesos_meta│
                  └─────────────────────────────────────────────┘
                                    │
                             ResultadoBacktest
                                    │
                            ResultsExporter
                            ├── CSV (6 archivos)
                            └── tearsheet.pdf
```

---

## Instalación

### Prerequisito
Python 3.12 instalado y accesible en el PATH.

### Paso 1 — Entorno virtual

```powershell
cd C:\Users\jsguz\Paraguay
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

### Paso 2 — Instalar el paquete en modo editable

```powershell
# Core + ML clásico + herramientas de desarrollo (P0 + P1)
pip install -e ".[ml,dev]"

# Verificar que el núcleo del sistema funciona
python quant_arena/scripts/run_mvp_calibracion.py
# Resultado esperado: "Estado : [OK] CONVERGENCIA EXITOSA"
```

### Paso 3 — Deep Learning (opcional, P2)

```powershell
# CPU build (recomendado para desarrollo)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dl]"

# GPU con CUDA 12.x (producción)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dl]"
```

### Paso 4 — NLP / LLM Sentiment (opcional, P2)

```powershell
pip install -e ".[nlp]"
```

### Paso 5 — Todo de una vez

```powershell
pip install -e ".[all,dev]"
```

---

## Ejecución

### Actualizar el dataset (si el parquet está desactualizado)

```powershell
pip install yfinance   # dependencia solo de este script utilitario
python quant_arena/scripts/fetch_sp500.py
```

### Backtest completo con defaults

```powershell
python quant_arena/scripts/run_backtest.py
```

Salida generada en `./resultados_backtest/`:
- `retornos_meta.csv` — serie diaria del meta-portafolio
- `retornos_estrategias.csv` — retornos por estrategia
- `pesos_juez.csv` — asignación TTT en cada rebalanceo
- `metricas_rolling.csv` — métricas brutas (ventana deslizante)
- `metricas_kalman.csv` — métricas filtradas por Kalman
- `resumen_estadistico.csv` — KPIs anualizados finales
- `tearsheet.pdf` — tear-sheet visual institucional

### Opciones de la CLI

```
--inicio    YYYY-MM-DD    Fecha de inicio del backtest     (default: 2005-01-01)
--fin       YYYY-MM-DD    Fecha de fin del backtest         (default: 2023-12-31)
--frecuencia INT          Días hábiles entre rebalanceos    (default: 21 ≈ mensual)
--ventana   INT           Ventana de métricas rolling       (default: 63 ≈ trimestral)
--metrica   STR           {sharpe,sortino,calmar,           (default: sharpe)
                           information_ratio,alpha_tstat}
--sigma     FLOAT         Prior TTT: incertidumbre inicial  (default: 1.6)
--gamma     FLOAT         TTT: drift temporal               (default: 0.036)
--calibrar                Ejecutar OptimizadorTTT antes del run
--estrategias NOMBRE...   Subconjunto de estrategias a incluir
--salida    PATH          Directorio de salida              (default: ./resultados_backtest)
--no-pdf                  No generar tear-sheet PDF
--no-csv                  No exportar CSVs
--mostrar                 Mostrar gráficos interactivos
```

### Ejemplos de uso avanzado

```powershell
# Solo estrategias sin dependencias de DL (rápido)
python quant_arena/scripts/run_backtest.py \
    --estrategias momentum_126d xgboost_trend hmm_garch olps_rmr \
    --inicio 2010-01-01 --fin 2022-12-31

# Ranking por Sortino en lugar de Sharpe
python quant_arena/scripts/run_backtest.py --metrica sortino

# Sin PDF, solo CSVs (útil en servidores sin display)
python quant_arena/scripts/run_backtest.py --no-pdf

# Calibrar hiperparámetros TTT antes del run
python quant_arena/scripts/run_backtest.py --calibrar --sigma 1.6 --gamma 0.036
```

### Suite de tests

```powershell
# Limpiar cache obsoleto y correr la suite completa
Remove-Item -Recurse -Force .pytest_cache
python -m pytest -v

# Solo un módulo
python -m pytest quant_arena/tests/test_metricas.py -v
```

---

## Uso programático (notebook / script)

```python
from quant_arena.scripts.run_backtest import PipelineConfig, SimulationPipeline

config = PipelineConfig(
    fecha_inicio="2015-01-01",
    fecha_fin="2023-12-31",
    frecuencia_rebalanceo=21,
    metrica_ranking="sharpe",
    estrategias_habilitadas=["momentum_126d", "xgboost_trend", "olps_rmr"],
    generar_pdf=True,
    guardar_csv=True,
    directorio_salida=Path("./mi_experimento"),
)

pipeline = SimulationPipeline(config)
resultado = pipeline.ejecutar()

# Acceder al objeto de resultados directamente
print(resultado.retornos_meta.describe())
print(resultado.pesos_juez.tail())

# Curvas de aprendizaje TTT
df_curvas = pipeline.juez.exportar_curvas_df()
snapshot   = pipeline.juez.snapshot_estado()
print(snapshot)
```

---

## Diagnóstico de dependencias

El sistema usa guards `try/except ImportError` en cada estrategia avanzada.
Si una dependencia falta, la estrategia se omite automáticamente durante el probe
y el backtest continúa con las estrategias operativas.

Para verificar qué estrategias están disponibles en tu entorno:

```python
from quant_arena.zoo.base_estrategia import RegistroZoo
RegistroZoo.autodescubrir()
print(RegistroZoo.listar_con_descripcion())
```

| Estrategia | Dependencia requerida | Instalación |
| :--- | :--- | :--- |
| `momentum_126d` | — (solo NumPy/Pandas) | incluida en core |
| `olps_rmr` | — (solo NumPy/Pandas) | incluida en core |
| `xgboost_trend` | xgboost | `pip install -e ".[ml]"` |
| `hmm_garch` | hmmlearn, arch | `pip install -e ".[ml]"` |
| `wavelet_lstm` | PyWavelets, torch | `pip install -e ".[ml,dl]"` |
| `tft_trend` | torch | `pip install -e ".[dl]"` |
| `mamba_ssm` | torch | `pip install -e ".[dl]"` |
| `stgnn_alpha` | torch | `pip install -e ".[dl]"` |
| `neural_ff` | torch | `pip install -e ".[dl]"` |
| `ppo_rl` | torch, gymnasium, stable-baselines3 | `pip install -e ".[dl]"` |
| `llm_sentiment` | transformers, torch | `pip install -e ".[nlp]"` |

---

## Historial de cambios de configuración

### 2026-05-30 — Remediación Gap Analysis (P0/P1)

**Problema:** El proyecto no podía instalarse ni ejecutarse por falta de
`pyproject.toml` y dependencias incompletas en `requirements.txt`.

**Solución aplicada:**

1. **`pyproject.toml` creado** en la raíz del proyecto.
   - Hace el paquete instalable con `pip install -e .`
   - Declara grupos de extras opcionales: `[ml]`, `[dl]`, `[nlp]`, `[dev]`, `[all]`
   - Configura `[tool.pytest.ini_options]` con `testpaths`

2. **`requirements.txt` actualizado** de 6 a 15 dependencias.
   - Añadidas: `pyarrow`, `xgboost`, `hmmlearn`, `arch`, `PyWavelets`,
     `torch`, `gymnasium`, `stable-baselines3`, `transformers`
   - Sección de Deep Learning marcada con comentario CPU/CUDA

3. **Cache de pytest limpiado.**
   `test_rotacion_total_da_dos` en `lastfailed` era un identificador obsoleto
   (el test fue renombrado a `test_rotacion_total_es_casi_dos`). No había bug.

4. **`run_backtest.py` creado** como orquestador maestro en arquitectura POO.

---

## Notas de producción

- **Causalidad estricta:** El `BacktestEngine` garantiza que en ningún paso
  del loop walk-forward se accede a datos con fecha posterior a `fecha_corte`.
  Esta garantía opera en tres niveles: filtrado de DataFrame, pesos generados en
  el período anterior, y registro en TTTJuez solo con métricas hasta `t_k`.

- **Dataset único:** El parquet contiene datos OHLCV del índice S&P 500 (`^GSPC`)
  como activo único. Las estrategias operan sobre la columna `Close`; el benchmark
  es la serie de retornos de la misma columna.

- **Modo degradado:** Si torch no está instalado, 6 de las 11 estrategias se
  omiten automáticamente. El sistema siempre arranca con al menos `momentum_126d`
  y `olps_rmr`, que solo requieren NumPy y Pandas.
