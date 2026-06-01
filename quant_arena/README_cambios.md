# README — Cambios Recientes: Zoo Test Suites

**Fecha:** 2026-05-27  
**Versiones:** v0.2.1 → v0.2.9  
**Autor:** Claude Sonnet 4.6

---

## Resumen

Se completaron los **9 modelos restantes** del Zoo de quant_arena agregando suites de
tests unitarios a cada archivo de estrategia. El criterio de "IMPLEMENTADO" requiere:
(1) lógica completa de `generar_señales()` y `calcular_retornos()`, y
(2) un bloque `if __name__ == "__main__":` con 6 tests que verifican el contrato
de la interfaz `AbstractStrategy`.

Los 9 modelos ya tenian logica completa. Solo faltaban los tests.

---

## Estado del Zoo — COMPLETO

| # | Modelo | Archivo | Registro | Version |
|---|--------|---------|----------|---------|
| 1 | XGBoost + Optuna | `xgboost_optuna_strategy.py` | `'xgboost_optuna'` | pre-existente |
| 2 | OLPS-RMR | `olps_rmr_strategy.py` | `'olps_rmr'` | v0.2.1 |
| 3 | HMM+GARCH | `hmm_garch_strategy.py` | `'hmm_garch'` | v0.2.2 |
| 4 | Wavelet-LSTM | `wavelet_lstm_strategy.py` | `'wavelet_lstm'` | v0.2.3 |
| 5 | Neural-FF | `neural_ff_strategy.py` | `'neural_ff'` | v0.2.4 |
| 6 | TFT | `tft_strategy.py` | `'tft_trend'` | v0.2.5 |
| 7 | PPO-RL | `ppo_rl_strategy.py` | `'ppo_rl'` | v0.2.6 |
| 8 | Mamba-SSM | `mamba_ssm_strategy.py` | `'mamba_ssm'` | v0.2.7 |
| 9 | ST-GNN | `stgnn_strategy.py` | `'stgnn_alpha'` | v0.2.8 |
| 10 | LLM-Sentiment | `llm_sentiment_strategy.py` | `'llm_sentiment'` | v0.2.9 |

---

## Estructura de cada test suite

Cada archivo tiene un bloque `if __name__ == "__main__":` con exactamente 6 tests:

| Test | Descripcion | Proposito |
|------|-------------|-----------|
| T1 | Instanciacion y propiedades basicas | Verifica `nombre`, `universo`, `descripcion` |
| T2 | Warm-up: datos insuficientes | Confirma que retorna neutral (0.0) antes del umbral |
| T3 | Operacion normal | Datos suficientes; senal valida en {-1, 0, 1} |
| T4 | Regimen extremo (bull/crash) | Sin crash con mu/sigma extremos |
| T5 | Sin columna 'Close' | Datos malformados -> neutral, no excepcion |
| T6 | Boundary condition | Exactamente en el umbral de entrenamiento minimo |

### Invariantes verificados en cada test

1. **Causalidad**: `datos.index <= fecha_corte` siempre
2. **Valores validos**: senal in {-1.0, 0.0, 1.0}, sin NaN
3. **Sin crash**: datos malformados -> neutral, no excepcion
4. **Contrato de indice**: Series retornada tiene como indice `universo`

### Helpers estandar (identicos en todos los archivos)

```python
def _make_ohlcv(n, mu=3e-4, sigma=0.012, start="2010-01-01", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = 1_000.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    noise = rng.uniform(0.001, 0.008, n)
    return pd.DataFrame({
        "Close": close,
        "Open":  close * (1 + rng.normal(0, 0.002, n)),
        "High":  close * (1 + np.abs(rng.normal(0, noise))),
        "Low":   close * (1 - np.abs(rng.normal(0, noise))),
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=dates)

def _check(s, label, ticker, valid={-1.0, 0.0, 1.0}):
    assert isinstance(s, pd.Series),    f"[{label}] No es pd.Series"
    assert not s.isnull().any(),        f"[{label}] NaN en senal"
    bad = set(s.values) - valid
    assert not bad,                     f"[{label}] Valores fuera de {{-1,0,1}}: {bad}"
    assert ticker in s.index,           f"[{label}] Ticker no en indice"
```

---

## Como ejecutar los tests

Desde la raiz del proyecto (`C:\Users\jsguz\Paraguay`):

```bash
python quant_arena/zoo/estrategias/olps_rmr_strategy.py
python quant_arena/zoo/estrategias/hmm_garch_strategy.py
python quant_arena/zoo/estrategias/wavelet_lstm_strategy.py
python quant_arena/zoo/estrategias/neural_ff_strategy.py
python quant_arena/zoo/estrategias/tft_strategy.py
python quant_arena/zoo/estrategias/ppo_rl_strategy.py
python quant_arena/zoo/estrategias/mamba_ssm_strategy.py
python quant_arena/zoo/estrategias/stgnn_strategy.py
python quant_arena/zoo/estrategias/llm_sentiment_strategy.py
```

> **IMPORTANTE:** No usar `python -m quant_arena.zoo.estrategias.FILE`.
> El doble import via `-m` activa el decorador `@RegistroZoo.registrar` dos veces
> y lanza `ValueError: Ya existe una estrategia registrada con ese nombre`.

---

## Detalle por modelo

### v0.2.1 — OLPS-RMR (`olps_rmr_strategy.py`)

- **Clase:** `OLPSRMRStrategy` | **Registro:** `'olps_rmr'`
- **Dependencias test:** solo numpy/pandas
- **min_train_days:** 30 | **window_size:** 10
- **Umbral T2:** n=15 | **Normal T3/T4/T5:** n=100 | **Boundary T6:** n=42

### v0.2.2 — HMM+GARCH (`hmm_garch_strategy.py`)

- **Clase:** `HMMGARCHStrategy` | **Registro:** `'hmm_garch'`
- **Dependencias test:** hmmlearn, arch
- **min_train_days:** 252 | **n_regimenes:** 3
- **Umbral T2:** n=100 | **Normal T3/T4/T5:** n=600 | **Boundary T6:** n=252
- **Nota:** Las advertencias `Model is not converging` de hmmlearn son normales (no fallos).

### v0.2.3 — Wavelet-LSTM (`wavelet_lstm_strategy.py`)

- **Clase:** `WaveletLSTMStrategy` | **Registro:** `'wavelet_lstm'`
- **Dependencias test:** pywt, torch
- **min_train_days:** 252 | **lookback:** 42 | **max_epochs:** 3
- **Umbral efectivo:** ~494 dias (min_train + lookback + buffer)
- **Umbral T2:** n=300 | **Normal T3/T4/T5:** n=700 | **Boundary T6:** n=494

### v0.2.4 — Neural-FF (`neural_ff_strategy.py`)

- **Clase:** `NeuralFFStrategy` | **Registro:** `'neural_ff'`
- **Dependencias test:** torch
- **min_train_days:** 252 | **epochs:** 5
- **Umbral efectivo:** ~257 dias
- **Umbral T2:** n=150 | **Normal T3/T4/T5:** n=600 | **Boundary T6:** n=257

### v0.2.5 — TFT (`tft_strategy.py`)

- **Clase:** `TFTStrategy` | **Registro:** `'tft_trend'`
- **Dependencias test:** torch
- **min_train_days:** 252 | **lookback:** 42 | **max_epochs:** 3 | **d_model:** 16 | **n_heads:** 2
- **Umbral efectivo:** ~494 dias
- **Umbral T2:** n=300 | **Normal T3/T4/T5:** n=700 | **Boundary T6:** n=494

### v0.2.6 — PPO-RL (`ppo_rl_strategy.py`)

- **Clase:** `PPOStrategy` | **Registro:** `'ppo_rl'`
- **Dependencias test:** gymnasium, stable-baselines3
- **min_train_days:** 252 | **total_timesteps:** 500
- **Umbral efectivo:** ~452 dias
- **Umbral T2:** n=200 | **Normal T3/T4/T5:** n=600 | **Boundary T6:** n=452

### v0.2.7 — Mamba-SSM (`mamba_ssm_strategy.py`)

- **Clase:** `MambaStrategy` | **Registro:** `'mamba_ssm'`
- **Dependencias test:** torch
- **min_train_days:** 252 | **lookback:** 63 | **max_epochs:** 3 | **d_model:** 16 | **n_layers:** 1 | **d_state:** 8
- **Umbral efectivo:** ~515 dias
- **Umbral T2:** n=300 | **Normal T3/T4/T5:** n=700 | **Boundary T6:** n=515

### v0.2.8 — ST-GNN (`stgnn_strategy.py`)

- **Clase:** `STGNNStrategy` | **Registro:** `'stgnn_alpha'`
- **Dependencias test:** torch
- **min_train_days:** 252 | **lookback:** 42 | **max_epochs:** 3 | **d_gcn:** 8 | **d_gru:** 16 | **n_gru_layers:** 1
- **Umbral efectivo:** ~494 dias
- **Umbral T2:** n=300 | **Normal T3/T4/T5:** n=700 | **Boundary T6:** n=494
- **Bugfix incluido:** `STGNNCore.forward()` — `h.view()` reemplazado por `h.reshape()`
  para manejar tensores no contiguos sin RuntimeError.

### v0.2.9 — LLM-Sentiment (`llm_sentiment_strategy.py`)

- **Clase:** `LLMSentimentStrategy` | **Registro:** `'llm_sentiment'`
- **Dependencias test:** ninguna adicional (modo fallback tecnico, sin FinBERT)
- **min_train_days:** 63 | **text_col:** None (proxy tecnico, sin red)
- **Umbral T2:** n=30 | **Normal T3/T4/T5:** n=200 | **Boundary T6:** n=63

---

## Bugfix: ST-GNN (`stgnn_strategy.py`)

Durante la implementacion de los tests se detecto y corrigio un bug en la capa GRU:

**Problema:** `STGNNCore.forward()` usaba `h.view(h.size(1), -1)` para aplanar el
estado oculto del GRU. Cuando el tensor no es contiguo en memoria (e.g. despues de
`permute()` o `transpose()`), `view()` lanza `RuntimeError: non-contiguous tensor`.

**Solucion:** Reemplazar `h.view(...)` por `h.reshape(...)`. A diferencia de `view`,
`reshape` llama `.contiguous()` internamente cuando es necesario.

```python
# Antes (bug):
h = h.view(h.size(1), -1)

# Despues (fix):
h = h.reshape(h.size(1), -1)
```

---

## Archivos modificados

| Archivo | Tipo de cambio |
|---------|---------------|
| `zoo/estrategias/olps_rmr_strategy.py` | Test suite agregada |
| `zoo/estrategias/hmm_garch_strategy.py` | Test suite agregada |
| `zoo/estrategias/wavelet_lstm_strategy.py` | Test suite agregada |
| `zoo/estrategias/neural_ff_strategy.py` | Test suite agregada |
| `zoo/estrategias/tft_strategy.py` | Test suite agregada |
| `zoo/estrategias/ppo_rl_strategy.py` | Test suite agregada |
| `zoo/estrategias/mamba_ssm_strategy.py` | Test suite agregada |
| `zoo/estrategias/stgnn_strategy.py` | Test suite agregada + bugfix `h.reshape()` |
| `zoo/estrategias/llm_sentiment_strategy.py` | Test suite agregada |
| `changelog_codigo.md.txt` | Entradas v0.2.1 a v0.2.9 agregadas |
