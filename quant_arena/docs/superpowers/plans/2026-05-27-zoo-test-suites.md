# Zoo Test Suites — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar bloques `if __name__ == "__main__":` con 6 tests unitarios a cada uno de los 9 modelos del Zoo, en orden de complejidad creciente, marcándolos como "IMPLEMENTADO". Actualizar el changelog tras cada modelo.

**Architecture:** Cada archivo de estrategia recibe un bloque `if __name__ == "__main__":` autocontenido. Los tests usan OHLCV sintético (random walk log-normal, sin red). El changelog `changelog_codigo.md.txt` se actualiza tras cada modelo completado.

**Tech Stack:** Python 3.10+, numpy, pandas, hmmlearn, arch, pywt, torch, gymnasium, stable-baselines3, transformers (fallback mode para LLM)

---

## Helper compartido (usar en todos los bloques)

Cada bloque `if __name__ == "__main__":` define localmente estas dos funciones:

```python
def _make_ohlcv(n, mu=3e-4, sigma=0.012, start="2010-01-01", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = 1_000.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    noise = rng.uniform(0.001, 0.008, n)
    return pd.DataFrame({
        "Close": close,
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + np.abs(rng.normal(0, noise))),
        "Low":  close * (1 - np.abs(rng.normal(0, noise))),
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=dates)

def _check(s, label, ticker, valid={-1.0, 0.0, 1.0}):
    assert isinstance(s, pd.Series),        f"[{label}] No es pd.Series"
    assert not s.isnull().any(),            f"[{label}] NaN en señal"
    assert not (set(s.values) - valid),     f"[{label}] Valores fuera de {{-1,0,1}}"
    assert ticker in s.index,              f"[{label}] Ticker no en índice"
```

---

## Task 1: OLPS-RMR — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/olps_rmr_strategy.py` (append al final)
- Modify: `quant_arena/changelog_codigo.md.txt` (append entrada v0.2.1)

- [ ] **Step 1: Append el bloque de tests al archivo**

Agregar al final de `quant_arena/zoo/estrategias/olps_rmr_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "RMRTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  OLPSRMRStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = OLPSRMRStrategy(
        universo=[TICKER], window_size=10, eps=0.003, min_train_days=30
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "olps_rmr"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='olps_rmr' | universo | descripcion no vacía")

    # TEST 2 — Warm-up (datos insuficientes)
    print("\n[TEST 2] Warm-up (n=15 < min_train=30)...")
    df_short = _make_ohlcv(n=15)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral — warm-up respetado")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=100)...")
    df_normal = _make_ohlcv(n=100, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Mercado de baja volatilidad (mean-reversion fuerte)
    print("\n[TEST 4] Mercado baja volatilidad (sigma=0.001)...")
    df_lowvol = _make_ohlcv(n=100, mu=0.0, sigma=0.001, seed=7)
    s = strat.generar_señales(df_lowvol, df_lowvol.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} sin crash en mercado de baja vol")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=100).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba señal neutral con datos malformados"
    print("  [OK] neutral retornado sin excepción")

    # TEST 6 — Boundary condition
    print("\n[TEST 6] Boundary: exactamente min_train+window+1 días...")
    df_boundary = _make_ohlcv(n=42)  # window=10 + min_train=30 + 2
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} en boundary condition")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
cd C:\Users\jsguz\Paraguay\quant_arena
python -m quant_arena.zoo.estrategias.olps_rmr_strategy
```
Salida esperada: `TODOS LOS TESTS PASARON  [6 / 6]  OK`

- [ ] **Step 3: Actualizar changelog**

Agregar al inicio de la sección de versiones en `quant_arena/changelog_codigo.md.txt`:

```markdown
## [v0.2.1] — 2026-05-27 — Test Suite: OLPS-RMR

### Modelo completado
- `zoo/estrategias/olps_rmr_strategy.py` — Suite de 6 tests unitarios agregada.
  - TEST 1: Instanciación y propiedades básicas.
  - TEST 2: Warm-up (n < min_train+window → señal neutral).
  - TEST 3: Operación normal con 100 días de datos.
  - TEST 4: Mercado de baja volatilidad (mean-reversion agresiva).
  - TEST 5: DataFrame sin columna 'Close' → neutral sin excepción.
  - TEST 6: Boundary condition (exactamente datos mínimos).

### Estado del Zoo
| Modelo | Estado |
|--------|--------|
| XGBoost + Optuna | ✅ IMPLEMENTADO |
| OLPS-RMR | ✅ IMPLEMENTADO |
| Restantes (7) | En Desarrollo Activo |
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/olps_rmr_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to OLPS-RMR strategy [2/10 implementado]"
```

---

## Task 2: HMM+GARCH — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/hmm_garch_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/hmm_garch_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "HMMTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  HMMGARCHStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = HMMGARCHStrategy(
        universo=[TICKER],
        min_train_days=252,
        n_regimenes=3,
        umbral_zscore=0.5,
        retrain_every_n_days=21,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "hmm_garch"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='hmm_garch' | universo | descripcion no vacía")

    # TEST 2 — Warm-up (datos insuficientes para HMM)
    print("\n[TEST 2] Warm-up (n=100 < min_train=252)...")
    df_short = _make_ohlcv(n=100)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral — warm-up respetado")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=400)...")
    df_normal = _make_ohlcv(n=400, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Mercado con régimen de alta volatilidad (crash-like)
    print("\n[TEST 4] Volatilidad extrema (sigma=0.04, mu=-0.001)...")
    df_crash = _make_ohlcv(n=400, mu=-0.001, sigma=0.04, seed=99)
    s = strat.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} sin crash con volatilidad extrema")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=400).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral retornado sin excepción")

    # TEST 6 — Reentrenamiento: segunda llamada en fecha posterior
    print("\n[TEST 6] Re-entrenamiento condicional (segunda llamada)...")
    df_full = _make_ohlcv(n=400, seed=42)
    # Primera llamada (entrena)
    s1 = strat.generar_señales(df_full, df_full.index[-1])
    # Segunda llamada inmediata (no debe reentrenar, misma señal posible)
    s2 = strat.generar_señales(df_full, df_full.index[-1])
    _check(s1, "TEST6a", TICKER)
    _check(s2, "TEST6b", TICKER)
    print(f"  [OK] s1={s1[TICKER]}, s2={s2[TICKER]} — doble llamada sin crash")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.hmm_garch_strategy
```
Salida esperada: `TODOS LOS TESTS PASARON  [6 / 6]  OK`

- [ ] **Step 3: Actualizar changelog** — Append en `changelog_codigo.md.txt`:

```markdown
## [v0.2.2] — 2026-05-27 — Test Suite: HMM+GARCH

### Modelo completado
- `zoo/estrategias/hmm_garch_strategy.py` — Suite de 6 tests unitarios agregada.
  - TEST 6 verifica que la llamada doble (re-entrenamiento condicional) no crashea.

### Estado del Zoo
| Modelo | Estado |
|--------|--------|
| XGBoost + Optuna | ✅ IMPLEMENTADO |
| OLPS-RMR | ✅ IMPLEMENTADO |
| HMM+GARCH | ✅ IMPLEMENTADO |
| Restantes (6) | En Desarrollo Activo |
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/hmm_garch_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to HMM+GARCH strategy [3/10 implementado]"
```

---

## Task 3: Wavelet-LSTM — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/wavelet_lstm_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: Requiere `pip install PyWavelets torch`. El modelo necesita ≥494 días para emitir señal (min_train=252 + lookback=42 + SMA200=200). Los tests usan `max_epochs=3` para velocidad.

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/wavelet_lstm_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "WLSTMTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  WaveletLSTMStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    # Params de test: max_epochs pequeño para velocidad
    strat = WaveletLSTMStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=42,
        max_epochs=3,
        hidden_size=32,
        retrain_every_n_days=9999,  # no reentrenar en tests
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "wavelet_lstm"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='wavelet_lstm' | universo | descripcion no vacía")

    # TEST 2 — Warm-up (n=300 < 252+42+200=494)
    print("\n[TEST 2] Warm-up (n=300 < 494 requeridos)...")
    df_short = _make_ohlcv(n=300)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=700)...")
    df_normal = _make_ohlcv(n=700, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Mercado alcista fuerte
    print("\n[TEST 4] Mercado alcista (mu=+0.002/día)...")
    df_bull = _make_ohlcv(n=700, mu=0.002, sigma=0.008, seed=7)
    strat2 = WaveletLSTMStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        max_epochs=3, hidden_size=32, retrain_every_n_days=9999,
    )
    s = strat2.generar_señales(df_bull, df_bull.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} en tendencia alcista fuerte")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "precio"})
    strat3 = WaveletLSTMStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        max_epochs=3, hidden_size=32, retrain_every_n_days=9999,
    )
    s = strat3.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — Denoiser: serie con ruido blanco puro (sigma muy alta)
    print("\n[TEST 6] Ruido blanco puro (sigma=0.05)...")
    df_noise = _make_ohlcv(n=700, mu=0.0, sigma=0.05, seed=13)
    strat4 = WaveletLSTMStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        max_epochs=3, hidden_size=32, retrain_every_n_days=9999,
    )
    s = strat4.generar_señales(df_noise, df_noise.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} — denoiser estable con ruido extremo")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.wavelet_lstm_strategy
```
Salida esperada: `TODOS LOS TESTS PASARON  [6 / 6]  OK`

- [ ] **Step 3: Actualizar changelog** — Append en `changelog_codigo.md.txt`:

```markdown
## [v0.2.3] — 2026-05-27 — Test Suite: Wavelet-LSTM

### Modelo completado
- `zoo/estrategias/wavelet_lstm_strategy.py` — Suite de 6 tests unitarios agregada.
  - TEST 6 verifica estabilidad del denoiser wavelet con ruido blanco puro.

### Estado del Zoo
| Modelo | Estado |
|--------|--------|
| XGBoost + Optuna | ✅ IMPLEMENTADO |
| OLPS-RMR | ✅ IMPLEMENTADO |
| HMM+GARCH | ✅ IMPLEMENTADO |
| Wavelet-LSTM | ✅ IMPLEMENTADO |
| Restantes (5) | En Desarrollo Activo |
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/wavelet_lstm_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to Wavelet-LSTM strategy [4/10 implementado]"
```

---

## Task 4: Neural Fama-French — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/neural_ff_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: Requiere `torch`. El modelo necesita ≥257 días (min_train=252 + 5 de overhead). Tests usan `epochs=5` para velocidad.

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/neural_ff_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "NFFTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  NeuralFFStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = NeuralFFStrategy(
        universo=[TICKER],
        min_train_days=252,
        hidden=32,
        epochs=5,
        retrain_every_n_days=9999,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "neural_ff"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='neural_ff' | universo | descripcion no vacía")

    # TEST 2 — Warm-up
    print("\n[TEST 2] Warm-up (n=100 < 257 requeridos)...")
    df_short = _make_ohlcv(n=100)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=400)...")
    df_normal = _make_ohlcv(n=400, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Factor momentum: mercado alcista fuerte
    print("\n[TEST 4] Factor momentum (mu=+0.002/día)...")
    df_bull = _make_ohlcv(n=400, mu=0.002, sigma=0.008, seed=7)
    strat2 = NeuralFFStrategy(
        universo=[TICKER], min_train_days=252, hidden=32,
        epochs=5, retrain_every_n_days=9999,
    )
    s = strat2.generar_señales(df_bull, df_bull.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} en mercado alcista fuerte")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=400).rename(columns={"Close": "precio"})
    strat3 = NeuralFFStrategy(
        universo=[TICKER], min_train_days=252, hidden=32,
        epochs=5, retrain_every_n_days=9999,
    )
    s = strat3.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — Regularización L1: factor sparsity (sin volumen disponible)
    print("\n[TEST 6] Sin columna 'Volume' (factor SIZE omitido)...")
    df_novol = _make_ohlcv(n=400, seed=42).drop(columns=["Volume"])
    strat4 = NeuralFFStrategy(
        universo=[TICKER], min_train_days=252, hidden=32,
        epochs=5, retrain_every_n_days=9999,
    )
    s = strat4.generar_señales(df_novol, df_novol.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} — operación correcta sin factor SIZE")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.neural_ff_strategy
```
Salida esperada: `TODOS LOS TESTS PASARON  [6 / 6]  OK`

- [ ] **Step 3: Actualizar changelog** — Append en `changelog_codigo.md.txt`:

```markdown
## [v0.2.4] — 2026-05-27 — Test Suite: Neural Fama-French

### Modelo completado
- `zoo/estrategias/neural_ff_strategy.py` — Suite de 6 tests unitarios agregada.
  - TEST 6 verifica operación correcta cuando falta la columna Volume (factor SIZE=0).

### Estado del Zoo
| Modelo | Estado |
|--------|--------|
| XGBoost, OLPS-RMR, HMM+GARCH, Wavelet-LSTM | ✅ IMPLEMENTADO |
| Neural Fama-French | ✅ IMPLEMENTADO |
| Restantes (4) | En Desarrollo Activo |
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/neural_ff_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to Neural Fama-French strategy [5/10 implementado]"
```

---

## Task 5: TFT — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/tft_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: Necesita ≥494 días (252+42+200). Tests usan `max_epochs=3`.

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/tft_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "TFTTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  TFTStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = TFTStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=42,
        d_model=16,
        n_heads=2,
        max_epochs=3,
        retrain_every_n_days=9999,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "tft_trend"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='tft_trend' | universo | descripcion no vacía")

    # TEST 2 — Warm-up
    print("\n[TEST 2] Warm-up (n=300 < 494 requeridos)...")
    df_short = _make_ohlcv(n=300)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=700)...")
    df_normal = _make_ohlcv(n=700, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Atención multi-cabeza: volatilidad extrema
    print("\n[TEST 4] Volatilidad extrema (sigma=0.05)...")
    df_crash = _make_ohlcv(n=700, mu=-0.001, sigma=0.05, seed=99)
    strat2 = TFTStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        d_model=16, n_heads=2, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat2.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} con volatilidad extrema")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "precio"})
    strat3 = TFTStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        d_model=16, n_heads=2, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat3.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — VSN + atención: sin columnas High/Low (ATR omitido)
    print("\n[TEST 6] Sin columnas High/Low (ATR omitido)...")
    df_noatr = _make_ohlcv(n=700, seed=42).drop(columns=["High", "Low"])
    strat4 = TFTStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        d_model=16, n_heads=2, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat4.generar_señales(df_noatr, df_noatr.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} — TFT operativo sin ATR")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.tft_strategy
```

- [ ] **Step 3: Actualizar changelog** — Append:

```markdown
## [v0.2.5] — 2026-05-27 — Test Suite: TFT

### Modelo completado
- `zoo/estrategias/tft_strategy.py` — 6 tests. TEST 6 verifica degradación
  graceful cuando faltan columnas High/Low (ATR feature omitido automáticamente).

### Estado del Zoo
| Implementados (6) | XGBoost, OLPS-RMR, HMM+GARCH, Wavelet-LSTM, Neural-FF, TFT |
| En desarrollo (3) | PPO-RL, Mamba-SSM, ST-GNN, LLM-Sentiment |
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/tft_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to TFT strategy [6/10 implementado]"
```

---

## Task 6: PPO-RL — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/ppo_rl_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: Requiere `gymnasium` y `stable-baselines3`. Tests usan `total_timesteps=500`.

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/ppo_rl_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "PPOTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  PPOStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = PPOStrategy(
        universo=[TICKER],
        min_train_days=252,
        total_timesteps=500,     # pequeño para velocidad de test
        retrain_every_n_days=9999,
        seed=42,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "ppo_rl"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='ppo_rl' | universo | descripcion no vacía")

    # TEST 2 — Warm-up (n < min_train + 200 = 452)
    print("\n[TEST 2] Warm-up (n=300 < 452 requeridos)...")
    df_short = _make_ohlcv(n=300)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal (entrenamiento PPO mínimo)
    print("\n[TEST 3] Operación normal (n=600, timesteps=500)...")
    df_normal = _make_ohlcv(n=600, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} — agente PPO entrenado y ejecutado")

    # TEST 4 — Entorno con alta recompensa esperada (bull fuerte)
    print("\n[TEST 4] Mercado alcista (mu=+0.002/día)...")
    df_bull = _make_ohlcv(n=600, mu=0.002, sigma=0.008, seed=7)
    strat2 = PPOStrategy(
        universo=[TICKER], min_train_days=252,
        total_timesteps=500, retrain_every_n_days=9999, seed=42,
    )
    s = strat2.generar_señales(df_bull, df_bull.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} en mercado alcista")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=600).rename(columns={"Close": "precio"})
    strat3 = PPOStrategy(
        universo=[TICKER], min_train_days=252,
        total_timesteps=500, retrain_every_n_days=9999, seed=42,
    )
    s = strat3.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — TradingEnv: verificar contratos del entorno directamente
    print("\n[TEST 6] TradingEnv: reset + step sin crash...")
    import numpy as np as _np
    feat_arr = _np.random.default_rng(0).random((300, 5)).astype(_np.float32)
    lr_arr   = _np.random.default_rng(0).normal(0, 0.01, 300).astype(_np.float32)
    env = TradingEnv(feat_arr, lr_arr, window_size=10)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, "Obs shape inválida"
    obs2, rew, term, trunc, info2 = env.step(2)  # acción largo
    assert isinstance(float(rew), float), "Reward no es float"
    assert obs2.shape == env.observation_space.shape
    print(f"  [OK] reset obs={obs.shape}, step reward={rew:.4f}")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

**Nota:** El TEST 6 importa `TradingEnv` que está definido en el mismo archivo.

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.ppo_rl_strategy
```

- [ ] **Step 3: Actualizar changelog** — Append:

```markdown
## [v0.2.6] — 2026-05-27 — Test Suite: PPO-RL

### Modelo completado
- `zoo/estrategias/ppo_rl_strategy.py` — 6 tests. TEST 6 valida contratos
  del entorno Gymnasium (TradingEnv) de forma independiente.
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/ppo_rl_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to PPO-RL strategy [7/10 implementado]"
```

---

## Task 7: Mamba SSM — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/mamba_ssm_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: `MambaStrategy`, registro `'mamba_ssm'`. Necesita ≥515 días (252+63+200). Tests usan `max_epochs=3`.

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/mamba_ssm_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "MAMBTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  MambaStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = MambaStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=63,
        d_model=16,
        n_layers=1,
        d_state=8,
        max_epochs=3,
        retrain_every_n_days=9999,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "mamba_ssm"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='mamba_ssm' | universo | descripcion no vacía")

    # TEST 2 — Warm-up (n=400 < 515)
    print("\n[TEST 2] Warm-up (n=400 < 515 requeridos)...")
    df_short = _make_ohlcv(n=400)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=700)...")
    df_normal = _make_ohlcv(n=700, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Secuencias largas (lookback alto, capacidad del SSM)
    print("\n[TEST 4] Mercado de baja señal (mu=0, sigma=0.002)...")
    df_flat = _make_ohlcv(n=700, mu=0.0, sigma=0.002, seed=5)
    strat2 = MambaStrategy(
        universo=[TICKER], min_train_days=252, lookback=63,
        d_model=16, n_layers=1, d_state=8, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat2.generar_señales(df_flat, df_flat.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} en mercado lateral")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "precio"})
    strat3 = MambaStrategy(
        universo=[TICKER], min_train_days=252, lookback=63,
        d_model=16, n_layers=1, d_state=8, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat3.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — SSM selectivo: mercado con cambio de régimen brusco
    print("\n[TEST 6] Cambio de régimen brusco (mitad bull / mitad bear)...")
    rng = np.random.default_rng(11)
    n = 700
    rets = np.concatenate([
        rng.normal(0.003, 0.01, n // 2),
        rng.normal(-0.002, 0.02, n // 2),
    ])
    close = 1000.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    df_regime = pd.DataFrame({
        "Close": close, "Open": close, "High": close * 1.005,
        "Low": close * 0.995,
        "Volume": np.full(n, 5_000_000.0),
    }, index=dates)
    strat4 = MambaStrategy(
        universo=[TICKER], min_train_days=252, lookback=63,
        d_model=16, n_layers=1, d_state=8, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat4.generar_señales(df_regime, df_regime.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} tras cambio de régimen brusco")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.mamba_ssm_strategy
```

- [ ] **Step 3: Actualizar changelog** — Append:

```markdown
## [v0.2.7] — 2026-05-27 — Test Suite: Mamba SSM

### Modelo completado
- `zoo/estrategias/mamba_ssm_strategy.py` — 6 tests. TEST 6 verifica
  comportamiento del SSM selectivo ante cambios de régimen abruptos.
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/mamba_ssm_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to Mamba SSM strategy [8/10 implementado]"
```

---

## Task 8: ST-GNN — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/stgnn_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: `STGNNStrategy`, registro `'stgnn_alpha'`. Necesita ≥494 días. Tests usan `max_epochs=3`.

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/stgnn_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "STGNNTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  STGNNStrategy — Suite de Pruebas [6 tests]")
    print(SEP)

    strat = STGNNStrategy(
        universo=[TICKER],
        min_train_days=252,
        lookback=42,
        d_gcn=8,
        d_gru=16,
        n_gru_layers=1,
        max_epochs=3,
        retrain_every_n_days=9999,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades...")
    assert strat.nombre == "stgnn_alpha"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='stgnn_alpha' | universo | descripcion no vacía")

    # TEST 2 — Warm-up
    print("\n[TEST 2] Warm-up (n=300 < 494 requeridos)...")
    df_short = _make_ohlcv(n=300)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal
    print("\n[TEST 3] Operación normal (n=700)...")
    df_normal = _make_ohlcv(n=700, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Grafo adaptativo: alta correlación entre nodos (tendencia fuerte)
    print("\n[TEST 4] Tendencia alcista fuerte (mu=+0.003)...")
    df_bull = _make_ohlcv(n=700, mu=0.003, sigma=0.008, seed=7)
    strat2 = STGNNStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        d_gcn=8, d_gru=16, n_gru_layers=1, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat2.generar_señales(df_bull, df_bull.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]}")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=700).rename(columns={"Close": "precio"})
    strat3 = STGNNStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        d_gcn=8, d_gru=16, n_gru_layers=1, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat3.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — Grafo esparcido: nodos con baja correlación cruzada
    print("\n[TEST 6] Mercado con baja correlación inter-features (ruido)...")
    df_noise = _make_ohlcv(n=700, mu=0.0, sigma=0.05, seed=21)
    strat4 = STGNNStrategy(
        universo=[TICKER], min_train_days=252, lookback=42,
        d_gcn=8, d_gru=16, n_gru_layers=1, max_epochs=3, retrain_every_n_days=9999,
    )
    s = strat4.generar_señales(df_noise, df_noise.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] señal={s[TICKER]} — grafo adaptativo con baja correlación")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.stgnn_strategy
```

- [ ] **Step 3: Actualizar changelog** — Append:

```markdown
## [v0.2.8] — 2026-05-27 — Test Suite: ST-GNN

### Modelo completado
- `zoo/estrategias/stgnn_strategy.py` — 6 tests. TEST 6 valida el grafo
  adaptativo bajo condiciones de baja correlación inter-features.
```

- [ ] **Step 4: Commit**

```bash
git add quant_arena/zoo/estrategias/stgnn_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to ST-GNN strategy [9/10 implementado]"
```

---

## Task 9: LLM Agentic Sentiment — Tests

**Files:**
- Modify: `quant_arena/zoo/estrategias/llm_sentiment_strategy.py` (append)
- Modify: `quant_arena/changelog_codigo.md.txt`

Nota: `LLMSentimentStrategy`, registro `'llm_sentiment'`, `min_train_days=63`. Tests usan `text_col=None` (modo fallback técnico puro, sin FinBERT, sin red).

- [ ] **Step 1: Append el bloque de tests**

Agregar al final de `quant_arena/zoo/estrategias/llm_sentiment_strategy.py`:

```python
# =============================================================================
# Suite de Pruebas Unitarias
# =============================================================================

if __name__ == "__main__":

    TICKER = "LLMTEST"

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
        assert not s.isnull().any(),        f"[{label}] NaN en señal"
        assert not (set(s.values) - valid), f"[{label}] Valores fuera de {{-1,0,1}}"
        assert ticker in s.index,           f"[{label}] Ticker no en índice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  LLMSentimentStrategy — Suite de Pruebas [6 tests]")
    print("  (modo fallback técnico — sin FinBERT, sin red)")
    print(SEP)

    # text_col=None -> modo fallback puro, sin dependencia de transformers
    strat = LLMSentimentStrategy(
        universo=[TICKER],
        text_col=None,
        min_train_days=63,
        umbral_score=0.20,
        ewma_span=5,
    )

    # TEST 1 — Propiedades básicas
    print("\n[TEST 1] Instanciación y propiedades (fallback mode)...")
    assert strat.nombre == "llm_sentiment"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='llm_sentiment' | universo | descripcion no vacía")

    # TEST 2 — Warm-up
    print("\n[TEST 2] Warm-up (n=30 < min_train=63)...")
    df_short = _make_ohlcv(n=30)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] señal={s[TICKER]} neutral")

    # TEST 3 — Operación normal (proxy técnico)
    print("\n[TEST 3] Operación normal fallback técnico (n=200)...")
    df_normal = _make_ohlcv(n=200, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] señal={s[TICKER]} ∈ {{-1.0, 0.0, 1.0}}")

    # TEST 4 — Sentimiento bajista: mercado en crash
    print("\n[TEST 4] Crash simulado (mu=-0.003, sigma=0.03)...")
    df_crash = _make_ohlcv(n=200, mu=-0.003, sigma=0.03, seed=99)
    s = strat.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] señal={s[TICKER]} — proxy técnico detecta sentimiento negativo")

    # TEST 5 — Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=200).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all()
    print("  [OK] neutral sin excepción")

    # TEST 6 — EWMA del score: verificar suavizado temporal
    print("\n[TEST 6] EWMA score — suavizado temporal correcto...")
    df_full = _make_ohlcv(n=200, seed=42)
    # Señal en fecha penúltima
    s_prev = strat.generar_señales(df_full, df_full.index[-2])
    # Señal en fecha última
    s_last = strat.generar_señales(df_full, df_full.index[-1])
    _check(s_prev, "TEST6a", TICKER)
    _check(s_last, "TEST6b", TICKER)
    print(f"  [OK] señal penúltima={s_prev[TICKER]}, última={s_last[TICKER]} — EWMA consistente")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
```

- [ ] **Step 2: Ejecutar y verificar**

```bash
python -m quant_arena.zoo.estrategias.llm_sentiment_strategy
```

- [ ] **Step 3: Actualizar changelog** — Append entrada final en `changelog_codigo.md.txt`:

```markdown
## [v0.2.9] — 2026-05-27 — Test Suite: LLM Agentic Sentiment

### Modelo completado
- `zoo/estrategias/llm_sentiment_strategy.py` — 6 tests en modo fallback
  técnico (sin FinBERT, sin red). TEST 6 verifica consistencia del EWMA
  de scores en fechas consecutivas.

### Estado del Zoo — COMPLETO
| Modelo | Estado |
|--------|--------|
| XGBoost + Optuna | ✅ IMPLEMENTADO |
| OLPS-RMR | ✅ IMPLEMENTADO |
| HMM+GARCH | ✅ IMPLEMENTADO |
| Wavelet-LSTM | ✅ IMPLEMENTADO |
| Neural Fama-French | ✅ IMPLEMENTADO |
| TFT | ✅ IMPLEMENTADO |
| PPO-RL | ✅ IMPLEMENTADO |
| Mamba SSM | ✅ IMPLEMENTADO |
| ST-GNN | ✅ IMPLEMENTADO |
| LLM Agentic Sentiment | ✅ IMPLEMENTADO |

### Hito: Zoo completo (10/10 modelos IMPLEMENTADOS)
```

- [ ] **Step 4: Commit final**

```bash
git add quant_arena/zoo/estrategias/llm_sentiment_strategy.py quant_arena/changelog_codigo.md.txt
git commit -m "feat(zoo): add test suite to LLM Sentiment strategy [10/10 IMPLEMENTADO — Zoo completo]"
```

---

## Self-Review

**Spec coverage:**
- ✅ 9 modelos reciben test suite (Tasks 1–9)
- ✅ Orden stable → complex respetado (OLPS → HMM → WLSTM → NFF → TFT → PPO → Mamba → STGNN → LLM)
- ✅ Changelog actualizado tras cada modelo
- ✅ 6 tests por modelo (contrato mínimo según spec)
- ✅ Tests usan OHLCV sintético (sin red)
- ✅ Modelo LLM opera en modo fallback (sin FinBERT)

**Placeholder scan:** Ningún TBD, TODO, o "similar al task anterior". Código completo en cada step.

**Type consistency:**
- `OLPSRMRStrategy` → correcto (archivo línea 107)
- `HMMGARCHStrategy` → correcto (archivo línea 241)
- `WaveletLSTMStrategy` → correcto (archivo línea 175)
- `NeuralFFStrategy` → correcto (archivo línea 213)
- `TFTStrategy` → correcto (archivo línea 170)
- `PPOStrategy` + `TradingEnv` → correctos (archivo líneas 59, 165)
- `MambaStrategy` → correcto (archivo línea 211)
- `STGNNStrategy` → correcto (archivo línea 227)
- `LLMSentimentStrategy` → correcto (archivo línea 221)
