# Design Spec: Zoo Test Suites para los 9 Modelos Restantes

**Fecha:** 2026-05-27
**Autor:** Claude (Sonnet 4.6)
**Estado:** Aprobado

---

## Contexto

El Zoo de quant_arena alberga 10 modelos de generación de alfa. El Modelo 4
(XGBoost + Optuna) está marcado como "IMPLEMENTADO" porque posee:
1. Lógica completa de `generar_señales()` y `calcular_retornos()`.
2. Un bloque `if __name__ == "__main__":` con 7 tests sintéticos que verifican
   el contrato de la interfaz `AbstractStrategy`.

Los 9 modelos restantes tienen la lógica completa, pero **ninguno tiene suite
de tests**. El criterio de "IMPLEMENTADO" en este proyecto requiere ambas partes.

---

## Objetivo

Agregar bloques `if __name__ == "__main__":` con 6 tests unitarios a cada uno
de los 9 modelos, en orden de complejidad creciente (stable → complex).

---

## Estructura de cada test suite

| Test | Descripción | Input sintético |
|------|-------------|-----------------|
| T1 | Instanciación y propiedades básicas | N/A |
| T2 | Warm-up: señal neutral con datos insuficientes | n=100 días |
| T3 | Operación normal (datos suficientes) | n=600 días, random walk |
| T4 | Régimen extremo (bull / crash) | n=600, mu extremo |
| T5 | DataFrame sin columna 'Close' → neutral sin crash | n=600, columna renombrada |
| T6 | Boundary condition: exactamente min_train_days | n=min_train exacto |

El OHLCV sintético se genera con:
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
        "Low": close * (1 - np.abs(rng.normal(0, noise))),
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=dates)
```

Los modelos LLM (llm_sentiment) operan en modo fallback técnico (sin FinBERT)
para que los tests no requieran descarga de modelos.

---

## Orden de implementación

| Tier | Modelo | Archivo | Dependencias test |
|------|--------|---------|-------------------|
| 1 | OLPS-RMR | `olps_rmr_strategy.py` | Solo numpy/pandas |
| 2 | HMM+GARCH | `hmm_garch_strategy.py` | hmmlearn, arch |
| 3a | Wavelet-LSTM | `wavelet_lstm_strategy.py` | pywt, torch |
| 3b | Neural-FF | `neural_ff_strategy.py` | torch |
| 3c | TFT | `tft_strategy.py` | torch |
| 4 | PPO-RL | `ppo_rl_strategy.py` | gymnasium, stable-baselines3 |
| 5a | Mamba-SSM | `mamba_ssm_strategy.py` | torch |
| 5b | ST-GNN | `stgnn_strategy.py` | torch |
| 5c | LLM-Sentiment | `llm_sentiment_strategy.py` | transformers (modo fallback) |

---

## Changelog

Después de completar cada modelo, se agrega una entrada a
`quant_arena/changelog_codigo.md.txt` con:
- Versión incremental
- Modelo completado
- Tests agregados
- Fecha

---

## Invariantes que debe respetar todo test

1. **Causalidad**: nunca se pasan datos futuros a `generar_señales`.
2. **Valores válidos**: señal ∈ {-1.0, 0.0, 1.0}, sin NaN.
3. **Sin crash**: datos malformados → retornar neutral, no lanzar excepción.
4. **Contrato de índice**: la Series retornada tiene como índice `universo`.
