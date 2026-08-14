#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/hmm_garch_strategy.py
# HMM+GARCH Strategy — Arbitraje por cambio de regimen y clustering de volatilidad
# =============================================================================
"""
Combina Hidden Markov Models (HMM) para deteccion de regimenes de mercado con
modelos GARCH(1,1) para pronostico de volatilidad condicional por regimen.

Pipeline:
    log_retornos -> GaussianHMM(K=3) -> regimen_actual
    -> GARCH(regimen) -> (mu_hat, sigma_hat_t+1)
    -> z_score = mu_hat / sigma_hat -> señal discretizada

Los K=3 regimenes se ordenan por mu y se mapean a:
    Estado 0: Bearish (mu mas negativa)
    Estado 1: Neutral / lateral
    Estado 2: Bullish (mu mas positiva)

El z-score condicional al regimen es equivalente a un Sharpe ex-ante por periodo.

Registro: 'hmm_garch'
"""
from __future__ import annotations

import logging
import sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except ImportError:
    _HMM_OK = False

try:
    from arch import arch_model
    _ARCH_OK = True
except ImportError:
    _ARCH_OK = False

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.core.excepciones import ConfiguracionInvalidaError, SupuestoEstadisticoError
from quant_arena.diagnostics.assumption_validator import AssumptionReport, AssumptionValidator
from quant_arena.diagnostics.model_selection import ReporteSeleccionK, seleccionar_k_hmm
from quant_arena.zoo.base_estrategia import RegistroZoo

logger = logging.getLogger(__name__)


# =============================================================================
# Componentes del modelo
# =============================================================================

class RegimeDetector:
    """
    Detecta regimenes de mercado mediante Gaussian HMM.

    Los regimenes se ordenan en cada ajuste por la media del retorno,
    garantizando consistencia semantica entre re-entrenamientos:
        indice 0 -> regimen de menor retorno promedio (bearish)
        indice 1 -> regimen intermedio
        indice 2 -> regimen de mayor retorno promedio (bullish)

    Args:
        n_regimenes:   Numero de estados ocultos del HMM (recomendado: 2-4).
        n_iter:        Iteraciones maximas del algoritmo Baum-Welch.
        random_state:  Semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_regimenes:  int = 3,
        n_iter:       int = 100,
        random_state: int = 42,
    ) -> None:
        if not _HMM_OK:
            raise ImportError("pip install hmmlearn")
        self._n_regimenes = n_regimenes
        self._hmm: Optional[GaussianHMM] = None
        self._orden_estados: Optional[np.ndarray] = None  # mapeo al orden semantico
        self._n_iter = n_iter
        self._rng    = random_state
        self.convergio: Optional[bool] = None  # diagnóstico real de Baum-Welch (H9)

    @property
    def n_regimenes(self) -> int:
        return self._n_regimenes

    def fit(self, log_rets: np.ndarray) -> "RegimeDetector":
        """
        Ajusta el HMM sobre la serie de log-retornos.

        Los estados se reordenan por mu ascendente despues del ajuste para
        garantizar que el estado 0 siempre sea el mas bajista.

        Args:
            log_rets: array (T,) de log-retornos diarios.
        """
        X = log_rets.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # solo el texto de consola de hmmlearn
            self._hmm = GaussianHMM(
                n_components = self._n_regimenes,
                covariance_type = "full",
                n_iter = self._n_iter,
                random_state = self._rng,
                tol = 1e-4,
            )
            self._hmm.fit(X)

        # Diagnóstico REAL de convergencia (H9): se lee monitor_.converged en
        # vez de solo suprimir el warning de consola sin verificar la señal.
        self.convergio = bool(self._hmm.monitor_.converged)
        if not self.convergio:
            logger.warning(
                f"RegimeDetector: Baum-Welch no convergió (K={self._n_regimenes}, "
                f"n_iter={self._n_iter}). Los parámetros por régimen pueden no "
                "estar bien identificados."
            )

        # Ordenar estados por media de retorno (bearish->bullish)
        medias           = self._hmm.means_.flatten()
        self._orden_estados = np.argsort(medias)  # indice[0] = estado con menor mu
        return self

    def predict_regimen(self, log_rets: np.ndarray) -> int:
        """
        Predice el regimen del ultimo periodo usando la secuencia de Viterbi.

        Args:
            log_rets: array (T,) de log-retornos hasta fecha_corte.

        Returns:
            regimen: int en {0, ..., n_regimenes-1} (0=bearish, max=bullish).
        """
        if self._hmm is None or self._orden_estados is None:
            raise RuntimeError("HMM no ajustado. Llama a fit() primero.")
        X = log_rets.reshape(-1, 1)
        estados_raw     = self._hmm.predict(X)
        ultimo_raw      = int(estados_raw[-1])
        # Mapear al orden semantico
        pos_en_orden    = np.where(self._orden_estados == ultimo_raw)[0]
        return int(pos_en_orden[0]) if len(pos_en_orden) > 0 else 1

    def predict_proba_regimen(self, log_rets: np.ndarray) -> np.ndarray:
        """
        Distribución posterior P(régimen | datos) del último período, en
        orden SEMÁNTICO (índice 0 = bearish, ... índice max = bullish).

        Base para exponer la incertidumbre de régimen a componentes externos
        (ej. el denominador de Kelly en §1.2, o un dd_limit condicionado a
        P(crisis) en §1.1) sin acoplar el motor a los internals de hmmlearn.

        Returns:
            array (n_regimenes,) que suma 1.0.
        """
        if self._hmm is None or self._orden_estados is None:
            raise RuntimeError("HMM no ajustado. Llama a fit() primero.")
        X = log_rets.reshape(-1, 1)
        posterior_raw = self._hmm.predict_proba(X)[-1]  # orden interno de hmmlearn
        return posterior_raw[self._orden_estados]  # remapeado a orden semántico

    def get_params_regimen(self, regimen: int) -> Tuple[float, float]:
        """
        Retorna (mu, sigma) del regimen especificado (en orden semantico).

        Returns:
            (mu, sigma): media y desviacion estandar del estado.
        """
        if self._hmm is None or self._orden_estados is None:
            raise RuntimeError("HMM no ajustado.")
        estado_original = int(self._orden_estados[regimen])
        mu    = float(self._hmm.means_[estado_original, 0])
        sigma = float(np.sqrt(self._hmm.covars_[estado_original, 0, 0]))
        return mu, sigma

    def get_retornos_por_regimen(
        self, log_rets: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Segmenta los retornos historicos por regimen (orden semantico).

        Util para ajustar un GARCH separado por cada regimen.
        """
        if self._hmm is None or self._orden_estados is None:
            return {}
        X           = log_rets.reshape(-1, 1)
        estados_raw = self._hmm.predict(X)
        resultado   = {}
        for sem_idx, raw_idx in enumerate(self._orden_estados):
            mask = estados_raw == raw_idx
            rets = log_rets[mask]
            if len(rets) > 20:
                resultado[sem_idx] = rets
        return resultado


class GARCHModeler:
    """
    Pronostico de volatilidad condicional via GARCH(1,1) por regimen.

    Para cada regimen se ajusta un modelo GARCH(1,1) independiente sobre
    los retornos asociados a ese estado. El pronostico one-step-ahead
    estima la varianza condicional del siguiente periodo.

    Args:
        dist: Distribucion de errores GARCH ('normal' | 'skewt' | 't').
    """

    def __init__(self, dist: str = "normal") -> None:
        if not _ARCH_OK:
            raise ImportError("pip install arch")
        self._dist   = dist
        self._modelos: Dict[int, object] = {}   # {regimen: modelo ajustado}
        self._sigma_fallback: float = 0.01      # sigma por defecto si GARCH falla

    def fit(self, retornos_por_regimen: Dict[int, np.ndarray]) -> "GARCHModeler":
        """
        Ajusta un GARCH(1,1) por cada regimen disponible.

        Args:
            retornos_por_regimen: {regimen: array de log-retornos del regimen}.
        """
        self._modelos = {}
        for regimen, rets in retornos_por_regimen.items():
            if len(rets) < 30:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = arch_model(
                        rets * 100.0,   # escalar para estabilidad numerica
                        vol = "Garch",
                        p   = 1,
                        q   = 1,
                        dist = self._dist,
                        rescale = False,
                    )
                    res = m.fit(disp="off", show_warning=False)
                    self._modelos[regimen] = res
            except Exception:
                pass
        return self

    def forecast_sigma(self, regimen: int, ultimos_rets: np.ndarray) -> float:
        """
        Pronostica la volatilidad condicional para el siguiente periodo.

        Args:
            regimen:     Regimen actual (semantico).
            ultimos_rets: array de retornos recientes (solo del regimen actual).

        Returns:
            sigma_hat: desviacion estandar pronosticada (en escala original).
        """
        if regimen not in self._modelos:
            return self._sigma_fallback
        try:
            model_fit = self._modelos[regimen]
            fcast     = model_fit.forecast(horizon=1, reindex=False)
            var_hat   = float(fcast.variance.iloc[-1, 0]) / 10_000.0   # revertir escala
            return float(np.sqrt(max(var_hat, 1e-10)))
        except Exception:
            return self._sigma_fallback


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("hmm_garch")
class HMMGARCHStrategy(AbstractStrategy):
    """
    Estrategia HMM+GARCH para quant_arena.

    Detecta el regimen de mercado actual con HMM y usa GARCH para pronosticar
    la volatilidad condicional. La señal es el z-score condicional:

        z = mu_regimen / sigma_hat_GARCH

    Interpretacion: z >> 0 -> largo (retorno esperado alto relativo a riesgo);
                    z << 0 -> corto; |z| < umbral -> neutral.

    Args:
        universo:             Tickers (un elemento).
        min_train_days:       Minimo de dias para ajustar HMM y GARCH.
        n_regimenes:          Numero de estados HMM (fijo, salvo que
                              `seleccionar_k_automaticamente=True`).
        umbral_zscore:        Umbral del z-score condicional para señal.
        retrain_every_n_days: Dias entre re-ajustes.
        close_col:            Columna de cierre.
        seed:                 Semilla.
        seleccionar_k_automaticamente: Si True, en cada re-ajuste se
                              selecciona K por BIC (`diagnostics.model_selection`)
                              en vez de usar `n_regimenes` fijo (H9). Default
                              False — comportamiento idéntico al original.
        k_range:              Rango de K a evaluar cuando la selección
                              automática está activa.
        validar_normalidad:   Si True, contrasta normalidad de los
                              log-retornos (`diagnostics.assumption_validator`)
                              en cada re-ajuste y usa GARCH dist='t' si se
                              rechaza, en vez de 'normal' fijo (H9). Default
                              False — comportamiento idéntico al original.
        alfa_supuestos:       Nivel de significancia para `validar_normalidad`.
    """

    def __init__(
        self,
        universo:             List[str],
        min_train_days:       int   = 252,
        n_regimenes:          int   = 3,
        umbral_zscore:        float = 0.5,
        retrain_every_n_days: int   = 21,
        close_col:            str   = "Close",
        seed:                 int   = 42,
        seleccionar_k_automaticamente: bool = False,
        k_range:              Sequence[int] = range(2, 7),
        validar_normalidad:   bool = False,
        alfa_supuestos:       float = 0.05,
    ) -> None:
        if not _HMM_OK:
            raise ImportError("pip install hmmlearn")
        if not _ARCH_OK:
            raise ImportError("pip install arch")

        super().__init__(nombre="hmm_garch", universo=universo)

        self._min_train    = min_train_days
        self._n_regimenes  = n_regimenes
        self._umbral       = umbral_zscore
        self._retrain_days = retrain_every_n_days
        self._close_col    = close_col
        self._seed         = seed

        self._seleccionar_k    = seleccionar_k_automaticamente
        self._k_range          = k_range
        self._validar_normalidad = validar_normalidad
        self._alfa_supuestos   = alfa_supuestos

        self._detector: Optional[RegimeDetector] = None
        self._garch:    Optional[GARCHModeler]   = None
        self._ultimo_entrenamiento: Optional[pd.Timestamp] = None

        # Diagnósticos del último re-ajuste (introspección para reportes/tests)
        self._ultimo_reporte_seleccion_k: Optional[ReporteSeleccionK] = None
        self._ultimo_reporte_normalidad:  Optional[AssumptionReport] = None

    @property
    def descripcion(self) -> str:
        return (
            f"HMM+GARCH | regimenes={self._n_regimenes} "
            f"| umbral_z=+-{self._umbral:.2f} | retrain={self._retrain_days}d"
        )

    @property
    def ultimo_reporte_seleccion_k(self) -> Optional[ReporteSeleccionK]:
        """Reporte de `seleccionar_k_hmm` del último re-ajuste (None si desactivado)."""
        return self._ultimo_reporte_seleccion_k

    @property
    def ultimo_reporte_normalidad(self) -> Optional[AssumptionReport]:
        """Reporte de `test_normalidad` del último re-ajuste (None si desactivado)."""
        return self._ultimo_reporte_normalidad

    # ------------------------------------------------------------------
    def _ajustar(self, log_rets: np.ndarray) -> None:
        """Ajusta HMM y GARCH sobre la serie de log-retornos."""
        np.random.seed(self._seed)

        n_regimenes_usar = self._n_regimenes
        if self._seleccionar_k:
            try:
                reporte_k = seleccionar_k_hmm(
                    log_rets, k_range=self._k_range, random_state=self._seed
                )
                n_regimenes_usar = reporte_k.k_optimo_bic
                self._ultimo_reporte_seleccion_k = reporte_k
            except (ConfiguracionInvalidaError, SupuestoEstadisticoError) as exc:
                logger.warning(
                    f"HMMGARCHStrategy: selección de K falló ({exc}); "
                    f"usando n_regimenes={self._n_regimenes} fijo."
                )
                self._ultimo_reporte_seleccion_k = None

        # Ajustar HMM
        self._detector = RegimeDetector(
            n_regimenes  = n_regimenes_usar,
            n_iter       = 100,
            random_state = self._seed,
        )
        self._detector.fit(log_rets)

        # Obtener retornos segmentados por regimen y ajustar GARCH
        rets_por_reg = self._detector.get_retornos_por_regimen(log_rets)

        dist_garch = "normal"
        if self._validar_normalidad:
            try:
                validador = AssumptionValidator(alfa=self._alfa_supuestos)
                reporte_norm = validador.test_normalidad(log_rets)
                self._ultimo_reporte_normalidad = reporte_norm
                dist_garch = "t" if reporte_norm.rechaza_h0 else "normal"
            except (ImportError, SupuestoEstadisticoError) as exc:
                logger.warning(
                    f"HMMGARCHStrategy: test de normalidad falló ({exc}); "
                    "usando GARCH dist='normal'."
                )
                self._ultimo_reporte_normalidad = None

        self._garch = GARCHModeler(dist=dist_garch)
        self._garch.fit(rets_por_reg)

    # ------------------------------------------------------------------
    # Incertidumbre de régimen — punto de extensión opcional para §1.1/§1.2
    # ------------------------------------------------------------------

    def incertidumbre_regimen(self, log_rets: np.ndarray) -> float:
        """
        1 − max(P(régimen | datos)): alta cuando el HMM no tiene claridad
        sobre en qué régimen está el mercado ahora mismo.

        Diseñado para ser consumido, vía duck-typing, por
        `BacktestEngine._pesos_via_sizer` (§1.2): si la estrategia expone
        este método, su valor puede sumarse a σ_skill_TTT en el
        denominador de Kelly — menos claridad de régimen, menor exposición.
        No forma parte de `AbstractStrategy` (no todas las estrategias
        tienen noción de "régimen"; forzarlo violaría ISP).

        Returns:
            float en [0, 1 − 1/n_regimenes]. 0.0 si el detector no está
            ajustado (sin información, sin penalización adicional).
        """
        if self._detector is None:
            return 0.0
        try:
            posterior = self._detector.predict_proba_regimen(log_rets)
            return float(1.0 - posterior.max())
        except Exception:
            return 0.0

    def probabilidad_crisis(self, log_rets: np.ndarray) -> float:
        """
        P(régimen bearish | datos) — el régimen semántico 0 (menor μ).

        Punto de extensión opcional para §1.1: un `RiskOverlay` externo
        podría condicionar `dd_limit` a esta probabilidad (más bajo con
        P(crisis) alta) en vez de escalar solo por volatilidad realizada.

        Returns:
            float en [0, 1]. 0.0 si el detector no está ajustado.
        """
        if self._detector is None:
            return 0.0
        try:
            posterior = self._detector.predict_proba_regimen(log_rets)
            return float(posterior[0])
        except Exception:
            return 0.0

    def _generar_prediccion(self, log_rets: np.ndarray) -> float:
        """
        Genera el z-score condicional al regimen actual.

        z_score = mu_regimen / sigma_GARCH_forecast

        Un z_score alto indica que el retorno esperado en el regimen actual
        es alto relativo a la volatilidad pronosticada -> señal larga.

        Returns:
            float: z-score condicional.
        """
        if self._detector is None or self._garch is None:
            return 0.0
        try:
            regimen          = self._detector.predict_regimen(log_rets)
            mu_reg, _        = self._detector.get_params_regimen(regimen)
            sigma_hat        = self._garch.forecast_sigma(regimen, log_rets[-63:])
            if sigma_hat < 1e-10:
                return 0.0
            return mu_reg / sigma_hat
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def generar_señales(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return neutral

        close    = hist[self._close_col].astype(float)
        log_rets = np.log(close / close.shift(1)).dropna().values.astype(np.float64)

        if len(log_rets) < self._min_train:
            return neutral

        # Sanitizacion: eliminar infinitos y NaN residuales
        log_rets = log_rets[np.isfinite(log_rets)]
        if len(log_rets) < self._min_train:
            return neutral

        necesita = (
            self._ultimo_entrenamiento is None
            or (fecha_corte - self._ultimo_entrenamiento).days >= self._retrain_days
        )
        if necesita:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._ajustar(log_rets)
                except Exception:
                    return neutral
            self._ultimo_entrenamiento = fecha_corte

        z_score = self._generar_prediccion(log_rets)
        señal   = 1.0 if z_score > self._umbral else (-1.0 if z_score < -self._umbral else 0.0)

        pesos = pd.Series(0.0, index=indice, dtype=float)
        if self._universo:
            pesos.iloc[0] = señal
        return pesos

    def calcular_retornos(
        self, datos: pd.DataFrame, pesos_historicos: pd.DataFrame
    ) -> pd.Series:
        if self._close_col not in datos.columns or not self._universo:
            return pd.Series(dtype=float)
        ticker = self._universo[0]
        if ticker not in pesos_historicos.columns:
            return pd.Series(dtype=float)
        ret_al, pesos_al = datos[self._close_col].pct_change().align(
            pesos_historicos[ticker], join="inner"
        )
        return (pesos_al * ret_al).dropna()

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
        assert not s.isnull().any(),        f"[{label}] NaN en senal"
        bad = set(s.values) - valid
        assert not bad,                     f"[{label}] Valores fuera de {{-1,0,1}}: {bad}"
        assert ticker in s.index,           f"[{label}] Ticker no en indice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  HMMGARCHStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = HMMGARCHStrategy(
        universo=[TICKER],
        min_train_days=252,
        n_regimenes=3,
        umbral_zscore=0.5,
        retrain_every_n_days=21,
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "hmm_garch"
    assert strat.universo == [TICKER]
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='hmm_garch' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up (datos insuficientes para HMM)
    print("\n[TEST 2] Warm-up (n=100 < min_train=252)...")
    df_short = _make_ohlcv(n=100)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] senal={s[TICKER]} neutral -- warm-up respetado")

    # TEST 3 -- Operacion normal
    print("\n[TEST 3] Operacion normal (n=600)...")
    df_normal = _make_ohlcv(n=600, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] senal={s[TICKER]} in {{-1.0, 0.0, 1.0}}")

    # TEST 4 -- Mercado con regimen de alta volatilidad (crash-like)
    print("\n[TEST 4] Volatilidad extrema (sigma=0.04, mu=-0.001)...")
    df_crash = _make_ohlcv(n=600, mu=-0.001, sigma=0.04, seed=99)
    s = strat.generar_señales(df_crash, df_crash.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] senal={s[TICKER]} sin crash con volatilidad extrema")

    # TEST 5 -- Sin columna 'Close'
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=600).rename(columns={"Close": "precio"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert not s.isnull().any()
    assert (s == 0.0).all(), "Esperaba neutral con datos malformados"
    print("  [OK] neutral retornado sin excepcion")

    # TEST 6 -- Boundary condition: exactamente min_train_days
    print("\n[TEST 6] Boundary condition (n=252 == min_train_days)...")
    df_boundary = _make_ohlcv(n=252)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary exacto respetado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
