"""
Walk-Forward ELO Engine v3.0 — Institutional Grade + TrueSkill Through Time
=============================================================================
Refactored con rigor académico institucional (Tier 1 Hedge Fund standard).

MEJORAS SOBRE v2.0:
─────────────────────────────────────────────────────────────────────────────
[8] TRUESKILL THROUGH TIME — TTT (Landfried & Mocskos, 2024):
    Reemplaza el ranking ELO estático por inferencia Bayesiana dinámica
    sobre la habilidad latente de cada estrategia. A diferencia del ELO
    (que actualiza puntualmente), TTT modela la habilidad como un proceso
    estocástico gaussiano-gaussiano y realiza inferencia GLOBAL sobre toda
    la historia de partidos (folds OOS), produciendo:
      · mu(t): habilidad media de la estrategia en el tiempo t
      · sigma(t): incertidumbre sobre esa habilidad
    El motor acumula todos los folds OOS en listas `composition`, `results`
    y `times` (sintaxis de logistic.py) y ejecuta `History.convergence()`
    UNA sola vez al final, explotando la inferencia global que TTT fue
    diseñado para realizar. Esto es superior a llamadas iterativas porque:
      - La suavización hacia atrás (backward pass) solo tiene sentido con
        toda la historia disponible (Landfried, 2024, sec. 3.2).
      - Evita re-entrenar TTT con datos IS, que violaría el principio de
        separación IS/OOS.

    Ref: Landfried, G. & Mocskos, E. (2024). "TrueSkill Through Time:
         reliable initial skill estimates and historical comparability."
         Manuscript. GitHub: github.com/glandfried/TrueSkillThroughTime

MEJORAS SOBRE v1.0:
─────────────────────────────────────────────────────────────────────────────
[1] PURGING (López de Prado, 2018 — "Advances in Financial Machine Learning",
    Cap. 7): Eliminación de observaciones de entrenamiento cuyos horizontes de
    etiqueta se solapan con el período de test. Previene fuga de información
    (data leakage) producida por la autocorrelación de retornos.

[2] EMBARGO (López de Prado, 2018, Cap. 7): Buffer temporal entre el final
    del set de entrenamiento y el inicio del set de test para absorber la
    correlación serial residual (microestructura, momentum de corto plazo).

[3] COMBINATORIAL PURGED CROSS-VALIDATION — CPCV (López de Prado, 2018,
    Cap. 12): En lugar de un único path walk-forward (sesgo de realización
    única), CPCV genera C(N,k) splits que producen una DISTRIBUCIÓN de
    performance OOS, reduciendo el Probability of Backtest Overfitting (PBO).

[4] PROBABILISTIC SHARPE RATIO — PSR (Bailey & López de Prado, 2012,
    "The Sharpe Ratio Efficient Frontier"): Corrige el SR por longitud de
    muestra y momentos de orden superior (asimetría, curtosis).

[5] DEFLATED SHARPE RATIO — DSR (Bailey & López de Prado, 2014, Journal of
    Portfolio Management 40(5)): Ajusta el PSR por selección múltiple de
    estrategias, penalizando el número de configuraciones evaluadas.

[6] CALMAR-ADJUSTED FITNESS FUNCTION: Reemplaza la rentabilidad media como
    función objetivo por el ratio de Calmar (CAGR / Max Drawdown), que
    penaliza explícitamente el riesgo de cola (tail risk).

[7] CONDITIONAL VALUE AT RISK (CVaR / Expected Shortfall) al 5%.
    (Rockafellar & Uryasev, 2000. Journal of Risk 2(3), 21-41.)

REFERENCIAS:
─────────────────────────────────────────────────────────────────────────────
  - López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    SSRN: https://ssrn.com/abstract=3104847
  - Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
    Journal of Portfolio Management, 40(5), 94-107.
    SSRN: https://ssrn.com/abstract=2460551
  - Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient
    Frontier. Journal of Risk, 15(2), 3-44.
    SSRN: https://ssrn.com/abstract=1821643
  - Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
    Value-at-Risk. Journal of Risk, 2(3), 21-41.
  - ScienceDirect (2024). Backtest overfitting in the ML era.
    DOI: 10.1016/j.knosys.2024.111110
"""

from data_manager import DataManager
from regime_detector import (
    HMMRegimeDetector, RegimeDetectorRegistry,
    VolatilityRegimeDetector, ChangePointRegimeDetector,
    MultifractalRegimeDetector
)
from strategy_zoo import (
    StrategyRegistry, BuyAndHold, TrendFollowing, MeanReversion,
    LowVolatility, VolatilityBreakout, FadeExtremes, MomentumCrossover,
    RSIMeanReversion, RangeBreakout
)
from strategy_adapter import adapt_strategies
from backtest_engine import BacktestEngine, BacktestResult
from ranking import RankingManager, BayesianELORanking
from itertools import combinations, product
from math import comb
from utils import align_series_to_index, sanity_check_market_data

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats

# TrueSkill Through Time — Landfried & Mocskos (2024)
# Sintaxis idéntica a logistic.py (oficial del paper)
try:
    from trueskillthroughtime import History, Player, Gaussian
    _TTT_AVAILABLE = True
except ImportError:
    _TTT_AVAILABLE = False
    logger_import = logging.getLogger(__name__)
    logger_import.warning(
        "trueskillthroughtime no instalado. "
        "Ejecutar: pip install trueskillthroughtime\n"
        "TTT estara deshabilitado; el motor cae en ELO clasico."
    )

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardResult:
    """Results from a single walk-forward split (v1 compatible + v2/v3 extras)"""
    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    top_3_strategies: List[str]
    bottom_3_strategies: List[str]
    top_3_return: float
    bottom_3_return: float
    all_return: float
    spread: float
    elo_accuracy: float
    # v2 additions
    embargo_days: int = 0
    purge_days: int = 0
    top_3_calmar: float = np.nan
    top_3_psr: float = np.nan
    top_3_cvar_5: float = np.nan
    n_effective_trials: int = 1
    # v3 additions — TTT per-strategy scores (populated after global TTT fit)
    # Dict[strategy_name -> mu_at_test_end]
    ttt_mu_at_test: Dict = field(default_factory=dict)
    # Dict[strategy_name -> sigma_at_test_end]
    ttt_sigma_at_test: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# TTT ACCUMULATOR — recolecta partidos OOS en formato logistic.py
# ══════════════════════════════════════════════════════════════════════════════

class TTTAccumulator:
    """
    Acumula partidos OOS en formato nativo de TrueSkill Through Time.

    DISEÑO (por qué global y no por split):
    ─────────────────────────────────────────────────────────────────────
    TTT realiza inferencia Bayesiana GLOBAL sobre toda la historia de
    partidos: el forward pass propaga creencias hacia adelante y el
    backward pass (suavización) las refina hacia atrás usando información
    futura. Llamar a History.convergence() con subconjuntos de splits
    desperdiciaría la suavización y equivaldría a truncar la distribución
    a posteriori artificialmente.

    Por tanto: acumular todos los folds OOS, luego convergencia única.

    MAPEO financiero → partidos TTT con obs="Continuous":
    ─────────────────────────────────────────────────────────────────────
    · "Jugador A" = estrategia; "jugador B" = benchmark (SPY alpha=0)
    · composition[t] = [["estrategia"], ["_benchmark_"]]
    · results[t]  = [strat_fold_ret, bench_fold_ret]
                    retornos totales compuestos del fold OOS:
                    r_fold = prod(1 + r_daily_t) - 1
      TTT infiere habilidad a partir de la MAGNITUD del diferencial,
      no solo de su signo — superior al modelo Ordinal binario.
    · obs[t] = "Continuous" (modelo de observación continua por partido).
    · times[t] = timestamp UNIX en días del PRIMER día del fold OOS
      (post-embargo, garantizando separación IS/OOS temporal).
    · sigma= global en History() — aplica uniformemente a todos los
      jugadores; elimina la necesidad del dict de priors por jugador.
    ─────────────────────────────────────────────────────────────────────

    Ref: Landfried & Mocskos (2024), sec. 3 — logistic.py líneas 14-20.
    """

    def __init__(self, ttt_gamma: float = 0.03, ttt_sigma: float = 1.0):
        """
        Args:
            ttt_gamma: Varianza dinámica del proceso estocástico de habilidad.
                       Controla cuán rápido puede cambiar mu entre períodos.
                       logistic.py usa gamma=0.015; example.py usa gamma=0.03.
                       Para estrategias de trading recomendado: 0.02–0.05.
            ttt_sigma: Sigma inicial de la distribución prior de habilidad
                       de las estrategias (incertidumbre inicial).
                       logistic.py usa sigma=0.2 para priors conocidos;
                       usamos 1.0 para prior no informativo.
        """
        self.ttt_gamma = ttt_gamma
        self.ttt_sigma = ttt_sigma

        # Listas acumuladas — modelo Continuo (obs="Continuous")
        self.composition: List[List[List[str]]] = []  # [[["strat"], ["_bm_"]], ...]
        self.results: List[List[float]] = []           # [[strat_fold_ret, bench_fold_ret]]
        self.times: List[float] = []                   # timestamp en días (float)
        self.obs: List[str] = []                       # "Continuous" por cada partido

        # Registro de estrategias únicas para extraer learning_curves
        self.strategy_names: set = set()

    def add_fold(
        self,
        strategy_name: str,
        strategy_oos_returns: pd.Series,
        benchmark_oos_returns: pd.Series,
        test_start: pd.Timestamp,
    ) -> None:
        """
        Registra un fold OOS como un partido TTT en modo Continuo.

        MODELO DE OBSERVACIÓN (obs="Continuous"):
          result = [strat_fold_ret, bench_fold_ret]
          donde cada escalar es el retorno total compuesto del fold:
            r_fold = prod(1 + r_t) - 1  para t en [test_start, test_end]

          TTT infiere la habilidad latente a partir de la MAGNITUD del
          diferencial de retorno, no solo de su signo. Esto preserva la
          información cuantitativa del Alpha generado en cada período.

        El tiempo `t` es el ordinal en días del primer día del período OOS
        (ya con embargo aplicado), asegurando separación temporal.

        Args:
            strategy_name: Nombre único de la estrategia (jugador A).
            strategy_oos_returns: Retornos diarios OOS de la estrategia.
            benchmark_oos_returns: Retornos diarios OOS del benchmark.
            test_start: Primer día del período OOS (post-embargo).
        """
        # Retorno compuesto total del fold: prod(1+r_t) - 1
        # Preserva la magnitud del P&L real del período; superior a la media
        # aritmética para folds largos (Jensen's inequality).
        strat_ret = float((1.0 + strategy_oos_returns).prod() - 1.0)
        bench_ret = float((1.0 + benchmark_oos_returns).prod() - 1.0)

        # Tiempo en días desde epoch (sintaxis times de logistic.py línea 19)
        t_days = float(test_start.timestamp() / (60 * 60 * 24))

        self.composition.append([[strategy_name], ["_benchmark_"]])
        self.results.append([strat_ret, bench_ret])
        self.obs.append("Continuous")
        self.times.append(t_days)
        self.strategy_names.add(strategy_name)

    def calibrate_ttt_parameters(
        self,
        grid_gamma: Optional[List[float]] = None,
        grid_sigma: Optional[List[float]] = None,
        calib_iterations: int = 3,
        calib_epsilon: float = 0.01,
    ) -> Dict[str, float]:
        """
        Grid search sobre (gamma, sigma) maximizando la Log-Evidencia Marginal.

        CRITERIO DE SELECCIÓN — Bayesian Model Selection:
        h.geometric_mean() calcula exp( mean( log P(result_t | history_{<t}) ) ),
        la media geométrica de la verosimilitud marginal (prior predictive) sobre
        todos los partidos OOS acumulados. Un valor alto indica que el modelo
        asigna alta probabilidad a los resultados reales ANTES de observarlos:
        es el criterio canónico de selección bayesiana de modelos
        (Bernardo & Smith, 1994, "Bayesian Theory", sec. 6.1).

        La calibración usa convergencia relajada (calib_iterations, calib_epsilon)
        para reducir el coste computacional del grid. El ajuste final en fit_ttt()
        usa los parámetros estrictos originales para máximo rigor bayesiano.

        GRIDS POR DEFECTO:
        · grid_gamma: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
          Cubre desde habilidad casi estacionaria (0.01) hasta alta variabilidad
          de régimen (0.10). Landfried (2024) recomienda 0.015–0.036 para ATP.
        · grid_sigma: [0.5, 1.0, 1.5, 2.0]
          Rango de incertidumbre prior: 0.5 (prior informativo) a 2.0 (difuso).

        Args:
            grid_gamma: Lista de valores de gamma a explorar.
            grid_sigma: Lista de valores de sigma a explorar.
            calib_iterations: Iteraciones TTT durante calibración (relajado).
            calib_epsilon:    Tolerancia TTT durante calibración (relajado).

        Returns:
            Dict con 'gamma', 'sigma', 'geometric_mean' del mejor par encontrado,
            y 'all_results' [(gamma, sigma, score), ...] para auditoría.
            Actualiza self.ttt_gamma y self.ttt_sigma in-place.
        """
        if not _TTT_AVAILABLE:
            logger.warning("TTT no disponible — calibración omitida.")
            return {'gamma': self.ttt_gamma, 'sigma': self.ttt_sigma,
                    'geometric_mean': float('nan'), 'all_results': []}

        if not self.composition:
            logger.warning("TTTAccumulator vacío — calibración omitida.")
            return {'gamma': self.ttt_gamma, 'sigma': self.ttt_sigma,
                    'geometric_mean': float('nan'), 'all_results': []}

        if grid_gamma is None:
            grid_gamma = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        if grid_sigma is None:
            grid_sigma = [0.5, 1.0, 1.5, 2.0]

        n_combos = len(grid_gamma) * len(grid_sigma)
        logger.info(
            f"TTT calibración: {n_combos} combinaciones | "
            f"gamma={grid_gamma} | sigma={grid_sigma} | "
            f"calib_iters={calib_iterations} | calib_eps={calib_epsilon}"
        )

        best_score: float = -float('inf')
        best_gamma: float = self.ttt_gamma
        best_sigma: float = self.ttt_sigma
        all_results: List[Tuple[float, float, float]] = []

        for gamma, sigma in product(grid_gamma, grid_sigma):
            try:
                h_calib = History(
                    composition=self.composition,
                    results=self.results,
                    times=self.times,
                    mu=0.0,
                    sigma=sigma,
                    gamma=gamma,
                    obs=self.obs,
                )
                h_calib.convergence(
                    iterations=calib_iterations,
                    epsilon=calib_epsilon,
                    verbose=False,
                )
                score: float = h_calib.geometric_mean()
                all_results.append((gamma, sigma, score))

                if score > best_score:
                    best_score = score
                    best_gamma = gamma
                    best_sigma = sigma

                logger.debug(
                    f"  gamma={gamma:.4f} | sigma={sigma:.4f} | "
                    f"geometric_mean={score:.6f}"
                )

            except Exception as exc:
                logger.debug(
                    f"  gamma={gamma:.4f} | sigma={sigma:.4f} → "
                    f"error durante calibración: {exc}"
                )

        self.ttt_gamma = best_gamma
        self.ttt_sigma = best_sigma

        logger.info(
            f"TTT calibración completada | "
            f"best_gamma={best_gamma:.4f} | best_sigma={best_sigma:.4f} | "
            f"geometric_mean={best_score:.6f}"
        )

        return {
            'gamma': best_gamma,
            'sigma': best_sigma,
            'geometric_mean': best_score,
            'all_results': all_results,
        }

    def fit_ttt(
        self,
        iterations: int = 6,
        epsilon: float = 1e-3,
        verbose: bool = False,
        auto_calibrate: bool = False,
        calib_grid_gamma: Optional[List[float]] = None,
        calib_grid_sigma: Optional[List[float]] = None,
    ) -> Optional["History"]:
        """
        Ajusta el modelo TTT sobre todos los partidos acumulados (obs=Continuous).

        DECISIONES DE DISEÑO:
        · mu=0.0: prior de habilidad cero (alpha neutro frente al benchmark).
        · sigma=: parámetro global del History constructor — más idiomático
          que el dict de priors (ver test_env_0_TTT del repo oficial, línea 370).
          Aplica uniformemente a todos los jugadores, incluyendo "_benchmark_".
        · gamma: varianza dinámica compartida — permite que mu(t) varíe.
        · obs=self.obs: lista de "Continuous" por partido, activando el modelo
          de observación que usa la magnitud del retorno diferencial.
        · convergence() retorna (step, n_iter); se verifica step > epsilon para
          detectar y loguear falta de convergencia antes de usar el modelo.
        · auto_calibrate=True: invoca calibrate_ttt_parameters() antes del ajuste
          final para seleccionar (gamma, sigma) óptimos por Log-Evidencia Marginal.
          La calibración usa convergencia relajada; el ajuste final usa la estricta.

        Args:
            iterations: Máximo de iteraciones de convergencia (ajuste final).
            epsilon: Tolerancia de convergencia (ajuste final).
            verbose: Imprimir progreso de convergencia.
            auto_calibrate: Si True, ejecuta grid search bayesiano antes del fit.
            calib_grid_gamma: Grid de gamma para calibración (None → defecto).
            calib_grid_sigma: Grid de sigma para calibración (None → defecto).

        Returns:
            Objeto History ajustado, o None si TTT no está disponible.
        """
        if not _TTT_AVAILABLE:
            logger.warning("TTT no disponible. Instalar: pip install trueskillthroughtime")
            return None

        if not self.composition:
            logger.warning("TTTAccumulator: no hay partidos acumulados.")
            return None

        if auto_calibrate:
            self.calibrate_ttt_parameters(
                grid_gamma=calib_grid_gamma,
                grid_sigma=calib_grid_sigma,
            )

        logger.info(
            f"TTT fit: {len(self.composition)} partidos | "
            f"{len(self.strategy_names)} estrategias | "
            f"gamma={self.ttt_gamma} | sigma={self.ttt_sigma} | obs=Continuous"
        )

        # sigma= global elimina el dict de priors: idiomático y sin riesgo de
        # omitir estrategias que aparezcan después del primer fit.
        h = History(
            composition=self.composition,
            results=self.results,
            times=self.times,
            mu=0.0,
            sigma=self.ttt_sigma,
            gamma=self.ttt_gamma,
            obs=self.obs,
        )
        step, n_iter = h.convergence(
            iterations=iterations, epsilon=epsilon, verbose=verbose
        )
        if step > epsilon:
            logger.warning(
                f"TTT no convergió en {n_iter} iteraciones "
                f"(step={step:.6f} > epsilon={epsilon:.6f}). "
                f"Considerar aumentar ttt_iterations o reducir epsilon."
            )
        return h

    def extract_learning_curves(
        self, h: "History"
    ) -> Dict[str, pd.DataFrame]:
        """
        Extrae las learning curves de cada estrategia.

        Sintaxis idéntica a logistic.py líneas 24-27:
          lc = h.learning_curves()
          mu = [tp[1].mu for tp in h.learning_curves()["a"]]
          sigma = [tp[1].sigma for tp in h.learning_curves()["a"]]

        Convierte el output TTT (lista de (time, Gaussian)) a DataFrames
        con columnas [time_days, time_date, mu, sigma] por estrategia.

        Returns:
            Dict[strategy_name -> DataFrame con columnas time_date, mu, sigma]
        """
        lc_raw = h.learning_curves()
        curves = {}

        for name in self.strategy_names:
            if name not in lc_raw:
                continue
            # lc_raw[name] = [(t_days, Gaussian), ...]
            # Sintaxis de logistic.py:
            #   mu = [tp[1].mu for tp in h.learning_curves()["a"]]
            #   sigma = [tp[1].sigma for tp in h.learning_curves()["a"]]
            times_d = [tp[0] for tp in lc_raw[name]]
            mus = [tp[1].mu for tp in lc_raw[name]]
            sigmas = [tp[1].sigma for tp in lc_raw[name]]

            curves[name] = pd.DataFrame({
                'time_days': times_d,
                'time_date': pd.to_datetime(
                    [t * 86400 for t in times_d], unit='s', utc=True
                ).tz_localize(None),
                'ttt_mu': mus,
                'ttt_sigma': sigmas,
            })

        return curves


@dataclass
class CPCVResult:
    """Aggregate result from Combinatorial Purged CV"""
    n_splits: int
    n_test_folds: int
    n_combinations: int
    spread_distribution: np.ndarray
    pbo: float
    mean_spread: float
    median_spread: float
    win_rate: float
    t_stat: float
    p_value: float
    dsr: float


# ══════════════════════════════════════════════════════════════════════════════
# METRICAS INSTITUCIONALES (PSR, DSR, Calmar, CVaR)
# ══════════════════════════════════════════════════════════════════════════════

def compute_probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).

    Corrige el Sharpe Ratio por longitud de muestra, asimetría y curtosis,
    produciendo la probabilidad P(SR_true > SR*) bajo distribución no-normal.

    Ref: Bailey & López de Prado (2012), SSRN 1821643. Eq. 7.

    Args:
        returns: Serie de retornos diarios.
        sr_benchmark: Sharpe ratio de referencia SR* (default 0).
        periods_per_year: Frecuencia (252 para diario).

    Returns:
        PSR in [0, 1]: Probabilidad de que el SR verdadero > sr_benchmark.
    """
    n = len(returns)
    if n < 30:
        return np.nan

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-12:
        return np.nan

    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)  # exceso de curtosis

    sr_hat = (mu / sigma) * np.sqrt(periods_per_year)

    # Varianza del SR con corrección por momentos superiores
    # Bailey & López de Prado (2012), Eq. 7
    var_sr = (1.0 / n) * (
        1.0
        + 0.5 * (sr_hat ** 2)
        - skew * sr_hat
        + (kurt / 4.0) * (sr_hat ** 2)
    )

    if var_sr <= 0:
        return np.nan

    z = (sr_hat - sr_benchmark) / np.sqrt(var_sr)
    return float(stats.norm.cdf(z))


def compute_deflated_sharpe_ratio(
    returns_list: List[pd.Series],
    periods_per_year: int = 252
) -> Tuple[float, float]:
    """
    Deflated Sharpe Ratio (DSR).

    Ajusta el PSR del mejor backtest por el numero de estrategias evaluadas,
    previniendo el p-hacking y la seleccion multiple (multiple testing).

    Ref: Bailey & López de Prado (2014), Journal of Portfolio Management
         40(5), 94-107. SSRN: 2460551. Eq. 8-9.

    Args:
        returns_list: Lista de Series de retornos de TODOS los backtests
                      evaluados (incluyendo los descartados). CRITICO: registrar
                      todos para evitar sesgo de seleccion.
        periods_per_year: Frecuencia de datos.

    Returns:
        Tuple (dsr, expected_max_sr):
          - dsr in [0,1]: Probabilidad de que el mejor SR seleccionado sea real.
          - expected_max_sr: SR maximo esperado por azar con N trials.
    """
    n_trials = len(returns_list)
    if n_trials == 0:
        return np.nan, np.nan

    sharpe_ratios = []
    for ret in returns_list:
        if len(ret) < 10 or ret.std() < 1e-12:
            continue
        sr = (ret.mean() / ret.std(ddof=1)) * np.sqrt(periods_per_year)
        sharpe_ratios.append(sr)

    if not sharpe_ratios:
        return np.nan, np.nan

    n = len(sharpe_ratios)
    best_sr = max(sharpe_ratios)

    # E[max(SR)] bajo H0 — Bailey & López de Prado (2014), Eq. 8
    # Constante de Euler-Mascheroni: gamma ~= 0.5772
    euler_mascheroni = 0.5772156649
    if n > 1:
        expected_max_sr = (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n)
            + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n * np.e))
        )
    else:
        expected_max_sr = 0.0

    best_idx = sharpe_ratios.index(best_sr)
    best_returns = returns_list[best_idx]

    dsr = compute_probabilistic_sharpe_ratio(
        best_returns,
        sr_benchmark=max(expected_max_sr, 0.0),
        periods_per_year=periods_per_year
    )

    return (float(dsr) if dsr is not None else np.nan), expected_max_sr


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|.

    Fitness function que penaliza el riesgo de cola explicitamente.
    Sustituye la rentabilidad media para evitar sobreajuste a periodos
    de baja volatilidad sin drawdowns.

    Args:
        returns: Serie de retornos diarios.
        periods_per_year: Frecuencia.

    Returns:
        Calmar ratio. NaN si drawdown es 0 o datos insuficientes.
    """
    if len(returns) < 20:
        return np.nan

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return np.nan
    cagr = (1 + total_return) ** (1 / n_years) - 1

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()  # valor negativo

    if abs(max_dd) < 1e-10:
        return np.nan

    return float(cagr / abs(max_dd))


def compute_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall.

    CVaR = E[R | R < VaR_{alpha}]: perdida esperada en el peor (1-alpha)%.

    Ref: Rockafellar & Uryasev (2000). Journal of Risk 2(3), 21-41.

    Args:
        returns: Serie de retornos diarios.
        confidence_level: Nivel alpha (0.95 para CVaR al 5%).

    Returns:
        CVaR (negativo = perdida esperada en escenario de cola).
    """
    if len(returns) < 20:
        return np.nan

    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return float(var_threshold)

    return float(tail_returns.mean())


# ══════════════════════════════════════════════════════════════════════════════
# PURGING & EMBARGO — Lopez de Prado (2018), Cap. 7
# ══════════════════════════════════════════════════════════════════════════════

def apply_purging_and_embargo(
    all_dates: pd.DatetimeIndex,
    train_end_idx: int,
    test_start_idx: int,
    test_end_idx: int,
    purge_days: int = 5,
    embargo_days: int = 10
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Implementa Purging y Embargo — López de Prado (2018), Cap. 7.

    PURGING: Elimina del training set las observaciones mas cercanas al
    inicio del test period. Las etiquetas de retorno calculadas con ventanas
    moviles crean solapamiento temporal que actua como canal de fuga IS->OOS.

    EMBARGO: Buffer temporal adicional para absorber correlacion serial
    residual de microestructura (bid-ask bounce, momentum de corto plazo).
    Regla empirica: embargo >= 0.01 * T (Lopez de Prado, 2018, Cap. 7).

    Args:
        all_dates: Indice completo de fechas.
        train_end_idx: Ultimo indice de entrenamiento (antes de purge).
        test_start_idx: Primer indice de test.
        test_end_idx: Ultimo indice de test.
        purge_days: Dias a eliminar del FINAL del training set.
        embargo_days: Dias adicionales de buffer (embargo).

    Returns:
        (train_dates_purged, test_dates)
    """
    total_buffer = purge_days + embargo_days
    purged_train_end_idx = max(0, train_end_idx - total_buffer)

    train_dates = all_dates[:purged_train_end_idx + 1]
    test_dates = all_dates[test_start_idx:test_end_idx + 1]

    if len(train_dates) == 0:
        logger.warning(
            f"Purging elimino todo el training set. "
            f"Reducir purge_days={purge_days} o embargo_days={embargo_days}."
        )

    return train_dates, test_dates


# ══════════════════════════════════════════════════════════════════════════════
# CPCV: Combinatorial Purged Cross-Validation Generator
# Lopez de Prado (2018), Cap. 12
# ══════════════════════════════════════════════════════════════════════════════

def generate_cpcv_splits(
    all_dates: pd.DatetimeIndex,
    n_folds: int = 6,
    n_test_folds: int = 2,
    purge_days: int = 5,
    embargo_days: int = 10
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, int, int]]:
    """
    Genera splits de Combinatorial Purged Cross-Validation (CPCV).

    Divide el dataset en N folds secuenciales y crea todas las C(N, k)
    combinaciones de k folds de test. Cada combinacion tiene purging aplicado
    en todas las fronteras train-test para eliminar data leakage.

    El numero C(N,k) de combinaciones produce una DISTRIBUCION de performance
    OOS, reduciendo drasticamente el PBO frente al walk-forward single-path.

    Ref: López de Prado (2018), Cap. 12, Algorithm 12.1.
         ScienceDirect (2024): CPCV supera a WF en PBO y DSR.

    Args:
        all_dates: Indice completo de fechas de mercado.
        n_folds: N — numero de grupos secuenciales (recomendado: 6-10).
        n_test_folds: k — folds usados como test (recomendado: 2).
        purge_days: Dias de purging en fronteras.
        embargo_days: Dias de embargo en fronteras.

    Returns:
        Lista de (train_dates, test_dates, combo_idx, total_combos).
    """
    total_days = len(all_dates)
    fold_size = total_days // n_folds

    if fold_size < 30:
        raise ValueError(
            f"Folds demasiado pequenos ({fold_size} dias). "
            f"Reducir n_folds={n_folds} o usar mas datos historicos."
        )

    fold_boundaries = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else total_days
        fold_boundaries.append((start, end))

    all_combinations = list(combinations(range(n_folds), n_test_folds))
    total_combos = len(all_combinations)

    splits = []
    for combo_idx, test_fold_indices in enumerate(all_combinations):
        test_fold_set = set(test_fold_indices)
        train_fold_indices = [i for i in range(n_folds) if i not in test_fold_set]

        train_idx_ranges = [fold_boundaries[i] for i in sorted(train_fold_indices)]
        test_idx_ranges = [fold_boundaries[i] for i in sorted(test_fold_indices)]

        train_indices = []
        for start, end in train_idx_ranges:
            train_indices.extend(range(start, end))

        test_indices = []
        for start, end in test_idx_ranges:
            test_indices.extend(range(start, end))

        # Purging en cada frontera train-test
        purged_train_indices = set(train_indices)
        for t_start, t_end in test_idx_ranges:
            purge_buffer = set(range(
                max(0, t_start - purge_days - embargo_days),
                t_start
            ))
            purged_train_indices -= purge_buffer

        final_train_indices = sorted(purged_train_indices)
        final_test_indices = sorted(test_indices)

        if len(final_train_indices) < 60 or len(final_test_indices) < 10:
            logger.warning(
                f"CPCV combo {combo_idx}: train={len(final_train_indices)}d, "
                f"test={len(final_test_indices)}d — skipping (muy pequeno)."
            )
            continue

        train_dates = all_dates[final_train_indices]
        test_dates = all_dates[final_test_indices]
        splits.append((train_dates, test_dates, combo_idx, total_combos))

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL v2.0
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardELOEngine:
    """
    Walk-Forward ELO Engine v3.0 — Institutional Grade + TrueSkill Through Time.

    Mejoras sobre v2.0:
    8. TTT (Landfried & Mocskos, 2024): inferencia Bayesiana dinámica global
       sobre la habilidad latente de cada estrategia usando todos los folds OOS.
       El motor acumula partidos en formato logistic.py y ejecuta
       History.convergence() una sola vez al final para aprovechar la
       suavización hacia atrás (backward pass) completa de TTT.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_days: int = 252,
        test_days: int = 63,
        purge_days: int = 5,
        embargo_days: int = 10,
        use_cpcv: bool = False,
        cpcv_n_folds: int = 6,
        cpcv_n_test_folds: int = 2,
        fitness_function: str = 'calmar',
        ttt_gamma: float = 0.03,
        ttt_sigma: float = 1.0,
        ttt_iterations: int = 6,
        ttt_auto_calibrate: bool = False,
        ranking_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            n_splits: Numero de splits walk-forward (ignorado si use_cpcv=True).
            min_train_days: Minimo de dias de entrenamiento.
            test_days: Longitud del periodo de test en dias.
            purge_days: Dias de purging en la frontera train/test.
            embargo_days: Dias de embargo adicional al purging.
            use_cpcv: Si True, usa CPCV en lugar de WF estandar.
            cpcv_n_folds: N folds para CPCV (rec: 6-10).
            cpcv_n_test_folds: k folds de test para CPCV (rec: 2).
            fitness_function: 'calmar' | 'sharpe' | 'return'
            ttt_gamma: Varianza dinamica del proceso de habilidad TTT.
                       logistic.py usa 0.015; example.py usa 0.03.
                       Para estrategias de trading: 0.02-0.05.
                       (Landfried & Mocskos, 2024)
            ttt_sigma: Sigma del prior de habilidad de las estrategias.
                       logistic.py usa 0.2 para priors conocidos;
                       1.0 es razonable como prior no informativo.
            ttt_iterations: Iteraciones maximas de convergencia TTT.
                            logistic.py usa 1 (rapido); 6 es preciso.
            ttt_auto_calibrate: Si True, ejecuta grid search sobre (gamma, sigma)
                                maximizando h.geometric_mean() antes del fit final.
                                Recomendado cuando se tienen >20 partidos OOS acumulados.
            ranking_kwargs: Parametros para BayesianELORanking.
        """
        self.n_splits = n_splits
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.use_cpcv = use_cpcv
        self.cpcv_n_folds = cpcv_n_folds
        self.cpcv_n_test_folds = cpcv_n_test_folds
        self.fitness_function = fitness_function
        self.ttt_gamma = ttt_gamma
        self.ttt_sigma = ttt_sigma
        self.ttt_iterations = ttt_iterations
        self.ttt_auto_calibrate = ttt_auto_calibrate

        self.ranking_kwargs = ranking_kwargs or {
            'initial_mu': 1500,
            'initial_sigma': 350,
            'min_sigma': 50,
            'tau': 1.0,
            'base_k': 32
        }

        self.data_manager = DataManager(cache_dir='./data_cache')
        self.backtest_engine = BacktestEngine(benchmark_symbol='SPY')
        self.strategy_registry = StrategyRegistry().create_default_universe()

        self.results: List[WalkForwardResult] = []
        # Registro de TODOS los backtests IS para DSR honesto
        self._all_backtest_returns: List[pd.Series] = []

        # v3: TTT Accumulator — recolecta partidos OOS en formato logistic.py
        self._ttt = TTTAccumulator(
            ttt_gamma=ttt_gamma,
            ttt_sigma=ttt_sigma
        )
        # benchmark returns OOS por split para TTT add_fold
        self._oos_benchmark_by_split: Dict[int, pd.Series] = {}

        logger.info(
            f"WalkForwardELOEngine v3.0 | "
            f"purge={purge_days}d, embargo={embargo_days}d, "
            f"fitness={fitness_function}, CPCV={use_cpcv}, "
            f"TTT gamma={ttt_gamma}, sigma={ttt_sigma}"
        )

    # ── METODO PRINCIPAL ───────────────────────────────────────────────────────

    def run_walk_forward_validation(
        self,
        symbol: str = 'SPY',
        start_date: str = '1997-01-01',
        end_date: str = '2024-01-01',
        initial_capital: float = 100_000
    ) -> pd.DataFrame:
        """
        Ejecuta la validacion walk-forward completa con mejoras institucionales.

        Returns:
            DataFrame con metricas de cada split (incluye PSR, Calmar, CVaR).
        """
        print("=" * 100)
        print("WALK-FORWARD ELO VALIDATION v3.0 — INSTITUTIONAL GRADE + TTT")
        print(f"  Purge={self.purge_days}d | Embargo={self.embargo_days}d | "
              f"Fitness={self.fitness_function} | CPCV={self.use_cpcv} | "
              f"TTT gamma={self.ttt_gamma}")
        print("=" * 100)

        self.adapted_strategies = adapt_strategies(
            list(self.strategy_registry.strategies.values()),
            capital=initial_capital
        )

        print(f"\n[Step 1] Cargando datos de mercado para {symbol}...")
        market_data = self.data_manager.fetch_data(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
        sanity_check_market_data(
            market_data, require_cols=['close', 'volume'], nan_threshold=0.05
        )
        benchmark_data = market_data.copy()
        all_dates = market_data.index

        print(f"  Trading days: {len(all_dates)} | "
              f"Rango: {all_dates[0].date()} -> {all_dates[-1].date()}")

        # Advertencia si embargo_days es inferior al minimo recomendado
        recommended_embargo = max(1, int(0.01 * self.min_train_days))
        if self.embargo_days < recommended_embargo:
            logger.warning(
                f"embargo_days={self.embargo_days} < {recommended_embargo} "
                f"(1% del periodo de entrenamiento). "
                f"Riesgo de correlacion serial residual sin purgar. "
                f"Ref: Lopez de Prado (2018), Cap. 7."
            )

        if self.use_cpcv:
            n_combos = comb(self.cpcv_n_folds, self.cpcv_n_test_folds)
            print(f"\n[Step 2] Modo CPCV: C({self.cpcv_n_folds},{self.cpcv_n_test_folds})"
                  f" = {n_combos} combinaciones...")
            splits = generate_cpcv_splits(
                all_dates,
                n_folds=self.cpcv_n_folds,
                n_test_folds=self.cpcv_n_test_folds,
                purge_days=self.purge_days,
                embargo_days=self.embargo_days
            )
        else:
            print(f"\n[Step 2] Modo Walk-Forward con Purge+Embargo...")
            splits = self._calculate_wf_splits_with_purge(all_dates)

        print(f"  {len(splits)} splits validos generados.")

        for split_pack in splits:
            if self.use_cpcv:
                train_dates, test_dates, combo_idx, total_combos = split_pack
                split_idx = combo_idx
                print(f"\n=== CPCV COMBO {combo_idx + 1}/{total_combos} ===")
            else:
                train_dates, test_dates, split_idx = split_pack
                print(f"\n=== SPLIT {split_idx + 1}/{len(splits)} ===")

            print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()} "
                  f"({len(train_dates)}d | purge={self.purge_days}d+emb={self.embargo_days}d)")
            print(f"  Test:  {test_dates[0].date()} -> {test_dates[-1].date()} "
                  f"({len(test_dates)}d)")

            result = self._run_single_split(
                split_idx=split_idx,
                train_dates=train_dates,
                test_dates=test_dates,
                market_data=market_data,
                benchmark_data=benchmark_data,
                initial_capital=initial_capital
            )
            self.results.append(result)

            print(f"  Top-3: {result.top_3_strategies}")
            print(f"  OOS -> Spread={result.spread:.2%} | "
                  f"Calmar={result.top_3_calmar:.3f} | "
                  f"PSR={result.top_3_psr:.3f} | "
                  f"CVaR(5%)={result.top_3_cvar_5:.3%} | "
                  f"ELO Acc={result.elo_accuracy:.1%}")

        return self._aggregate_results()

    # ── SPLITS CON PURGE+EMBARGO ───────────────────────────────────────────────

    def _calculate_wf_splits_with_purge(
        self, all_dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, int]]:
        """
        Genera splits walk-forward (expanding window) con Purging y Embargo.

        Lopez de Prado (2018), Cap. 7: elimina `purge_days + embargo_days`
        dias del final del training set creando un buffer limpio ante el OOS.

        Returns:
            Lista de (train_dates_purged, test_dates, split_idx).
        """
        total_days = len(all_dates)
        min_train = max(self.min_train_days, 2 * 252)
        remaining = total_days - min_train
        n_possible = min(self.n_splits, remaining // self.test_days)

        if n_possible < 2:
            raise ValueError(
                f"Datos insuficientes. Necesario: >={min_train + 2 * self.test_days}d. "
                f"Disponible: {total_days}d."
            )

        splits = []
        for i in range(n_possible):
            raw_train_end_idx = min_train + i * self.test_days - 1
            test_start_idx = raw_train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_days - 1, total_days - 1)

            if test_end_idx <= test_start_idx:
                break

            train_dates, test_dates = apply_purging_and_embargo(
                all_dates=all_dates,
                train_end_idx=raw_train_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
                purge_days=self.purge_days,
                embargo_days=self.embargo_days
            )

            if len(train_dates) >= self.min_train_days and len(test_dates) >= 10:
                splits.append((train_dates, test_dates, i))

        return splits

    # ── EJECUCION DE UN SPLIT ─────────────────────────────────────────────────

    def _run_single_split(
        self,
        split_idx: int,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        market_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        initial_capital: float
    ) -> WalkForwardResult:
        """Ejecuta un unico split IS/OOS con metricas institucionales v2.0."""

        # FASE IS: entrenamiento ELO
        print("  -> Entrenando ELO (IS, purged)...")
        train_market = market_data.loc[market_data.index.isin(train_dates)]
        train_bench = benchmark_data.loc[benchmark_data.index.isin(train_dates)]

        regimes = self._detect_regimes_no_lookahead(train_market)
        train_results = self._run_backtests_on_period(
            train_market, train_bench, initial_capital
        )

        # Registrar TODOS los backtests IS para el DSR global
        for name, res in train_results.items():
            series = getattr(res, 'alpha_returns', None)
            if series is None:
                series = res.returns
            if series is not None and len(series) > 10:
                self._all_backtest_returns.append(series.copy())

        ranking_manager = self._train_elo_ranking(
            train_results, regimes, train_dates,
            fitness=self.fitness_function
        )
        train_leaderboard = ranking_manager.get_leaderboard('alpha', None)

        if train_leaderboard.empty:
            raise RuntimeError(
                f"No se generaron rankings ELO para el split {split_idx}."
            )

        top_3 = train_leaderboard.head(3)['strategy'].tolist()
        bottom_3 = train_leaderboard.tail(3)['strategy'].tolist()

        # FASE OOS: evaluacion
        print("  -> Evaluando en OOS...")
        test_market = market_data.loc[market_data.index.isin(test_dates)]
        test_bench = benchmark_data.loc[benchmark_data.index.isin(test_dates)]
        test_results = self._run_backtests_on_period(
            test_market, test_bench, initial_capital
        )

        # Calcular retornos OOS por estrategia
        test_returns_dict = {}
        for name, result in test_results.items():
            series = getattr(result, 'alpha_returns', None) or result.returns
            series = series.reindex(test_market.index).fillna(0)
            test_returns_dict[name] = series

        def _ann(s): return s.mean() * 252

        top_3_return = float(np.mean(
            [_ann(test_returns_dict.get(s, pd.Series([0]))) for s in top_3]
        ))
        bottom_3_return = float(np.mean(
            [_ann(test_returns_dict.get(s, pd.Series([0]))) for s in bottom_3]
        ))
        all_return = float(np.mean([_ann(v) for v in test_returns_dict.values()]))
        spread = top_3_return - bottom_3_return

        elo_accuracy = self._calculate_elo_accuracy(
            test_results, top_3, bottom_3, test_dates
        )

        # METRICAS INSTITUCIONALES v2.0
        valid_top3 = [s for s in top_3 if s in test_returns_dict]
        if valid_top3:
            top_3_combined = pd.concat(
                [test_returns_dict[s] for s in valid_top3], axis=1
            ).mean(axis=1)
        else:
            top_3_combined = pd.Series([0.0])

        calmar = compute_calmar_ratio(top_3_combined)
        psr = compute_probabilistic_sharpe_ratio(top_3_combined, sr_benchmark=0.0)
        cvar = compute_cvar(top_3_combined, confidence_level=0.95)

        # v3: ALIMENTAR TTT ACCUMULATOR con partidos OOS
        # Cada estrategia juega contra el benchmark en este fold.
        # Sintaxis de resultado: [1.,0.] si estrategia gana, [0.,1.] si pierde.
        # Idéntico a logistic.py líneas 17-18:
        #   results = [[1.,0.] if normal(target[i]) > normal(opponents[i]) else [0.,1.] ...]
        # El tiempo del partido = timestamp del primer día OOS (post-embargo).
        bench_oos = test_market['close'].pct_change().dropna()
        bench_oos = bench_oos.reindex(test_market.index).fillna(0)
        self._oos_benchmark_by_split[split_idx] = bench_oos

        for strat_name, strat_returns in test_returns_dict.items():
            self._ttt.add_fold(
                strategy_name=strat_name,
                strategy_oos_returns=strat_returns,
                benchmark_oos_returns=bench_oos,
                test_start=test_dates[0],
            )

        return WalkForwardResult(
            split_idx=split_idx,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            top_3_strategies=top_3,
            bottom_3_strategies=bottom_3,
            top_3_return=top_3_return,
            bottom_3_return=bottom_3_return,
            all_return=all_return,
            spread=spread,
            elo_accuracy=elo_accuracy,
            embargo_days=self.embargo_days,
            purge_days=self.purge_days,
            top_3_calmar=calmar,
            top_3_psr=psr,
            top_3_cvar_5=cvar,
            n_effective_trials=len(self.adapted_strategies)
        )

    # ── DETECCION DE REGIMEN ─────────────────────────────────────────────────

    def _detect_regimes_no_lookahead(self, data: pd.DataFrame) -> pd.Series:
        """Detecta regimenes usando UNICAMENTE los datos proporcionados."""
        try:
            vol_detector = VolatilityRegimeDetector(
                window=20, num_regimes=2, method='quantile'
            )
            regimes = vol_detector.detect_regimes(data)
            regimes = vol_detector.expand_to_full_index(
                regimes, data.index, method='ffill'
            )
            return regimes
        except Exception as e:
            logger.warning(f"Deteccion de regimen fallida: {e}. Usando regimen unico.")
            return pd.Series(index=data.index, data=[0] * len(data), name='regime')

    # ── BACKTESTS POR PERIODO ─────────────────────────────────────────────────

    def _run_backtests_on_period(
        self,
        market_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, BacktestResult]:
        """Ejecuta backtests de todas las estrategias en un periodo especifico."""
        results = {}
        for strategy in self.adapted_strategies:
            try:
                result = self.backtest_engine.run_backtest(
                    strategy,
                    market_data,
                    initial_capital=initial_capital,
                    benchmark_data=benchmark_data
                )
                result.returns = result.returns.reindex(market_data.index).fillna(0)
                if getattr(result, 'alpha_returns', None) is not None:
                    result.alpha_returns = (
                        result.alpha_returns.reindex(market_data.index).fillna(0)
                    )
                results[strategy.name] = result

            except Exception as e:
                logger.warning(f"Backtest fallido para {strategy.name}: {e}")
                dummy = pd.Series(0, index=market_data.index)
                results[strategy.name] = BacktestResult(
                    returns=dummy,
                    positions=dummy,
                    trades=pd.DataFrame(),
                    equity_curve=pd.Series(initial_capital, index=market_data.index),
                    metrics={},
                    alpha_returns=dummy,
                    beta=0.0,
                    tail_metrics={}
                )
        return results

    # ── ENTRENAMIENTO ELO CON FITNESS FUNCTION INSTITUCIONAL ─────────────────

    def _train_elo_ranking(
        self,
        results: Dict[str, BacktestResult],
        regimes: pd.Series,
        dates: pd.DatetimeIndex,
        fitness: str = 'calmar'
    ) -> RankingManager:
        """
        Entrena el sistema ELO con la funcion de fitness seleccionada.

        fitness='calmar': Los matchups se puntuan por Calmar rolling (63d),
                         penalizando estrategias con drawdowns extremos.
        fitness='sharpe': Sharpe ratio rolling (63d).
        fitness='return': Retorno diario simple (v1.0, NO recomendado).

        La ponderacion por Calmar vs. retorno diario es critica para evitar
        el sobreajuste a estrategias high-return/high-drawdown.
        """
        ranking_manager = RankingManager(
            ranking_class=BayesianELORanking,
            **self.ranking_kwargs
        )

        strategy_names = list(results.keys())
        pairs = list(combinations(strategy_names, 2))
        fitness_scores = self._compute_fitness_scores(results, dates, fitness)

        for date in dates:
            regime_label = None
            try:
                if date in regimes.index:
                    regime_label = int(regimes.loc[date])
            except Exception:
                pass

            daily_scores = {
                name: fitness_scores.get(name, {}).get(date, np.nan)
                for name in strategy_names
            }

            for s1, s2 in pairs:
                sc1 = daily_scores.get(s1, np.nan)
                sc2 = daily_scores.get(s2, np.nan)
                if np.isnan(sc1) or np.isnan(sc2):
                    continue

                eps = 1e-12
                if sc1 > sc2 + eps:
                    outcome = 1.0
                elif sc2 > sc1 + eps:
                    outcome = 0.0
                else:
                    outcome = 0.5

                for regime in [regime_label, None]:
                    ranking_manager.update(
                        strategy_a_name=s1,
                        strategy_b_name=s2,
                        outcome=outcome,
                        metric='alpha',
                        regime=regime,
                        timestamp=pd.Timestamp(date)
                    )

        return ranking_manager

    def _compute_fitness_scores(
        self,
        results: Dict[str, BacktestResult],
        dates: pd.DatetimeIndex,
        fitness: str
    ) -> Dict[str, Dict]:
        """
        Pre-computa scores de fitness por estrategia y por fecha.

        Para 'calmar' y 'sharpe', usa ventanas rolling de 63 dias (1 quarter)
        para capturar el perfil riesgo/retorno reciente sin look-ahead.
        Implementacion vectorizada con Pandas para eficiencia.
        """
        fitness_dict = {}
        window = 63  # 1 quarter rolling

        for name, result in results.items():
            series = getattr(result, 'alpha_returns', None)
            if series is None:
                series = result.returns
            series = series.reindex(dates).fillna(0)

            scores_per_date = {}

            if fitness == 'return':
                # Retorno diario simple — v1.0 (NO recomendado)
                for date in dates:
                    try:
                        scores_per_date[date] = float(series.loc[date])
                    except Exception:
                        scores_per_date[date] = np.nan

            elif fitness == 'sharpe':
                # Sharpe rolling 63d (vectorizado)
                rolling_mean = series.rolling(window, min_periods=20).mean()
                rolling_std = series.rolling(window, min_periods=20).std()
                rolling_sharpe = (
                    rolling_mean / rolling_std.clip(lower=1e-12)
                ) * np.sqrt(252)
                scores_per_date = rolling_sharpe.to_dict()

            elif fitness == 'calmar':
                # Calmar rolling 63d = CAGR_anualizado / |Max_DD| (vectorizado)
                cum_ret = (1 + series).cumprod()
                rolling_max = cum_ret.rolling(window, min_periods=20).max()
                drawdown = (cum_ret - rolling_max) / rolling_max.clip(lower=1e-12)
                rolling_max_dd = drawdown.rolling(window, min_periods=20).min()
                rolling_ann_ret = series.rolling(window, min_periods=20).mean() * 252
                rolling_calmar = (
                    rolling_ann_ret / rolling_max_dd.abs().clip(lower=1e-10)
                )
                scores_per_date = rolling_calmar.to_dict()

            else:
                raise ValueError(
                    f"fitness_function='{fitness}' no reconocido. "
                    f"Usar 'calmar', 'sharpe' o 'return'."
                )

            fitness_dict[name] = scores_per_date

        return fitness_dict

    # ── ELO ACCURACY ──────────────────────────────────────────────────────────

    def _calculate_elo_accuracy(
        self,
        test_results: Dict[str, BacktestResult],
        top_3: List[str],
        bottom_3: List[str],
        test_dates: pd.DatetimeIndex
    ) -> float:
        """Tasa de acierto diaria del ranking ELO en OOS."""
        if not top_3 or not bottom_3:
            return 0.0

        wins = 0
        total = 0

        for date in test_dates:
            daily_returns = {}
            for name, result in test_results.items():
                series = getattr(result, 'alpha_returns', None) or result.returns
                try:
                    daily_returns[name] = float(series.loc[date])
                except Exception:
                    daily_returns[name] = 0.0

            top_avg = np.mean([daily_returns.get(s, 0.0) for s in top_3])
            bot_avg = np.mean([daily_returns.get(s, 0.0) for s in bottom_3])

            if top_avg > bot_avg:
                wins += 1
            total += 1

        return wins / total if total > 0 else 0.0

    # ── AGREGACION DE RESULTADOS CON DSR Y PBO ────────────────────────────────

    def _aggregate_results(self) -> pd.DataFrame:
        """
        Agrega resultados de todos los splits y calcula:
        - Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014)
        - Probability of Backtest Overfitting (PBO)
        - Metricas de riesgo agregadas (Calmar, PSR, CVaR)
        """
        if not self.results:
            return pd.DataFrame()

        results_data = []
        for r in self.results:
            results_data.append({
                'split': r.split_idx + 1,
                'train_start': r.train_start,
                'train_end': r.train_end,
                'test_start': r.test_start,
                'test_end': r.test_end,
                'train_days': (r.train_end - r.train_start).days,
                'test_days': (r.test_end - r.test_start).days,
                'purge_days': r.purge_days,
                'embargo_days': r.embargo_days,
                'top_3_strategies': ', '.join(r.top_3_strategies),
                'bottom_3_strategies': ', '.join(r.bottom_3_strategies),
                'top_3_return': r.top_3_return,
                'bottom_3_return': r.bottom_3_return,
                'all_return': r.all_return,
                'spread': r.spread,
                'elo_accuracy': r.elo_accuracy,
                'top_3_calmar': r.top_3_calmar,
                'top_3_psr': r.top_3_psr,
                'top_3_cvar_5': r.top_3_cvar_5,
            })

        df = pd.DataFrame(results_data)

        # Deflated Sharpe Ratio global (todos los backtests IS)
        dsr, expected_max_sr = compute_deflated_sharpe_ratio(
            self._all_backtest_returns
        )

        # Probability of Backtest Overfitting (PBO)
        n_total = len(df)
        n_overfit = (df['spread'] <= 0).sum()
        pbo = n_overfit / n_total if n_total > 0 else np.nan

        # ── v3: TTT INFERENCIA GLOBAL ─────────────────────────────────────────
        # Ajustar History UNA SOLA VEZ con todos los partidos OOS acumulados.
        # Sintaxis directa de logistic.py líneas 21-27:
        #   h = History(composition, results, times, priors, mu=2.0, gamma=0.015)
        #   h.convergence()
        #   lc = h.learning_curves()
        #   mu = [tp[1].mu for tp in lc["a"]]
        #   sigma = [tp[1].sigma for tp in lc["a"]]
        ttt_curves: Dict[str, pd.DataFrame] = {}
        h_ttt = None

        if _TTT_AVAILABLE and len(self._ttt.composition) > 0:
            print(f"\n[TTT] Ajustando History con {len(self._ttt.composition)} "
                  f"partidos OOS ({len(self._ttt.strategy_names)} estrategias)...")
            h_ttt = self._ttt.fit_ttt(
                iterations=self.ttt_iterations,
                epsilon=1e-3,
                verbose=False,
                auto_calibrate=self.ttt_auto_calibrate,
            )
            if h_ttt is not None:
                ttt_curves = self._ttt.extract_learning_curves(h_ttt)
                print(f"  TTT convergido | curvas extraidas para "
                      f"{len(ttt_curves)} estrategias.")

                # Enriquecer el DataFrame de resultados con ttt_mu y ttt_sigma
                # para cada split: buscamos el mu/sigma más cercano al test_start
                for col in ['ttt_top3_mu_mean', 'ttt_top3_sigma_mean',
                            'ttt_top3_mu_end', 'ttt_benchmark_mu']:
                    df[col] = np.nan

                for idx, row in df.iterrows():
                    split_res = self.results[idx]
                    test_start_ts = pd.Timestamp(row['test_start'])

                    # mu medio del Top-3 en este fold
                    top3_mus = []
                    top3_sigmas = []
                    for strat in split_res.top_3_strategies:
                        if strat not in ttt_curves:
                            continue
                        curve = ttt_curves[strat]
                        # Punto más cercano al test_start del fold
                        closest = (curve['time_date'] - test_start_ts).abs().idxmin()
                        top3_mus.append(curve.loc[closest, 'ttt_mu'])
                        top3_sigmas.append(curve.loc[closest, 'ttt_sigma'])

                    if top3_mus:
                        df.at[idx, 'ttt_top3_mu_mean'] = float(np.mean(top3_mus))
                        df.at[idx, 'ttt_top3_sigma_mean'] = float(np.mean(top3_sigmas))

                    # mu del benchmark TTT en este fold
                    lc_raw = h_ttt.learning_curves()
                    if '_benchmark_' in lc_raw:
                        bm_curve_raw = lc_raw['_benchmark_']
                        # Tiempo del fold en días
                        t_fold = float(
                            test_start_ts.timestamp() / (60 * 60 * 24)
                        )
                        # Punto más cercano
                        diffs = [abs(tp[0] - t_fold) for tp in bm_curve_raw]
                        ci = int(np.argmin(diffs))
                        df.at[idx, 'ttt_benchmark_mu'] = bm_curve_raw[ci][1].mu

                # Guardar las learning curves completas en un CSV por estrategia
                self.ttt_curves_ = ttt_curves
            else:
                self.ttt_curves_ = {}
        else:
            self.ttt_curves_ = {}
            if not _TTT_AVAILABLE:
                print("\n[TTT] Modulo no disponible — instalar: "
                      "pip install trueskillthroughtime")
        # ─────────────────────────────────────────────────────────────────────

        print("\n" + "=" * 100)
        print("RESULTADOS DE VALIDACION WALK-FORWARD v3.0 — INSTITUTIONAL GRADE + TTT")
        print("=" * 100)

        print(f"\n[Prediccion ELO]")
        print(f"  Spread medio (Top-3 vs Bottom-3): {df['spread'].mean():.2%}")
        print(f"  ELO Accuracy media:               {df['elo_accuracy'].mean():.1%}")
        print(f"  Splits con spread > 0:            {(df['spread'] > 0).sum()}/{n_total}")
        print(f"  ELO Accuracy > 50%:               {(df['elo_accuracy'] > 0.5).sum()}/{n_total}")

        print(f"\n[Metricas de Riesgo OOS (v2.0)]")
        print(f"  Calmar ratio medio (Top-3):   {df['top_3_calmar'].mean():.3f}")
        print(f"  PSR medio (Top-3):            {df['top_3_psr'].mean():.3f}  "
              f"[Bailey & LdP 2012]")
        print(f"  CVaR(5%) medio (Top-3):       {df['top_3_cvar_5'].mean():.3%}")

        print(f"\n[Anti-Overfitting (v2.0)]")
        print(f"  Deflated Sharpe Ratio (DSR):  {dsr:.3f}  "
              f"[Bailey & LdP 2014 | N={len(self._all_backtest_returns)} trials]")
        print(f"  E[max SR] por azar:           {expected_max_sr:.3f}  "
              f"({len(self._all_backtest_returns)} trials evaluados)")
        print(f"  Prob. Backtest Overfitting:   {pbo:.1%}  "
              f"[{n_overfit}/{n_total} splits con spread <= 0]")

        if dsr > 0.95:
            print("\n  ALPHA SIGNIFICATIVO: DSR > 0.95 — Supera el benchmark de "
                  "seleccion multiple.")
        elif dsr > 0.90:
            print("\n  ALPHA MARGINAL: DSR in [0.90, 0.95] — Requiere mas periodos OOS.")
        else:
            print("\n  NO SE RECHAZA H0: DSR < 0.90 — Revisar diseno de estrategias.")

        if pbo < 0.10:
            print("  PBO < 10%: Baja probabilidad de overfitting. ACEPTABLE.")
        elif pbo < 0.25:
            print("  PBO in [10%, 25%]: Riesgo moderado. Considerar CPCV.")
        else:
            print("  PBO > 25%: ALTO RIESGO de overfitting. "
                  "Usar CPCV y reducir el espacio de parametros.")

        all_top = []
        for s in df['top_3_strategies']:
            all_top.extend(s.split(', '))
        freq = pd.Series(all_top).value_counts()
        print(f"\n[Estabilidad del Ranking — Top-5 estrategias en Top-3]")
        for strat, count in freq.head(5).items():
            pct = count / n_total
            tag = "ESTABLE" if pct > 0.5 else "INESTABLE"
            print(f"  {strat}: {count}/{n_total} ({pct:.0%}) — {tag}")

        # v3: Resumen TTT
        if ttt_curves:
            print(f"\n[TTT — TrueSkill Through Time | Landfried & Mocskos (2024)]")
            print(f"  Partidos OOS ajustados:  {len(self._ttt.composition)}")
            print(f"  Gamma (volatilidad):     {self.ttt_gamma}")
            print(f"  Estrategias con curvas:  {len(ttt_curves)}")
            # Top-5 por mu final (última observación TTT)
            final_mus = {}
            for name, curve in ttt_curves.items():
                if not curve.empty:
                    final_mus[name] = curve['ttt_mu'].iloc[-1]
            if final_mus:
                top_ttt = sorted(final_mus.items(), key=lambda x: x[1], reverse=True)
                print(f"  Ranking por mu_final TTT (Top-5):")
                for rank, (name, mu_val) in enumerate(top_ttt[:5], 1):
                    sigma_val = ttt_curves[name]['ttt_sigma'].iloc[-1]
                    print(f"    #{rank} {name}: mu={mu_val:.4f}, sigma={sigma_val:.4f}")
            if 'ttt_top3_mu_mean' in df.columns:
                print(f"  ttt_top3_mu_mean (media splits): "
                      f"{df['ttt_top3_mu_mean'].mean():.4f}")
                print(f"  ttt_top3_sigma_mean (media splits): "
                      f"{df['ttt_top3_sigma_mean'].mean():.4f}")
            print(f"  -> Curvas completas en engine.ttt_curves_ "
                  f"(DataFrame por estrategia con time_date, ttt_mu, ttt_sigma)")

        return df


# ══════════════════════════════════════════════════════════════════════════════
# TESTS ESTADISTICOS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

def compute_elo_predictive_power(validation_results: pd.DataFrame) -> Dict:
    """
    Test estadistico de poder predictivo del ELO en OOS (v2.0).

    H0: spread <= 0 (ELO no predice performance futura)
    H1: spread > 0  (ELO tiene poder predictivo real)

    Tests: t-test one-sided + bootstrap IC 95% (no parametrico).

    Ref: Lopez de Prado (2018), Cap. 11.
         Bailey & Lopez de Prado (2014), SSRN 2460551.
    """
    spreads = validation_results['spread'].dropna()
    n = len(spreads)

    if n < 3:
        logger.warning("Insuficientes splits para test estadistico (n<3).")
        return {}

    mean_spread = spreads.mean()
    std_spread = spreads.std(ddof=1)
    t_stat = mean_spread / (std_spread / np.sqrt(n))
    p_value = 1 - stats.t.cdf(t_stat, df=n - 1)
    win_rate = (spreads > 0).mean()

    # Bootstrap IC 95% para la media (no parametrico, 10k replicas)
    np.random.seed(42)
    bootstrap_means = [
        spreads.sample(n, replace=True).mean() for _ in range(10_000)
    ]
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    psr_spread = compute_probabilistic_sharpe_ratio(spreads, sr_benchmark=0.0)

    print("\n" + "=" * 80)
    print("TEST DE PODER PREDICTIVO DEL ELO v2.0")
    print("=" * 80)
    print(f"\n  Spread medio:           {mean_spread:.2%}")
    print(f"  T-estadistico:          {t_stat:.3f}")
    print(f"  P-valor (one-sided):    {p_value:.4f}")
    print(f"  Win rate:               {win_rate:.1%} ({(spreads > 0).sum()}/{n})")
    print(f"  IC Bootstrap 95%:       [{ci_lower:.2%}, {ci_upper:.2%}]")
    print(f"  PSR del spread:         {psr_spread:.3f}  [Bailey & LdP 2012]")

    if p_value < 0.05 and mean_spread > 0 and ci_lower > 0:
        print("\n  ELO TIENE PODER PREDICTIVO ESTADISTICAMENTE SIGNIFICATIVO.")
        print(f"  Top-3 supera Bottom-3 en {mean_spread:.2%} "
              f"(p={p_value:.3f}, IC=[{ci_lower:.2%},{ci_upper:.2%}]).")
    elif p_value < 0.10:
        print("\n  Evidencia marginal de poder predictivo (p<0.10).")
        print("  Requiere mas splits OOS para conclusion definitiva.")
    else:
        print("\n  ELO NO PREDICE PERFORMANCE FUTURA (p>=0.10).")
        print("  Considerar: mas datos, CPCV, o revisar fitness_function.")

    return {
        'mean_spread': float(mean_spread),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'win_rate': float(win_rate),
        'ci_lower_95': float(ci_lower),
        'ci_upper_95': float(ci_upper),
        'psr_spread': float(psr_spread) if psr_spread is not None else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCION PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward_validation():
    """
    Punto de entrada con configuracion institucional recomendada.

    Configuracion de referencia para Hedge Fund Tier 1:
    ─────────────────────────────────────────────────────────────────────────
    purge_days=5:   1 semana. Cubre horizonte de etiquetas de retorno t+1..t+5.
    embargo_days=10: 2 semanas. Absorbe autocorrelacion serial de microestructura.
                    Regla: >= 0.01 * train_size (Lopez de Prado 2018, Cap. 7).
    fitness='calmar': Penaliza drawdowns, reduciendo sobreajuste a estrategias
                      con retornos altos pero ruina potencial.
    use_cpcv=True:  Validacion final — distribucion OOS vs. unica realizacion.
    ─────────────────────────────────────────────────────────────────────────
    """
    import json

    engine = WalkForwardELOEngine(
        n_splits=50,
        min_train_days=252,
        test_days=21,
        purge_days=5,
        embargo_days=10,
        use_cpcv=False,             # Cambiar a True para validacion final
        cpcv_n_folds=6,
        cpcv_n_test_folds=2,
        fitness_function='calmar',
        ttt_gamma=0.03,             # Punto de partida si auto_calibrate=False
        ttt_sigma=1.0,              # Punto de partida si auto_calibrate=False
        ttt_iterations=6,           # Convergencia final (estricta)
        ttt_auto_calibrate=False,   # True: grid search (gamma, sigma) por geometric_mean
    )

    results_df = engine.run_walk_forward_validation(
        symbol='SPY',
        start_date='1997-01-01',
        end_date='2024-01-01',
        initial_capital=100_000
    )

    stats_results = compute_elo_predictive_power(results_df)

    results_df.to_csv('walk_forward_elo_results_v3.csv', index=False)
    print("\nResultados -> walk_forward_elo_results_v3.csv")

    with open('elo_predictive_power_stats_v3.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    print("Estadisticas -> elo_predictive_power_stats_v3.json")

    # Exportar learning curves TTT por estrategia (ttt_mu, ttt_sigma vs time)
    if hasattr(engine, 'ttt_curves_') and engine.ttt_curves_:
        for strat_name, curve_df in engine.ttt_curves_.items():
            safe_name = strat_name.replace(' ', '_').replace('/', '_')
            fname = f'ttt_curve_{safe_name}.csv'
            curve_df.to_csv(fname, index=False)
        print(f"Curvas TTT exportadas para {len(engine.ttt_curves_)} estrategias.")

    return results_df, stats_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_walk_forward_validation()
