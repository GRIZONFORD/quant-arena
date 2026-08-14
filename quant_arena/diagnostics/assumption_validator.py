# =============================================================================
# FILE: quant_arena/diagnostics/assumption_validator.py
# Validación de supuestos estadísticos — el pipeline HMM→GARCH→señal (y
# cualquier otro modelo del Zoo) no debería entrenar sin que sus supuestos
# hayan sido contrastados. Ver H8/H9 en el plan de diseño.
# =============================================================================
"""
Regla de diseño: ningún test de este módulo LANZA una excepción solo porque
H0 se rechaza — el rechazo es un resultado informativo esperado (ver H9:
los retornos financieros son leptocúrticos, se espera rechazar normalidad),
no un error. `SupuestoEstadisticoError` se reserva para cuando el test
mismo no puede computarse de forma bien definida (datos degenerados, matriz
singular) — el borde defensivo, no el resultado.

Cada `test_*` retorna un `AssumptionReport` inmutable con: el estadístico,
el p-valor, la decisión (rechaza/no rechaza H0 a nivel `alfa`) y una acción
recomendada — de modo que rechazar H0 siempre tiene una siguiente pregunta
respondida, nunca un callejón sin salida.

Protocolo de contraste (α=0.05 por defecto), tabla completa en el plan:

| Supuesto            | Prueba                        | H0                              |
|---------------------|--------------------------------|----------------------------------|
| Normalidad          | Jarque-Bera / Shapiro-Wilk     | Los datos ~ N(μ,σ²)              |
| Estacionariedad     | ADF + KPSS (confirmatorias)   | ADF: raíz unitaria / KPSS: estac.|
| Independencia       | Ljung-Box + Durbin-Watson      | No hay autocorrelación           |
| Homocedasticidad    | Breusch-Pagan                  | Varianza constante               |
| Multicolinealidad   | VIF                            | (umbral, no es un test formal)   |
| Orden de Markov     | χ² (Anderson-Goodman/Billingsley)| La secuencia es Markov orden 1 |
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2, shapiro

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.tools import add_constant
    from statsmodels.tsa.stattools import adfuller, kpss
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False

from quant_arena.core.excepciones import ConfiguracionInvalidaError, SupuestoEstadisticoError

_N_SHAPIRO_MAX = 5000  # Shapiro-Wilk es más potente en n chico; JB es asintótico

SerieNumerica = Union[np.ndarray, "pd.Series[float]", Sequence[float]]


@dataclass(frozen=True)
class AssumptionReport:
    """
    Resultado inmutable de contrastar un supuesto estadístico.

    Args:
        supuesto:            Nombre del supuesto contrastado (ej. 'normalidad').
        metodo:               Test(s) usado(s) (ej. 'Shapiro-Wilk', 'ADF+KPSS').
        estadistico:          Estadístico de prueba del test primario.
        p_valor:              p-valor del test primario.
        alfa:                 Nivel de significancia usado en la decisión.
        rechaza_h0:            True si se rechaza H0 al nivel `alfa`.
        decision:             Descripción legible de la decisión.
        accion_recomendada:   Qué hacer si se rechaza H0 (nunca vacío).
        detalle:              Estadísticos secundarios (ej. KPSS cuando el
                              primario es ADF, o Durbin-Watson junto a Ljung-Box).
    """
    supuesto:           str
    metodo:              str
    estadistico:         float
    p_valor:              float
    alfa:                 float
    rechaza_h0:            bool
    decision:             str
    accion_recomendada:   str
    detalle:              Mapping[str, float] = field(default_factory=dict)


class AssumptionValidator:
    """
    Batería de contrastes de supuestos estadísticos sobre series de retornos
    o residuos. Cada método es independiente y se puede invocar por separado.
    """

    def __init__(self, alfa: float = 0.05) -> None:
        if not _STATSMODELS_OK:
            raise ImportError("pip install statsmodels")
        if not (0.0 < alfa < 1.0):
            raise ConfiguracionInvalidaError(f"alfa={alfa} debe estar en (0, 1).")
        self.alfa = alfa

    # ------------------------------------------------------------------
    # Normalidad
    # ------------------------------------------------------------------

    def test_normalidad(self, serie: SerieNumerica) -> AssumptionReport:
        """
        Normalidad de `serie`: Shapiro-Wilk si n < 5000 (más potente en
        muestras chicas), Jarque-Bera si n >= 5000 (asintótico, JB es
        computacionalmente estable en n grande donde Shapiro-Wilk degrada).

        H0: los datos provienen de una distribución N(μ, σ²).
        """
        x = self._limpiar(serie)
        if len(x) < 8:
            raise SupuestoEstadisticoError(
                f"test_normalidad requiere n>=8, recibido n={len(x)}."
            )

        jb_stat, jb_p, _skew, _kurt = jarque_bera(x)

        if len(x) < _N_SHAPIRO_MAX:
            estadistico, p_valor = shapiro(x)
            metodo = "Shapiro-Wilk"
            detalle = {"jarque_bera_stat": float(jb_stat), "jarque_bera_p": float(jb_p)}
        else:
            estadistico, p_valor = float(jb_stat), float(jb_p)
            metodo = "Jarque-Bera"
            detalle = {}

        rechaza = bool(p_valor < self.alfa)
        return AssumptionReport(
            supuesto="normalidad",
            metodo=metodo,
            estadistico=float(estadistico),
            p_valor=float(p_valor),
            alfa=self.alfa,
            rechaza_h0=rechaza,
            decision=(
                f"Se rechaza normalidad (p={p_valor:.4g} < {self.alfa})."
                if rechaza else
                f"No se rechaza normalidad (p={p_valor:.4g} >= {self.alfa})."
            ),
            accion_recomendada=(
                "Usar emisiones t de Student (o GMM-HMM) y GARCH con dist='t' "
                "o 'skewt' en vez de dist='normal'."
                if rechaza else
                "Emisiones gaussianas y GARCH dist='normal' son razonables."
            ),
            detalle=detalle,
        )

    # ------------------------------------------------------------------
    # Estacionariedad
    # ------------------------------------------------------------------

    def test_estacionariedad(self, serie: SerieNumerica) -> AssumptionReport:
        """
        ADF (H0: raíz unitaria / no estacionaria) + KPSS (H0: estacionaria) —
        confirmatorias en direcciones opuestas. Se reporta 'estacionaria'
        solo si ambos tests coinciden; si discrepan (posible tendencia-
        estacionaria vs diferencia-estacionaria), se marca 'inconclusa'.
        """
        x = self._limpiar(serie)
        if len(x) < 20:
            raise SupuestoEstadisticoError(
                f"test_estacionariedad requiere n>=20, recibido n={len(x)}."
            )

        try:
            adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
        except Exception as exc:
            raise SupuestoEstadisticoError(f"ADF no pudo computarse: {exc}") from exc

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # InterpolationWarning fuera de tabla
                kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
        except Exception as exc:
            raise SupuestoEstadisticoError(f"KPSS no pudo computarse: {exc}") from exc

        adf_dice_estacionaria = bool(adf_p < self.alfa)
        kpss_dice_estacionaria = bool(kpss_p >= self.alfa)

        if adf_dice_estacionaria and kpss_dice_estacionaria:
            decision_txt, rechaza, accion = "estacionaria (ADF y KPSS coinciden)", False, (
                "No se requiere diferenciación adicional."
            )
        elif not adf_dice_estacionaria and not kpss_dice_estacionaria:
            decision_txt, rechaza, accion = "no estacionaria (ADF y KPSS coinciden)", True, (
                "Diferenciar la serie (ya se usan log-retornos) o modelar la "
                "no-estacionariedad explícitamente como cambio de régimen (HMM)."
            )
        else:
            decision_txt, rechaza, accion = (
                "inconclusa (ADF y KPSS discrepan — posible tendencia-estacionaria)",
                True,
                "Tratar con cautela: inspeccionar tendencia determinística; "
                "el HMM de régimen ya captura parte de esta no-estacionariedad "
                "aparente al modelar la media condicional por estado.",
            )

        return AssumptionReport(
            supuesto="estacionariedad",
            metodo="ADF+KPSS",
            estadistico=float(adf_stat),
            p_valor=float(adf_p),
            alfa=self.alfa,
            rechaza_h0=rechaza,
            decision=decision_txt,
            accion_recomendada=accion,
            detalle={"kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p)},
        )

    # ------------------------------------------------------------------
    # Independencia de residuos
    # ------------------------------------------------------------------

    def test_independencia_residuos(
        self,
        serie: SerieNumerica,
        lags: Sequence[int] = (10, 20),
    ) -> AssumptionReport:
        """
        Ljung-Box (H0: no hay autocorrelación hasta el lag dado) en los
        lags indicados + Durbin-Watson (estadístico descriptivo, sin
        p-valor formal, reportado en `detalle`).

        Se rechaza independencia si Ljung-Box rechaza en CUALQUIERA de los
        lags evaluados (peor caso = p-valor mínimo).
        """
        x = self._limpiar(serie)
        max_lag = max(lags)
        if len(x) < max_lag + 10:
            raise SupuestoEstadisticoError(
                f"test_independencia_residuos requiere n>={max_lag + 10}, recibido n={len(x)}."
            )

        tabla = acorr_ljungbox(x, lags=list(lags), return_df=True)
        peor_lag = int(tabla["lb_pvalue"].idxmin())
        p_valor = float(tabla.loc[peor_lag, "lb_pvalue"])
        estadistico = float(tabla.loc[peor_lag, "lb_stat"])
        dw = float(durbin_watson(x))

        rechaza = bool(p_valor < self.alfa)
        return AssumptionReport(
            supuesto="independencia_residuos",
            metodo=f"Ljung-Box(lags={list(lags)})",
            estadistico=estadistico,
            p_valor=p_valor,
            alfa=self.alfa,
            rechaza_h0=rechaza,
            decision=(
                f"Se rechaza independencia en lag={peor_lag} (p={p_valor:.4g})."
                if rechaza else
                f"No se rechaza independencia (peor p={p_valor:.4g} en lag={peor_lag})."
            ),
            accion_recomendada=(
                "Añadir términos AR al modelo o aumentar el número de estados "
                "del HMM — autocorrelación residual sugiere un régimen faltante."
                if rechaza else
                "Los residuos son consistentes con ruido blanco condicional al régimen."
            ),
            detalle={"durbin_watson": dw},
        )

    # ------------------------------------------------------------------
    # Homocedasticidad
    # ------------------------------------------------------------------

    def test_homocedasticidad(self, serie: SerieNumerica, nlags: int = 1) -> AssumptionReport:
        """
        ARCH-LM (Engle, 1982): Lagrange-Multiplier test de heterocedasticidad
        condicional, estructuralmente un Breusch-Pagan de `serie_t²` sobre
        sus propios rezagos al cuadrado (`serie_{t-1}², ..., serie_{t-nlags}²`)
        — a diferencia de un Breusch-Pagan genérico con exógenas en NIVEL
        (que no detecta clustering de volatilidad por simetría: retornos
        lag muy positivos y muy negativos preceden ambos a varianza alta,
        cancelando la correlación lineal con el nivel). H0: varianza
        condicional constante (sin efectos ARCH).
        """
        x = self._limpiar(serie)
        if len(x) < 20 + nlags:
            raise SupuestoEstadisticoError(
                f"test_homocedasticidad requiere n>={20 + nlags}, recibido n={len(x)}."
            )

        try:
            bp_stat, bp_p, _f_stat, _f_p = het_arch(x, nlags=nlags)
        except Exception as exc:
            raise SupuestoEstadisticoError(f"ARCH-LM no pudo computarse: {exc}") from exc

        rechaza = bool(bp_p < self.alfa)
        return AssumptionReport(
            supuesto="homocedasticidad",
            metodo=f"ARCH-LM(nlags={nlags})",
            estadistico=float(bp_stat),
            p_valor=float(bp_p),
            alfa=self.alfa,
            rechaza_h0=rechaza,
            decision=(
                f"Se rechaza homocedasticidad (p={bp_p:.4g})."
                if rechaza else
                f"No se rechaza homocedasticidad (p={bp_p:.4g})."
            ),
            accion_recomendada=(
                "Esperable en series financieras (volatility clustering) — "
                "justifica formalmente el uso de GARCH ya presente en el pipeline; "
                "no se requiere acción correctiva adicional."
                if rechaza else
                "La varianza constante simplifica el modelo; GARCH sigue siendo "
                "válido pero no es estrictamente necesario por este criterio."
            ),
            detalle={},
        )

    # ------------------------------------------------------------------
    # Multicolinealidad (VIF) — no es un test de hipótesis, es un umbral
    # ------------------------------------------------------------------

    def calcular_vif(self, features: pd.DataFrame, umbral: float = 10.0) -> pd.DataFrame:
        """
        Variance Inflation Factor por columna de `features`.

        Returns:
            DataFrame indexado por nombre de feature con columnas
            ['vif', 'excede_umbral'].
        """
        if features.shape[1] < 2:
            raise SupuestoEstadisticoError(
                "calcular_vif requiere >= 2 columnas (VIF no está definido para una sola)."
            )
        datos = features.dropna()
        if len(datos) < features.shape[1] + 2:
            raise SupuestoEstadisticoError(
                f"calcular_vif requiere n > n_features+1, recibido n={len(datos)}, "
                f"n_features={features.shape[1]}."
            )

        con_const = add_constant(datos)
        try:
            vifs = {
                col: float(variance_inflation_factor(con_const.values, con_const.columns.get_loc(col)))
                for col in datos.columns
            }
        except Exception as exc:
            raise SupuestoEstadisticoError(f"VIF no pudo computarse: {exc}") from exc

        return pd.DataFrame({
            "vif": pd.Series(vifs),
            "excede_umbral": pd.Series({k: v >= umbral for k, v in vifs.items()}),
        })

    # ------------------------------------------------------------------
    # Orden de Markov de una secuencia de estados (Viterbi)
    # ------------------------------------------------------------------

    def test_orden_markov(
        self,
        secuencia_estados: Union[np.ndarray, Sequence[int]],
        n_estados: Optional[int] = None,
    ) -> AssumptionReport:
        """
        Test de razón de verosimilitud (Anderson & Goodman, 1957 / Billingsley,
        1961) de orden-1 vs orden-2: H0 = la cadena es Markov de orden 1
        (P(s_t | s_{t-1}, s_{t-2}) = P(s_t | s_{t-1})).

        Estadístico G² = 2·Σ n_ijk·ln(n_ijk / n̂_ijk), n̂_ijk = n_ij·n_jk/n_j,
        con distribución asintótica χ²(K(K-1)²) bajo H0.

        Args:
            secuencia_estados: Secuencia de enteros en {0, ..., K-1} (ej. la
                               secuencia de Viterbi de un HMM ajustado).
            n_estados:         K. Si None, se infiere como max(secuencia)+1.
        """
        seq = np.asarray(list(secuencia_estados), dtype=int)
        if len(seq) < 30:
            raise SupuestoEstadisticoError(
                f"test_orden_markov requiere n>=30, recibido n={len(seq)}."
            )
        K = n_estados if n_estados is not None else int(seq.max()) + 1
        if K < 2:
            raise SupuestoEstadisticoError("test_orden_markov requiere >= 2 estados distintos.")

        conteo = np.zeros((K, K, K), dtype=float)
        for t in range(2, len(seq)):
            i, j, k = seq[t - 2], seq[t - 1], seq[t]
            conteo[i, j, k] += 1.0

        n_ij_ = conteo.sum(axis=2)        # [i, j]
        n_jk = conteo.sum(axis=0)         # [j, k]
        n_j__ = conteo.sum(axis=(0, 2))   # [j]

        g_stat = 0.0
        for i in range(K):
            for j in range(K):
                for k in range(K):
                    n_ijk = conteo[i, j, k]
                    if n_ijk <= 0.0 or n_j__[j] <= 0.0:
                        continue
                    esperado = n_ij_[i, j] * n_jk[j, k] / n_j__[j]
                    if esperado <= 0.0:
                        continue
                    g_stat += n_ijk * np.log(n_ijk / esperado)
        g_stat *= 2.0

        df = K * (K - 1) ** 2
        p_valor = float(chi2.sf(g_stat, df)) if df > 0 else float("nan")
        rechaza = bool(p_valor < self.alfa)

        return AssumptionReport(
            supuesto="orden_markov",
            metodo="G-test orden-1 vs orden-2 (Anderson-Goodman)",
            estadistico=float(g_stat),
            p_valor=p_valor,
            alfa=self.alfa,
            rechaza_h0=rechaza,
            decision=(
                f"Se rechaza Markov de orden 1 (p={p_valor:.4g}, df={df})."
                if rechaza else
                f"No se rechaza Markov de orden 1 (p={p_valor:.4g}, df={df})."
            ),
            accion_recomendada=(
                "Considerar un HMM de orden superior o un HSMM (duración "
                "explícita de estado) — el proceso tiene memoria más allá "
                "del estado inmediatamente anterior."
                if rechaza else
                "El supuesto de Markov de orden 1 del HMM es adecuado."
            ),
            detalle={"df": float(df)},
        )

    # ------------------------------------------------------------------
    # Conveniencia
    # ------------------------------------------------------------------

    def validar_serie_completa(self, serie: SerieNumerica) -> Dict[str, AssumptionReport]:
        """
        Ejecuta normalidad, estacionariedad, independencia y homocedasticidad
        sobre `serie` en un solo llamado. VIF y orden de Markov requieren
        entradas de otra forma (matriz de features / secuencia de estados)
        y se invocan por separado.
        """
        return {
            "normalidad": self.test_normalidad(serie),
            "estacionariedad": self.test_estacionariedad(serie),
            "independencia_residuos": self.test_independencia_residuos(serie),
            "homocedasticidad": self.test_homocedasticidad(serie),
        }

    @staticmethod
    def _limpiar(serie: SerieNumerica) -> np.ndarray:
        x = np.asarray(serie, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        return x
