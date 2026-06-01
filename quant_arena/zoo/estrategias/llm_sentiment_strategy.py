#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/zoo/estrategias/llm_sentiment_strategy.py
# LLM Agentic Sentiment Strategy — FinBERT + fallback tecnico
# =============================================================================
"""
Combina sentiment financiero LLM (FinBERT / distilbert-finance) con proxies
tecnicos de sentimiento derivados de OHLCV cuando no hay texto disponible.

Pipeline con texto:
    textos_noticias -> FinBERT pipeline -> score_sentimiento [-1, 1]
    -> EWMA(score, span=5) -> señal discreta

Pipeline de fallback (OHLCV):
    - Volatilidad realizada como proxy de incertidumbre (alta vol -> bearish)
    - Anomalia de volumen: vol / media_90d (picos de volumen con retorno negativo = miedo)
    - Distancia al maximo 52 semanas (momentum negativo = bearish)
    - Skewness de retornos rolling (asimetria negativa = tail risk percibido)

La score final se combina: score = alpha*llm_score + (1-alpha)*tech_score
donde alpha=1.0 si hay texto, alpha=0.0 en modo fallback puro.

Registro: 'llm_sentiment'
"""
from __future__ import annotations

import sys, warnings, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HF_OK = True
except ImportError:
    _HF_OK = False

from quant_arena.core.abstracciones import AbstractStrategy
from quant_arena.zoo.base_estrategia import RegistroZoo

logger = logging.getLogger(__name__)

# Modelos FinBERT candidatos (en orden de preferencia)
_FINBERT_CANDIDATES = [
    "ProsusAI/finbert",
    "yiyanghkust/finbert-tone",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
]


# =============================================================================
# Componentes
# =============================================================================

class FinBERTScorer:
    """
    Wrapper sobre un pipeline Hugging Face de clasificacion de sentimiento
    financiero. Convierte etiquetas positivo/negativo/neutro en scores [-1, 1].

    Args:
        model_name: Nombre del modelo HuggingFace (FinBERT o compatible).
        max_length: Longitud maxima de tokens por texto.
        batch_size: Tamano de batch para inferencia.
        device:     'cpu' | 'cuda'.
    """

    _LABEL_MAP_POS = {"positive", "pos", "bullish", "label_2"}
    _LABEL_MAP_NEG = {"negative", "neg", "bearish", "label_0"}

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length:  int = 128,
        batch_size:  int = 16,
        device:      str = "cpu",
    ) -> None:
        if not _HF_OK:
            raise ImportError("pip install transformers torch")
        self._model_name = model_name
        self._max_length  = max_length
        self._batch_size  = batch_size
        self._device      = device
        self._pipe        = None

    def _load(self) -> None:
        """Carga el pipeline (lazy, solo si se necesita)."""
        if self._pipe is not None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for candidate in _FINBERT_CANDIDATES:
                try:
                    self._pipe = hf_pipeline(
                        "text-classification",
                        model       = candidate,
                        tokenizer   = candidate,
                        device      = 0 if (self._device == "cuda" and _HF_OK) else -1,
                        max_length  = self._max_length,
                        truncation  = True,
                        batch_size  = self._batch_size,
                    )
                    self._model_name = candidate
                    break
                except Exception:
                    continue
            if self._pipe is None:
                raise RuntimeError("No se pudo cargar ningun modelo FinBERT.")

    def score_texts(self, texts: List[str]) -> float:
        """
        Clasifica una lista de textos y retorna el score promedio en [-1, 1].

        Mapeo de etiquetas:
            positive/bullish -> +1.0
            negative/bearish -> -1.0
            neutral/other    -> 0.0

        Args:
            texts: Lista de textos de noticias financieras.

        Returns:
            score: Promedio ponderado por confianza en [-1, 1].
        """
        self._load()
        if not texts:
            return 0.0
        try:
            resultados = self._pipe(texts)
            scores = []
            for r in resultados:
                label  = r["label"].lower()
                conf   = float(r["score"])
                if any(p in label for p in self._LABEL_MAP_POS):
                    scores.append(+conf)
                elif any(n in label for n in self._LABEL_MAP_NEG):
                    scores.append(-conf)
                else:
                    scores.append(0.0)
            return float(np.mean(scores)) if scores else 0.0
        except Exception:
            return 0.0


class TechnicalSentimentProxy:
    """
    Proxy de sentimiento a partir de datos OHLCV, usado cuando no hay texto.

    Calcula cuatro senales complementarias normalizadas a [-1, 1]:
        1. vol_fear:    Alta volatilidad realizada -> sentimiento negativo.
        2. vol_anomaly: Picos de volumen con retorno negativo -> miedo.
        3. hi52w_dist:  Distancia relativa al maximo de 52 semanas -> momentum.
        4. ret_skew:    Asimetria de retornos rolling -> riesgo de cola percibido.
    """

    def __init__(
        self,
        vol_window:   int = 21,
        vol_ma_window: int = 90,
        hi52w_window: int = 252,
        skew_window:  int = 63,
    ) -> None:
        self._vol_w    = vol_window
        self._vma_w    = vol_ma_window
        self._hi52w    = hi52w_window
        self._skew_w   = skew_window

    def compute(self, datos: pd.DataFrame, close_col: str = "Close",
                volume_col: str = "Volume") -> pd.Series:
        """
        Calcula el score de sentimiento tecnico para cada fecha.

        Returns:
            pd.Series con scores en [-1, 1], mismo indice que datos.
        """
        close  = datos[close_col].astype(float)
        log_ret = np.log(close / close.shift(1))

        # 1. Volatilidad realizada -> miedo (normalizada por percentil historico)
        vol_real = log_ret.rolling(self._vol_w, min_periods=self._vol_w).std()
        vol_pct  = vol_real.rank(pct=True)       # 0=baja vol, 1=alta vol
        vol_fear = -(2.0 * vol_pct - 1.0)        # invertir: alta vol -> negativo

        # 2. Anomalia de volumen + direccion del retorno
        vol_anomaly = pd.Series(0.0, index=datos.index)
        if volume_col in datos.columns:
            vol_ser  = datos[volume_col].astype(float)
            vol_ma   = vol_ser.rolling(self._vma_w, min_periods=21).mean()
            vol_ratio = (vol_ser / vol_ma.replace(0, np.nan)).fillna(1.0)
            # Pico de volumen + retorno negativo = panico vendedor
            vol_anomaly = vol_ratio.clip(0.5, 3.0) * np.sign(log_ret.fillna(0.0))
            # Normalizar a [-1, 1]
            denom = vol_anomaly.abs().rolling(self._vma_w, min_periods=21).max().replace(0, 1.0)
            vol_anomaly = (vol_anomaly / denom).clip(-1.0, 1.0)

        # 3. Distancia al maximo de 52 semanas (momentum negativo = bearish)
        hi52w = close.rolling(self._hi52w, min_periods=126).max()
        hi52w_dist = (close / hi52w.replace(0, np.nan) - 1.0).clip(-0.5, 0.0) * 4.0
        # [-0.5, 0] -> [-2, 0] -> clip [-1, 1] = [-1, 0]  (siempre <=0 o =0)

        # 4. Skewness de retornos rolling (asimetria negativa = tail risk)
        skew_ser  = log_ret.rolling(self._skew_w, min_periods=30).skew()
        skew_norm = np.tanh(skew_ser / 2.0)   # comprimir a aprox (-1, 1)

        # Combinar con pesos iguales
        score = (0.35 * vol_fear + 0.25 * vol_anomaly + 0.20 * hi52w_dist + 0.20 * skew_norm)
        return score.clip(-1.0, 1.0)


# =============================================================================
# AbstractStrategy — wrapper Zoo
# =============================================================================

@RegistroZoo.registrar("llm_sentiment")
class LLMSentimentStrategy(AbstractStrategy):
    """
    Estrategia de sentimiento LLM+tecnico para quant_arena.

    Usa FinBERT (Hugging Face) para clasificar noticias financieras cuando
    la columna de texto esta disponible en `datos`. En ausencia de texto,
    cae a un proxy tecnico multi-factor derivado de OHLCV.

    El score final de sentimiento se suaviza con EWMA y se discretiza:
        score_suavizado > +umbral  -> largo (+1)
        score_suavizado < -umbral  -> corto (-1)
        otherwise                  -> neutral (0)

    Args:
        universo:             Tickers (un elemento).
        text_col:             Columna de texto de noticias en datos (None=fallback puro).
        close_col:            Columna de cierre.
        volume_col:           Columna de volumen.
        umbral_score:         Umbral del score para emitir señal.
        ewma_span:            Span de EWMA para suavizar score de sentimiento.
        min_train_days:       Dias minimos de historial OHLCV.
        finbert_model:        Nombre modelo HuggingFace (None=auto-detect).
        max_text_length:      Tokens maximos por texto.
        device:               'cpu' | 'cuda'.
    """

    def __init__(
        self,
        universo:        List[str],
        text_col:        Optional[str]  = None,
        close_col:       str            = "Close",
        volume_col:      str            = "Volume",
        umbral_score:    float          = 0.20,
        ewma_span:       int            = 5,
        min_train_days:  int            = 63,
        finbert_model:   Optional[str]  = None,
        max_text_length: int            = 128,
        device:          Optional[str]  = None,
    ) -> None:
        super().__init__(nombre="llm_sentiment", universo=universo)

        self._text_col    = text_col
        self._close_col   = close_col
        self._volume_col  = volume_col
        self._umbral      = umbral_score
        self._ewma_span   = ewma_span
        self._min_train   = min_train_days
        self._max_len     = max_text_length

        if device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Scorer LLM (lazy load)
        self._llm_scorer: Optional[FinBERTScorer] = None
        if _HF_OK and text_col is not None:
            model = finbert_model or _FINBERT_CANDIDATES[0]
            self._llm_scorer = FinBERTScorer(
                model_name  = model,
                max_length  = max_text_length,
                device      = self._device,
            )

        # Proxy tecnico (siempre disponible)
        self._tech_proxy = TechnicalSentimentProxy()

        # Cache de scores LLM por fecha (evitar re-clasificar)
        self._cache_llm: Dict[pd.Timestamp, float] = {}

    @property
    def descripcion(self) -> str:
        modo = "LLM+tech" if (self._llm_scorer is not None) else "tech-proxy"
        return (
            f"LLM-Sentiment [{modo}] | umbral={self._umbral:.2f} "
            f"| ewma={self._ewma_span}d"
        )

    # ------------------------------------------------------------------
    def _score_llm_en_fecha(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> Optional[float]:
        """
        Extrae y clasifica el texto disponible hasta fecha_corte.

        Si datos[text_col] contiene una lista/str de noticias en esa fecha,
        las clasifica con FinBERT y retorna el score agregado.

        Returns:
            float en [-1, 1] o None si no hay texto util.
        """
        if self._text_col is None or self._text_col not in datos.columns:
            return None
        if self._llm_scorer is None:
            return None

        # Buscar texto en la fecha exacta o la mas reciente anterior
        mask = datos.index <= fecha_corte
        texto_series = datos.loc[mask, self._text_col].dropna()
        if texto_series.empty:
            return None

        ultimo_texto = texto_series.iloc[-1]
        fecha_texto  = texto_series.index[-1]

        # Cache por fecha de texto
        if fecha_texto in self._cache_llm:
            return self._cache_llm[fecha_texto]

        # Normalizar a lista de strings
        if isinstance(ultimo_texto, str) and ultimo_texto.strip():
            textos = [ultimo_texto]
        elif isinstance(ultimo_texto, (list, tuple)):
            textos = [t for t in ultimo_texto if isinstance(t, str) and t.strip()]
        else:
            return None

        if not textos:
            return None

        try:
            score = self._llm_scorer.score_texts(textos)
            self._cache_llm[fecha_texto] = score
            return score
        except Exception:
            return None

    def _score_tecnico_en_fecha(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> float:
        """
        Calcula el score tecnico de sentimiento para la fecha corte.
        """
        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns or len(hist) < self._min_train:
            return 0.0
        scores = self._tech_proxy.compute(hist, self._close_col, self._volume_col)
        if scores.empty or scores.isna().all():
            return 0.0
        return float(scores.dropna().iloc[-1]) if not scores.dropna().empty else 0.0

    def _score_historico(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        """
        Construye serie de scores diarios hasta fecha_corte para aplicar EWMA.

        Usa proxy tecnico para todos los dias (rapido), y sobreescribe con
        scores LLM cuando estan disponibles (slow pero mas preciso).
        """
        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns:
            return pd.Series(dtype=float)

        # Score tecnico para toda la historia
        tech_scores = self._tech_proxy.compute(hist, self._close_col, self._volume_col)

        # Mezclar con LLM si hay columna de texto
        if self._text_col is not None and self._text_col in hist.columns:
            texto_s = hist[self._text_col].dropna()
            for fecha, texto in texto_s.items():
                if fecha in self._cache_llm:
                    tech_scores[fecha] = self._cache_llm[fecha]
                else:
                    # Clasificar en batch solo el ultimo para no exceder tiempo
                    if fecha == texto_s.index[-1]:
                        llm_s = self._score_llm_en_fecha(hist, fecha)
                        if llm_s is not None:
                            tech_scores[fecha] = llm_s

        return tech_scores

    # ------------------------------------------------------------------
    def generar_señales(
        self, datos: pd.DataFrame, fecha_corte: pd.Timestamp
    ) -> pd.Series:
        indice  = pd.Index(self._universo)
        neutral = pd.Series(0.0, index=indice, dtype=float)

        hist = datos.loc[datos.index <= fecha_corte]
        if self._close_col not in hist.columns or len(hist) < self._min_train:
            return neutral

        # Construir serie de scores y suavizar con EWMA
        try:
            scores_hist = self._score_historico(datos, fecha_corte)
        except Exception:
            return neutral

        if scores_hist.empty or scores_hist.isna().all():
            return neutral

        scores_clean = scores_hist.dropna()
        if scores_clean.empty:
            return neutral

        # EWMA para suavizar ruido de sentimiento
        score_suavizado = float(
            scores_clean.ewm(span=self._ewma_span, adjust=False).mean().iloc[-1]
        )

        # Discretizar
        if score_suavizado > self._umbral:
            senal = 1.0
        elif score_suavizado < -self._umbral:
            senal = -1.0
        else:
            senal = 0.0

        pesos = pd.Series(0.0, index=indice, dtype=float)
        if self._universo:
            pesos.iloc[0] = senal
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
        assert not s.isnull().any(),        f"[{label}] NaN en senal"
        bad = set(s.values) - valid
        assert not bad,                     f"[{label}] Valores fuera de {{-1,0,1}}: {bad}"
        assert ticker in s.index,           f"[{label}] Ticker no en indice"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  LLMSentimentStrategy -- Suite de Pruebas [6 tests]")
    print(SEP)

    strat = LLMSentimentStrategy(
        universo=[TICKER],
        min_train_days=63,
        text_col=None,   # fallback mode: no FinBERT required
    )

    # TEST 1 -- Propiedades basicas
    print("\n[TEST 1] Instanciacion y propiedades...")
    assert strat.nombre == "llm_sentiment",         "nombre incorrecto"
    assert strat.universo == [TICKER],              "universo incorrecto"
    assert isinstance(strat.descripcion, str) and len(strat.descripcion) > 0
    print("  [OK] nombre='llm_sentiment' | universo | descripcion no vacia")

    # TEST 2 -- Warm-up (n=30 < min_train_days=63 -> neutral)
    print("\n[TEST 2] Warm-up (n=30 < 63) -> neutral...")
    df_short = _make_ohlcv(n=30)
    s = strat.generar_señales(df_short, df_short.index[-1])
    _check(s, "TEST2", TICKER)
    assert s[TICKER] == 0.0, f"Esperaba neutral, obtuvo {s[TICKER]}"
    print(f"  [OK] senal={s[TICKER]} neutral -- warm-up respetado")

    # TEST 3 -- Operacion normal (n=200 > 63)
    print("\n[TEST 3] Operacion normal (n=200, seed=42)...")
    df_normal = _make_ohlcv(n=200, seed=42)
    s = strat.generar_señales(df_normal, df_normal.index[-1])
    _check(s, "TEST3", TICKER)
    print(f"  [OK] senal={s[TICKER]} in {{-1.0, 0.0, 1.0}}")

    # TEST 4 -- Regimen bull extremo (mu=0.008, sigma=0.002)
    print("\n[TEST 4] Regimen bull extremo (mu=0.008, sigma=0.002, seed=7)...")
    df_bull = _make_ohlcv(n=200, mu=0.008, sigma=0.002, seed=7)
    s = strat.generar_señales(df_bull, df_bull.index[-1])
    _check(s, "TEST4", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- regimen extremo sin crash")

    # TEST 5 -- Sin columna 'Close' -> neutral sin excepcion (directo, sin _check)
    print("\n[TEST 5] Sin columna 'Close' -> neutral sin crash...")
    df_bad = _make_ohlcv(n=200).rename(columns={"Close": "Cierre"})
    s = strat.generar_señales(df_bad, df_bad.index[-1])
    assert isinstance(s, pd.Series),       "TEST5: No es pd.Series"
    assert not s.isnull().any(),           "TEST5: NaN en senal"
    assert (s == 0.0).all(),              "TEST5: Esperaba senal neutral con datos malformados"
    print("  [OK] neutral retornado sin excepcion")

    # TEST 6 -- Boundary condition (n=63, exactamente el umbral)
    print(f"\n[TEST 6] Boundary condition (n=63 = min_train_days exacto)...")
    df_boundary = _make_ohlcv(n=63, seed=7)
    s = strat.generar_señales(df_boundary, df_boundary.index[-1])
    _check(s, "TEST6", TICKER)
    print(f"  [OK] senal={s[TICKER]} -- boundary respetado")

    print(f"\n{SEP}")
    print("  TODOS LOS TESTS PASARON  [6 / 6]  OK")
    print(f"{SEP}\n")
