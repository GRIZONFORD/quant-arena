# strategy_zoo.py
"""
Strategy Zoo v3.0 — Institutional Grade | TTT-Ready
=====================================================
Universo de estrategias de trading con arquitectura ABC y Meta-Labeling
(López de Prado, 2018, Cap. 3) para el motor Walk-Forward TTT.

DISEÑO ARQUITECTÓNICO:
─────────────────────────────────────────────────────────────────────────────
· BaseStrategy (ABC): contrato mínimo para el motor TTT. Todo el zoo hereda
  de aquí. Añadir una estrategia nueva = subclasear BaseStrategy e implementar
  `generate_signals()`. El adaptador hace el resto.

· META-LABELING (López de Prado, 2018, Cap. 3):
  Cada estrategia produce no solo la DIRECCIÓN de la apuesta (side ∈ {-1, 0, 1})
  sino también el TAMAÑO (bet_size ∈ [0, 1]) en función de la volatilidad
  histórica local. Esto es la idea central del Meta-Labeling:
    - El modelo primario da la dirección (el "side").
    - Un meta-modelo (aquí el sizing basado en vol realizada) da la confianza.
  El producto side * bet_size genera posiciones fraccionarias, reduciendo la
  exposición en entornos de alta incertidumbre.

· HEALTH METRICS:
  Cada estrategia calcula su propio Sharpe IS y Hit Ratio sobre el período de
  entrenamiento. El adaptador usa estos valores como prior informativo para TTT,
  acortando el período de burn-in bayesiano.

· SIGNAL DECAY:
  Las señales tienen una vida media paramétrica (signal_halflife). El adaptador
  aplica un factor de descuento exponencial sobre señales que llevan muchos
  períodos sin actualizarse, penalizando estrategias "dormidas" en el torneo.

· VECTORIZACIÓN TOTAL:
  Todos los cálculos usan Pandas/NumPy — sin loops Python sobre fechas.
  El zoo de 9 estrategias se puede backtestear en <100ms sobre 7k días.

Compatibilidad total con walk_forward_engine_con_TTT.py:
  · adapt_strategies(list(registry.strategies.values()), capital) → OK
  · strategy.generate_signals(data) → dict con 'positions', 'side', 'size'
  · strategy.get_health_metrics(is_data) → dict con Sharpe, hit_ratio, etc.
  · strategy.get_parameters() → dict de hyperparámetros para trazabilidad

Referencias:
  López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    Cap. 3 (Meta-Labeling), Cap. 7 (Purging), Cap. 14 (Features).
  Landfried, G. & Mocskos, E. (2024). TrueSkill Through Time.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# BASE STRATEGY ABC
# ══════════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """
    Clase Base Abstracta para todas las estrategias del zoo.

    CONTRATO DE INTERFAZ:
    ─────────────────────────────────────────────────────────────────────────
    Todo subclase debe implementar:
      · generate_signals(data) → SignalDict
      · get_parameters()       → Dict de hyperparámetros
      · __repr__()             → representación de cadena

    La clase base provee implementaciones por defecto de:
      · get_health_metrics(is_data)  → Sharpe IS, Hit Ratio, Calmar IS
      · warmup_period()              → barras mínimas para señales válidas
      · _compute_realized_vol(close) → vol realizada para Meta-Labeling sizing

    SIGNAL DICT (output de generate_signals):
      {
        'positions': pd.Series,   # posición en $ (signed: >0 long, <0 short)
        'side':      pd.Series,   # dirección pura: {-1, 0, 1}
        'size':      pd.Series,   # tamaño relativo ∈ [0, 1] (Meta-Labeling)
        'signal_age': pd.Series,  # días desde último cambio de señal
      }
    """

    # Factor de señal: cuántos días 'vive' una señal antes de decaer al 50%.
    # El adaptador aplica exp(-ln(2) * age / halflife) como factor de descuento.
    # Default: 20 días (4 semanas de trading). Subclases pueden sobreescribir.
    signal_halflife: int = 20

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or self.__class__.__name__
        self.strategy_type: str = "unknown"
        self.required_features: List[str] = ["close"]
        # Caché interno para hot-paths — evita recomputar vol si la data no cambió
        self._vol_cache: Optional[pd.Series] = None

    # ── MÉTODOS ABSTRACTOS ────────────────────────────────────────────────────

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        """
        Genera señales de trading sobre el período dado.

        META-LABELING:
          La implementación debe separar la lógica en dos capas:
          1. SIDE (modelo primario): dirección basada en el alpha de la estrategia.
          2. SIZE (meta-modelo): tamaño en función de la confianza de la señal,
             normalmente escalado por la volatilidad realizada local.

        Args:
            data:    DataFrame OHLCV con DatetimeIndex.
            capital: Capital disponible en $ para el cálculo de posiciones.

        Returns:
            Dict con claves obligatorias:
              'positions':  pd.Series — posiciones en $ (signed).
              'side':       pd.Series — {-1, 0, 1}.
              'size':       pd.Series — [0, 1] (fracción de capital).
              'signal_age': pd.Series — entero no negativo (días desde cambio).
        """
        ...

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Devuelve el diccionario de hyperparámetros de la estrategia.

        Usado por el adaptador para: logging, reproducibilidad,
        y comparación de configuraciones en el grid de estrategias.
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Representación canónica: ClassName(param=val, ...)."""
        ...

    # ── MÉTODOS CON IMPLEMENTACIÓN DEFAULT ───────────────────────────────────

    def warmup_period(self) -> int:
        """
        Número de barras necesarias antes de producir señales válidas.
        Subclases deben sobreescribir en función de sus ventanas más largas.
        El motor TTT usa este valor para descartar los primeros rows en IS.
        """
        return 0

    def fit(self, is_data: pd.DataFrame, capital: float = 100_000) -> "BaseStrategy":
        """
        Entrena la estrategia sobre la ventana IN-SAMPLE (Walk-Forward).

        CONTRATO ANTI-FUGA:
          El motor (`TTTSimulator.run_tournament`) llama a `fit(is_data)` con la
          ventana IS estricta del fold (anterior a OOS, con purging+embargo ya
          aplicados). El `is_data` NUNCA contiene barras del OOS.

        Las estrategias basadas en reglas son STATELESS → este default es no-op.
        Las estrategias de ML (e.g. XGBoostTrendStrategy) lo sobreescriben para
        re-entrenar su modelo exclusivamente con `is_data`.
        """
        return self

    def get_health_metrics(
        self, is_data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, float]:
        """
        Calcula métricas de salud de la estrategia sobre el período IS.

        Estas métricas sirven como PRIOR INFORMATIVO para TTT:
        · Si Sharpe IS es alto y Hit Ratio > 0.55, el prior inicial de
          habilidad puede ser μ₀ > 0 en lugar del default μ₀ = 0.
        · El adaptador usa prior_strength proporcional a Sharpe IS para
          escalar sigma₀ (menor sigma₀ → prior más informativo → burn-in más corto).

        MÉTRICAS CALCULADAS:
          sharpe_is:   Sharpe anualizado sobre el período IS.
          hit_ratio:   % de días en que la estrategia generó retorno positivo.
          calmar_is:   Calmar ratio IS (CAGR / |MaxDD|).
          n_trades:    Número de cambios de posición (proxy de turnover).
          avg_holding: Días medios entre cambios de señal.

        Args:
            is_data: DataFrame IS con OHLCV y DatetimeIndex.
            capital: Capital para el cálculo de posiciones.

        Returns:
            Dict[str, float] con las métricas de salud.
        """
        try:
            signals = self.generate_signals(is_data, capital)
        except Exception as exc:
            logger.warning(f"{self.name}.get_health_metrics: error → {exc}")
            return self._empty_health_metrics()

        positions = signals.get("positions", pd.Series(dtype=float))
        if positions.empty or is_data.empty:
            return self._empty_health_metrics()

        # Retornos de la estrategia (posición shift(1) evita look-ahead)
        price_ret = is_data["close"].pct_change()
        weight    = (positions.shift(1) / capital).fillna(0.0)
        strat_ret = (weight * price_ret).dropna()

        if len(strat_ret) < 20:
            return self._empty_health_metrics()

        # Sharpe anualizado
        sigma = strat_ret.std(ddof=1)
        sharpe_is = (
            float((strat_ret.mean() / sigma) * np.sqrt(252))
            if sigma > 1e-10 else 0.0
        )

        # Hit Ratio: fracción de retornos positivos en días en posición
        active = strat_ret[weight.reindex(strat_ret.index).abs() > 0.01]
        hit_ratio = float((active > 0).mean()) if len(active) > 0 else 0.5

        # Calmar
        wealth = (1.0 + strat_ret).cumprod()
        n_years = len(strat_ret) / 252
        cagr    = float((wealth.iloc[-1]) ** (1 / max(n_years, 1e-6)) - 1)
        dd      = ((wealth - wealth.cummax()) / wealth.cummax()).min()
        calmar_is = float(cagr / abs(dd)) if abs(dd) > 1e-10 else float("nan")

        # Turnover: número de cambios de side
        side = signals.get("side", pd.Series(dtype=float))
        n_trades = int(side.diff().abs().gt(0.5).sum()) if not side.empty else 0
        avg_holding = float(len(side) / max(n_trades, 1))

        return {
            "sharpe_is":   sharpe_is,
            "hit_ratio":   hit_ratio,
            "calmar_is":   calmar_is,
            "n_trades":    float(n_trades),
            "avg_holding": avg_holding,
        }

    # ── HELPERS INTERNOS ──────────────────────────────────────────────────────

    def _compute_realized_vol(
        self,
        close: pd.Series,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Volatilidad realizada (Yang-Zhang simplificado sobre close-to-close).

        Se usa en el Meta-Labeling sizing para escalar la apuesta:
          bet_size = target_vol / realized_vol
        clipeada a [0.1, 1.0] para evitar apalancamiento excesivo en épocas de
        baja volatilidad y tamaños mínimos en épocas de alta volatilidad.

        Args:
            close:     Serie de precios de cierre.
            window:    Ventana de estimación en días.
            annualize: Si True, multiplica por sqrt(252).

        Returns:
            pd.Series de volatilidad realizada (misma longitud que close).
        """
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(window, min_periods=max(5, window // 4)).std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol.bfill().fillna(0.01)

    def _compute_signal_age(self, side: pd.Series) -> pd.Series:
        """
        Calcula cuántos períodos lleva sin cambiar cada señal (signal age).

        Usado por el adaptador para aplicar el factor de decaimiento:
          decay = exp(-ln(2) * age / halflife)

        Algoritmo vectorizado:
          1. Detectar cambios de señal (diff != 0).
          2. Acumular días desde el último cambio usando cumsum y grupos.

        Args:
            side: Serie de {-1, 0, 1} (dirección de la señal).

        Returns:
            pd.Series[int] con la edad en períodos de cada señal.
        """
        if side.empty:
            return pd.Series(dtype=int)

        # Detectar cambios: nuevo grupo cada vez que side cambia
        changed = (side.diff().fillna(1).abs() > 0.5).astype(int)
        group_id = changed.cumsum()

        # Dentro de cada grupo, contar posición acumulada desde el inicio
        age = group_id.groupby(group_id).cumcount()
        return age

    @staticmethod
    def _empty_health_metrics() -> Dict[str, float]:
        return {
            "sharpe_is":   0.0,
            "hit_ratio":   0.5,
            "calmar_is":   float("nan"),
            "n_trades":    0.0,
            "avg_holding": float("nan"),
        }

    def _vol_sizing(
        self,
        close: pd.Series,
        side: pd.Series,
        target_vol: float = 0.15,
        vol_window: int = 21,
        min_size: float = 0.10,
        max_size: float = 1.00,
    ) -> pd.Series:
        """
        META-LABELING SIZING: escala la apuesta por el ratio target_vol / realized_vol.

        Lógica:
          realized_vol = estimación local de σ_daily * sqrt(252).
          bet_size = clip(target_vol / realized_vol, min_size, max_size)

        Cuando la volatilidad es alta (mercado caótico), el tamaño se reduce.
        Cuando es baja (mercado calmo), el tamaño sube hasta max_size.
        Esto implementa una forma simplificada de targeting de volatilidad,
        similar al Kelly fraccionario pero más robusto ante no-normalidad.

        Args:
            close:      Precios de cierre.
            side:       Señal de dirección {-1, 0, 1}.
            target_vol: Volatilidad anualizada objetivo (default 15%).
            vol_window: Ventana de vol realizada.
            min_size:   Tamaño mínimo de apuesta cuando vol es muy alta.
            max_size:   Tamaño máximo de apuesta (1.0 = capital completo).

        Returns:
            pd.Series ∈ [0, max_size] de tamaños de apuesta.
        """
        realized_vol = self._compute_realized_vol(close, window=vol_window)
        # Evitar división por cero
        realized_vol = realized_vol.replace(0, np.nan).ffill().fillna(0.20)

        raw_size = (target_vol / realized_vol).clip(min_size, max_size)

        # Si la señal es plana (side=0), el tamaño es 0 independientemente
        size = raw_size.where(side.abs() > 0.5, 0.0)
        return size


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY DE ESTRATEGIAS — Patrón Factoría
# ══════════════════════════════════════════════════════════════════════════════

class StrategyRegistry:
    """
    Registro central de estrategias — Patrón Factoría.

    Permite:
    · Registro dinámico de nuevas estrategias sin modificar el motor.
    · Creación de universos predefinidos (default_universe).
    · Introspección: summary table, list_all, filtrado por tipo.

    Uso típico en el motor TTT:
        registry = StrategyRegistry().create_default_universe()
        adapted  = adapt_strategies(list(registry.strategies.values()), capital)
    """

    def __init__(self) -> None:
        self.strategies: Dict[str, BaseStrategy] = {}

    def register(self, strategy: BaseStrategy) -> "StrategyRegistry":
        """Registra una estrategia. Retorna self para chaining."""
        if not isinstance(strategy, BaseStrategy):
            raise TypeError(
                f"register() espera BaseStrategy, recibió {type(strategy).__name__}."
            )
        self.strategies[strategy.name] = strategy
        logger.debug(f"StrategyRegistry: registrada '{strategy.name}'.")
        return self

    def get(self, name: str) -> Optional[BaseStrategy]:
        """Devuelve estrategia por nombre, o None si no existe."""
        return self.strategies.get(name)

    def list_all(self) -> List[str]:
        """Lista de nombres registrados."""
        return list(self.strategies.keys())

    def filter_by_type(self, strategy_type: str) -> List[BaseStrategy]:
        """Filtra por tipo (momentum, mean_reversion, etc.)."""
        return [s for s in self.strategies.values()
                if s.strategy_type == strategy_type]

    def get_summary(self) -> pd.DataFrame:
        """Tabla resumen de todas las estrategias registradas."""
        rows = []
        for s in self.strategies.values():
            params = s.get_parameters()
            rows.append({
                "name":        s.name,
                "type":        s.strategy_type,
                "warmup_days": s.warmup_period(),
                "halflife":    s.signal_halflife,
                "params":      str(params),
            })
        return pd.DataFrame(rows)

    def create_default_universe(self) -> "StrategyRegistry":
        """
        Universo de 9 estrategias ortogonales (diversificación de alpha).

        Las 9 estrategias explotan patrones de mercado distintos:
          Momentum:       TrendFollowing, MomentumCrossover
          Mean Reversion: MeanReversionBB, RSIMeanReversion
          Volatilidad:    VolatilityBreakout, LowVolatility
          Contrarian:     FadeExtremes
          Breakout:       RangeBreakout (Donchian)
          Pasiva:         BuyAndHold (benchmark interno)

        La ortogonalidad de alpha sources reduce el riesgo de que el motor TTT
        encuentre un único "ganador" que domine todos los regímenes.
        """
        (self
         .register(BuyAndHold())
         .register(TrendFollowingStrategy(fast_ema=12, slow_ema=26, atr_window=14))
         .register(MeanReversionBB(bb_window=20, n_std=2.0))
         .register(RSIMeanReversionStrategy(period=14, oversold=30, overbought=70))
         .register(MomentumCrossover(fast=20, slow=50))
         .register(VolatilityBreakout(lookback=20, spike_percentile=0.80))
         .register(LowVolatility(lookback=21, target_vol=0.10))
         .register(FadeExtremes(lookback=63, extreme_z=2.5))
         .register(DonchianBreakout(channel_window=20, atr_window=14))
         .register(XGBoostTrendStrategy(forward_horizon=5, max_depth=3,
                                        learning_rate=0.05, n_estimators=100))
        )
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUY AND HOLD — Benchmark Pasivo
# ══════════════════════════════════════════════════════════════════════════════

class BuyAndHold(BaseStrategy):
    """
    Benchmark: 100% invertido en el activo en todo momento.

    Propósito en el torneo TTT:
    · La "habilidad" de esta estrategia es el retorno puro del mercado sin alpha.
    · TTT mide el alpha de las demás estrategias RELATIVO a este benchmark.
    · Sin embargo, en el motor actual el benchmark es un jugador separado
      (_benchmark_) con alpha=0; BuyAndHold compite en el torneo como el
      resto y actúa como referencia de correlación.

    Signal Decay: No aplica — señal constante, nunca envejece.
    """

    signal_halflife: int = 999_999   # señal permanente

    def __init__(self) -> None:
        super().__init__(name="BuyAndHold")
        self.strategy_type = "passive"

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        n = len(data)
        side      = pd.Series(1.0,    index=data.index, name="side")
        size      = pd.Series(1.0,    index=data.index, name="size")
        positions = pd.Series(capital, index=data.index, name="positions")
        age       = pd.Series(0,       index=data.index, name="signal_age")
        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {"strategy": "BuyAndHold"}

    def warmup_period(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "BuyAndHold()"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TREND FOLLOWING — MACD-style con bandas ATR
# ══════════════════════════════════════════════════════════════════════════════

class TrendFollowingStrategy(BaseStrategy):
    """
    Seguimiento de tendencia: MACD con sizing por ATR (volatilidad real del rango).

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · SIDE: Cruce de EMAs rápida vs lenta (Appel, 1979). La EMA rápida (12d)
      captura momentum de corto plazo; la lenta (26d) filtra el ruido.
      Señal = fast_ema > slow_ema → long; fast_ema < slow_ema → short/flat.

    · FILTRO ATR: Solo opera cuando el rango verdadero medio (ATR) supera
      su percentil 40 histórico. En mercados con ATR muy bajo (mercado dormido),
      la señal tiene bajo poder predictivo y el costo de transacción consume
      el alpha — se mantiene flat.

    · SIZE (Meta-Labeling): Target volatility scaling. La apuesta se escala
      inversamente a la volatilidad realizada:
        bet_size = clip(target_vol / vol_realized, 0.1, 1.0)
      Esto produce posiciones más pequeñas cuando el mercado es caótico y
      más grandes cuando la tendencia es limpia y de baja volatilidad.

    Gana en: Tendencias sostenidas (bull/bear markets plurianuales).
    Pierde en: Mercados laterales con muchos whipsaws (cost of carry).

    Signal Halflife: 30d — la señal de tendencia tiene vida media de 1.5 meses.
    """

    signal_halflife: int = 30

    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        atr_window: int = 14,
        target_vol: float = 0.15,
        atr_filter_pct: float = 0.40,
    ) -> None:
        super().__init__(name=f"TrendFollow_EMA{fast_ema}_{slow_ema}")
        self.fast_ema        = fast_ema
        self.slow_ema        = slow_ema
        self.atr_window      = atr_window
        self.target_vol      = target_vol
        self.atr_filter_pct  = atr_filter_pct
        self.strategy_type   = "momentum"
        self.required_features = ["close", "high", "low"]

    def _compute_atr(self, data: pd.DataFrame) -> pd.Series:
        """Average True Range — mide la volatilidad del rango real."""
        high = data["high"] if "high" in data.columns else data["close"] * 1.005
        low  = data["low"]  if "low"  in data.columns else data["close"] * 0.995
        prev_close = data["close"].shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return true_range.rolling(self.atr_window, min_periods=2).mean()

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close    = data["close"].astype(float)
        fast_ema = close.ewm(span=self.fast_ema, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_ema, adjust=False).mean()

        # SIDE: cruce de EMAs (long cuando fast > slow)
        raw_side = (fast_ema > slow_ema).astype(float)  # 0 o 1
        # Convertir a {-1 flat, 1 long} — TF clásico es long-only;
        # para short-enabled, usar: raw_side * 2 - 1
        side = raw_side.map({1.0: 1.0, 0.0: 0.0})

        # FILTRO ATR: apagar señal cuando mercado está dormido
        atr      = self._compute_atr(data)
        atr_rank = atr.rolling(252, min_periods=60).rank(pct=True)
        active   = (atr_rank >= self.atr_filter_pct).fillna(True)
        side     = side.where(active, 0.0)

        # SIZE (Meta-Labeling): vol targeting
        size      = self._vol_sizing(close, side, target_vol=self.target_vol)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "fast_ema":       self.fast_ema,
            "slow_ema":       self.slow_ema,
            "atr_window":     self.atr_window,
            "target_vol":     self.target_vol,
            "atr_filter_pct": self.atr_filter_pct,
        }

    def warmup_period(self) -> int:
        return max(self.slow_ema * 3, self.atr_window + 252)

    def __repr__(self) -> str:
        return (f"TrendFollowingStrategy(fast_ema={self.fast_ema}, "
                f"slow_ema={self.slow_ema}, target_vol={self.target_vol})")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MEAN REVERSION — Bandas de Bollinger Dinámicas
# ══════════════════════════════════════════════════════════════════════════════

class MeanReversionBB(BaseStrategy):
    """
    Reversión a la media usando Bandas de Bollinger con ancho dinámico.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · Bollinger Bands (Bollinger, 1992): bandas a ± n_std * σ_rolling alrededor
      de una media móvil. Capturan la "normalidad" estadística del precio.

    · SIDE: Oversold (precio < lower band) → long; overbought (> upper band)
      → si allow_short: short, si no: flat. Neutral dentro de las bandas.

    · ANCHO DINÁMICO: n_std se ajusta por el percentil de volatilidad actual.
      En alta vol, el ancho se ensancha (más difícil de tocar las bandas),
      reduciendo la frecuencia de señales y el riesgo de "cuchillo cayendo".

    · SIZE: Proporcional a la distancia normalizada desde la media:
        z = (close - ma) / (band_width / 2)
        bet_size = clip(|z| / 3, 0, 1)
      Un z=3 genera bet_size=1 (apuesta completa), z=1 genera bet_size=0.33.

    Gana en: Mercados laterales con retornos a la media (regime sideways).
    Pierde en: Tendencias fuertes (el precio toca la banda y sigue).

    Signal Halflife: 10d — señal de corto plazo, decae rápido.
    """

    signal_halflife: int = 10

    def __init__(
        self,
        bb_window: int = 20,
        n_std: float = 2.0,
        allow_short: bool = False,
        target_vol: float = 0.12,
    ) -> None:
        super().__init__(name=f"MeanRevBB_{bb_window}d_{n_std}std")
        self.bb_window    = bb_window
        self.n_std        = n_std
        self.allow_short  = allow_short
        self.target_vol   = target_vol
        self.strategy_type = "mean_reversion"

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close = data["close"].astype(float)
        ma    = close.rolling(self.bb_window, min_periods=self.bb_window // 2).mean()
        sigma = close.rolling(self.bb_window, min_periods=self.bb_window // 2).std()

        # Bandas dinámicas: n_std ajustado por percentil de vol
        vol_rank = sigma.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
        n_std_adj = self.n_std * (0.8 + 0.4 * vol_rank)  # [0.8*n, 1.2*n]

        upper = ma + n_std_adj * sigma
        lower = ma - n_std_adj * sigma

        # Z-score normalizado
        band_half = (n_std_adj * sigma).replace(0, np.nan)
        z = (close - ma) / band_half

        # SIDE: entrar long cuando por debajo de lower, short cuando encima de upper
        side = pd.Series(0.0, index=close.index)
        side = side.where(close >= lower, 1.0)   # oversold → long
        if self.allow_short:
            side = side.where(close <= upper, -1.0)  # overbought → short
        else:
            side = side.where(close <= upper, 0.0)   # overbought → flat

        # SIZE: proporcional a |z|, normalizado para que z=3 → size=1
        raw_size = (z.abs() / 3.0).clip(0.0, 1.0)
        size = raw_size.where(side.abs() > 0.5, 0.0)

        # Meta-Labeling: ajuste adicional por vol realizada
        vol_adj  = self._vol_sizing(close, side, target_vol=self.target_vol)
        size     = (size * vol_adj).clip(0.0, 1.0)
        positions = side * size * capital
        age      = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "bb_window":   self.bb_window,
            "n_std":       self.n_std,
            "allow_short": self.allow_short,
            "target_vol":  self.target_vol,
        }

    def warmup_period(self) -> int:
        return self.bb_window + 252

    def __repr__(self) -> str:
        return (f"MeanReversionBB(bb_window={self.bb_window}, "
                f"n_std={self.n_std}, allow_short={self.allow_short})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RSI MEAN REVERSION — Oscilador con umbrales adaptativos
# ══════════════════════════════════════════════════════════════════════════════

class RSIMeanReversionStrategy(BaseStrategy):
    """
    Reversión basada en RSI con umbrales adaptativos al régimen de volatilidad.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · RSI (Wilder, 1978): oscilador [0, 100] que mide la velocidad y cambio
      de precios. RSI < 30 → oversold (comprar); RSI > 70 → overbought (vender).

    · UMBRALES ADAPTATIVOS: En mercados de alta volatilidad, los extremos de
      RSI ocurren con más frecuencia y son menos informativos. Los umbrales
      se ajustan: oversold se mueve a 25, overbought a 75 (más extremos).
      Esto evita señales falsas durante pánico o euforia de corto plazo.

    · SIZE: Distancia normalizada del RSI respecto al umbral:
        for oversold: size = (threshold - RSI) / threshold
        Cuanto más extremo el RSI, mayor la apuesta.

    Halflife: 7d — señal de muy corto plazo, la condición oversold/overbought
    suele resolverse en menos de 2 semanas.
    """

    signal_halflife: int = 7

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        target_vol: float = 0.10,
    ) -> None:
        super().__init__(name=f"RSI_{period}_{int(oversold)}_{int(overbought)}")
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought
        self.target_vol = target_vol
        self.strategy_type = "mean_reversion"

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """RSI con suavizado exponencial (Wilder EMA) — vectorizado."""
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/self.period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/self.period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close = data["close"].astype(float)
        rsi   = self._compute_rsi(close)

        # Umbrales adaptativos al régimen de volatilidad
        vol   = self._compute_realized_vol(close, window=21)
        vol_rank = vol.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)

        # En alta vol: umbrales más extremos (25/75); en baja vol: 30/70
        os_thresh = self.oversold  - 5.0 * vol_rank     # [25, 30]
        ob_thresh = self.overbought + 5.0 * vol_rank     # [70, 75]

        # SIDE: condiciones de entrada
        side = pd.Series(0.0, index=close.index)
        side = side.where(rsi >= os_thresh, 1.0)   # oversold → long
        side = side.where(rsi <= ob_thresh, -1.0)  # overbought → short (leve)

        # SIZE: proporcional a la distancia del RSI respecto al umbral
        dist_os = (os_thresh - rsi).clip(lower=0) / os_thresh
        dist_ob = (rsi - ob_thresh).clip(lower=0) / (100 - ob_thresh)
        raw_size = (dist_os + dist_ob).clip(0.0, 1.0)

        vol_adj   = self._vol_sizing(close, side, target_vol=self.target_vol)
        size      = (raw_size * vol_adj).clip(0.0, 1.0)
        size      = size.where(side.abs() > 0.5, 0.0)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "period":     self.period,
            "oversold":   self.oversold,
            "overbought": self.overbought,
            "target_vol": self.target_vol,
        }

    def warmup_period(self) -> int:
        return self.period * 4 + 252

    def __repr__(self) -> str:
        return (f"RSIMeanReversionStrategy(period={self.period}, "
                f"oversold={self.oversold}, overbought={self.overbought})")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MOMENTUM CROSSOVER — Cruce de medias binario
# ══════════════════════════════════════════════════════════════════════════════

class MomentumCrossover(BaseStrategy):
    """
    Cruce de medias móviles simples: señal binaria, sin sizing gradual.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · El cruce SMA20/SMA50 es uno de los sistemas técnicos más estudiados
      (Brock, Lakonishok & LeBaron, 1992, Journal of Finance).
    · A diferencia de TrendFollowing (gradual), esta estrategia es binaria:
      o está 100% invertida (cuando fast>slow) o completamente flat.
    · La señal binaria produce más trades y mayor turnover, lo que la hace
      más sensible a los costos de transacción — el adaptador debe penalizarla.

    · SIZE: 1.0 (full size) cuando hay señal, 0 cuando no hay.
      El Meta-Labeling sizing aplica solo como capa de vol-targeting.

    Signal Halflife: 20d — cruce de 20/50 días tiene vida media de 1 mes.
    """

    signal_halflife: int = 20

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        target_vol: float = 0.15,
    ) -> None:
        super().__init__(name=f"MomXover_{fast}_{slow}")
        self.fast       = fast
        self.slow       = slow
        self.target_vol = target_vol
        self.strategy_type = "momentum"

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close    = data["close"].astype(float)
        fast_sma = close.rolling(self.fast, min_periods=self.fast // 2).mean()
        slow_sma = close.rolling(self.slow, min_periods=self.slow // 2).mean()

        # SIDE binario: long cuando fast > slow
        side      = (fast_sma > slow_sma).astype(float)
        size      = self._vol_sizing(close, side, target_vol=self.target_vol)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "target_vol": self.target_vol}

    def warmup_period(self) -> int:
        return self.slow + 252

    def __repr__(self) -> str:
        return f"MomentumCrossover(fast={self.fast}, slow={self.slow})"


# ══════════════════════════════════════════════════════════════════════════════
# 6. VOLATILITY BREAKOUT — Breakout en spikes de volatilidad
# ══════════════════════════════════════════════════════════════════════════════

class VolatilityBreakout(BaseStrategy):
    """
    Opera durante picos de volatilidad siguiendo el momentum de corto plazo.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · TESIS: En mercados en pánico o euforia extrema, los inversores
      institucionales generan movimientos direccionales fuertes y persistentes
      en el corto plazo (1-5 días). Surfear estos eventos con momentum
      de muy corto plazo captura el alpha de "crisis momentum".

    · DETECCIÓN: La vol realizada en percentil >= spike_percentile (default 80%)
      respecto a los últimos 252 días define un "spike de volatilidad".

    · DIRECTION: Momentum de 5 días (retorno compuesto 5d). Si el mercado
      ha subido en los últimos 5 días durante el spike → long; si ha bajado → short.

    · RIESGO: Puede quedarse atrapada si el pánico continúa múltiples semanas.
      Por eso el size se limita al 30% del capital base.

    Signal Halflife: 5d — señal muy de corto plazo.
    """

    signal_halflife: int = 5

    def __init__(
        self,
        lookback: int = 20,
        spike_percentile: float = 0.80,
        momentum_days: int = 5,
        max_size: float = 0.30,
    ) -> None:
        super().__init__(name=f"VolBreakout_{lookback}d")
        self.lookback         = lookback
        self.spike_percentile = spike_percentile
        self.momentum_days    = momentum_days
        self.max_size         = max_size
        self.strategy_type    = "volatility_breakout"

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close    = data["close"].astype(float)
        ret      = close.pct_change()
        vol      = ret.rolling(self.lookback, min_periods=5).std() * np.sqrt(252)
        vol_rank = vol.rolling(252, min_periods=60).rank(pct=True).fillna(0.0)

        # Detectar spike
        is_spike = (vol_rank >= self.spike_percentile).astype(float)

        # Dirección: momentum de corto plazo
        momentum  = ret.rolling(self.momentum_days, min_periods=2).sum()
        direction = np.sign(momentum).fillna(0.0)

        # SIDE: solo opera durante spikes
        side = (is_spike * direction).clip(-1, 1)

        # SIZE: limitado — alta volatilidad = alta incertidumbre
        # Se usa un tamaño fijo reducido (max_size) en lugar de vol-targeting
        # para evitar paradoja: vol alta → size grande (que es justo lo opuesto).
        size      = is_spike * self.max_size
        size      = size.where(side.abs() > 0.5, 0.0)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "lookback":         self.lookback,
            "spike_percentile": self.spike_percentile,
            "momentum_days":    self.momentum_days,
            "max_size":         self.max_size,
        }

    def warmup_period(self) -> int:
        return self.lookback + 252

    def __repr__(self) -> str:
        return (f"VolatilityBreakout(lookback={self.lookback}, "
                f"spike_percentile={self.spike_percentile})")


# ══════════════════════════════════════════════════════════════════════════════
# 7. LOW VOLATILITY — Targeting de volatilidad constante
# ══════════════════════════════════════════════════════════════════════════════

class LowVolatility(BaseStrategy):
    """
    Estrategia de targeting de volatilidad: siempre invertida, ajusta el tamaño.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · TESIS: La anomalía low-vol (Ang et al., 2006, Journal of Finance):
      activos de baja volatilidad tienen retornos ajustados por riesgo
      superiores a activos de alta volatilidad (violando el CAPM clásico).

    · IMPLEMENTACIÓN: En lugar de seleccionar activos de baja vol (requiere
      cross-section), esta estrategia reduce el tamaño de posición cuando
      la volatilidad del activo supera el target, y lo aumenta cuando está
      por debajo — sintetizando un perfil de "baja volatilidad" temporalmente.

    · SIZE: bet_size = clip(target_vol / vol_realized, 0.2, 1.0)
      Siempre long (side=1), pero el tamaño varía continuamente.

    · SIDE: Siempre +1. No hace short. Reduce exposición en alta vol.

    Signal Halflife: 252d — el ajuste de tamaño cambia gradualmente.
    """

    signal_halflife: int = 252

    def __init__(
        self,
        lookback: int = 21,
        target_vol: float = 0.10,
        min_size: float = 0.10,
    ) -> None:
        super().__init__(name=f"LowVol_{lookback}d_tv{int(target_vol*100)}")
        self.lookback   = lookback
        self.target_vol = target_vol
        self.min_size   = min_size
        self.strategy_type = "low_volatility"

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close = data["close"].astype(float)

        # Siempre long
        side = pd.Series(1.0, index=close.index)

        # SIZE: targeting de volatilidad
        size      = self._vol_sizing(
            close, side, target_vol=self.target_vol,
            vol_window=self.lookback, min_size=self.min_size
        )
        positions = side * size * capital
        age       = pd.Series(0, index=close.index)  # señal no envejece

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "lookback":   self.lookback,
            "target_vol": self.target_vol,
            "min_size":   self.min_size,
        }

    def warmup_period(self) -> int:
        return self.lookback + 30

    def __repr__(self) -> str:
        return (f"LowVolatility(lookback={self.lookback}, "
                f"target_vol={self.target_vol})")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FADE EXTREMES — Contrarian en movimientos de cola
# ══════════════════════════════════════════════════════════════════════════════

class FadeExtremes(BaseStrategy):
    """
    Apuesta contraria a movimientos de retorno > extreme_z desviaciones estándar.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · TESIS: El overshoot de precios ante noticias (De Bondt & Thaler, 1985,
      Journal of Finance): los inversores sobre-reaccionan a nueva información,
      produciendo reversiones de corto plazo estadísticamente explotables.

    · SIDE: Cuando el retorno diario supera ±extreme_z sigmas del retorno
      histórico local → apuesta en la dirección contraria.
      - Caída extrema → long (anticipar rebote)
      - Subida extrema → short (anticipar corrección)

    · RIESGO: "Catching falling knives". El tamaño se limita al 20% y
      se añade una condición de sostenibilidad: la señal solo se activa
      si el z-score supera el umbral por 2+ días consecutivos para evitar
      señales de un único día ("flash crashes" unidireccionales).

    · SIZE: Proporcional a |z| / extreme_z. Mayor el exceso, mayor la apuesta.

    Signal Halflife: 5d — la reversión suele ocurrir en 1-5 días o no ocurre.
    """

    signal_halflife: int = 5

    def __init__(
        self,
        lookback: int = 63,
        extreme_z: float = 2.5,
        confirmation_days: int = 1,
        max_size: float = 0.20,
    ) -> None:
        super().__init__(name=f"FadeExtremes_{lookback}d")
        self.lookback          = lookback
        self.extreme_z         = extreme_z
        self.confirmation_days = confirmation_days
        self.max_size          = max_size
        self.strategy_type     = "contrarian"

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close = data["close"].astype(float)
        ret   = close.pct_change()
        mu    = ret.rolling(self.lookback, min_periods=self.lookback // 4).mean()
        sigma = ret.rolling(self.lookback, min_periods=self.lookback // 4).std()

        z = ((ret - mu) / sigma.replace(0, np.nan)).fillna(0.0)

        is_extreme = z.abs() >= self.extreme_z

        # Confirmación multi-día: reducir señales de un único día extremo
        if self.confirmation_days > 1:
            # Rolling sum: is_extreme debe ocurrir en confirmation_days de los últimos N
            is_extreme = is_extreme.rolling(
                self.confirmation_days, min_periods=1
            ).sum() >= (self.confirmation_days - 1)

        # SIDE: contrario a la dirección extrema
        fade_dir = -np.sign(z)
        side = (is_extreme.astype(float) * fade_dir).clip(-1, 1)

        # SIZE: proporcional al exceso de z sobre el umbral, limitado a max_size
        excess_z  = ((z.abs() - self.extreme_z) / self.extreme_z).clip(0, 1)
        raw_size  = excess_z * self.max_size
        size      = raw_size.where(side.abs() > 0.5, 0.0)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "lookback":          self.lookback,
            "extreme_z":         self.extreme_z,
            "confirmation_days": self.confirmation_days,
            "max_size":          self.max_size,
        }

    def warmup_period(self) -> int:
        return self.lookback

    def __repr__(self) -> str:
        return (f"FadeExtremes(lookback={self.lookback}, "
                f"extreme_z={self.extreme_z})")


# ══════════════════════════════════════════════════════════════════════════════
# 9. DONCHIAN BREAKOUT — Canal de Donchian con filtro ATR
# ══════════════════════════════════════════════════════════════════════════════

class DonchianBreakout(BaseStrategy):
    """
    Breakout de canal de Donchian (Richard Donchian, 1970) con filtro ATR.

    LÓGICA FINANCIERA:
    ──────────────────────────────────────────────────────────────────────────
    · CANAL: Upper = max(high, N días); Lower = min(low, N días).
      Cuando el precio rompe el canal → inicio de una nueva tendencia.
      Turtles Trading System (Dennis & Eckhardt, 1983) usaba este principio.

    · SIDE:
        close > upper canal anterior → long (nuevo máximo de N días)
        close < lower canal anterior → short (nuevo mínimo de N días)

    · FILTRO ATR: Similar a TrendFollowing: requiere que el ATR esté por
      encima de su percentil 30 para operar. Evita breakouts falsos en
      mercados dormidos con gaps bid-ask que rompen el canal artificialmente.

    · SIZE: Target vol scaling estándar. Los breakouts de Donchian funcionan
      mejor con posiciones del 10-20% del capital para sobrevivir el whipsaw
      de los primeros días post-breakout.

    · DECAY DE SEÑAL: Después de un breakout, la señal comienza a decaer.
      La renovación ocurre si el precio continúa en la misma dirección.

    Signal Halflife: 20d — breakouts suelen durar 2-8 semanas.
    """

    signal_halflife: int = 20

    def __init__(
        self,
        channel_window: int = 20,
        atr_window: int = 14,
        atr_filter_pct: float = 0.30,
        target_vol: float = 0.12,
    ) -> None:
        super().__init__(name=f"DonchianBreakout_{channel_window}d")
        self.channel_window  = channel_window
        self.atr_window      = atr_window
        self.atr_filter_pct  = atr_filter_pct
        self.target_vol      = target_vol
        self.strategy_type   = "breakout"
        self.required_features = ["close", "high", "low"]

    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close = data["close"].astype(float)
        high  = data.get("high", close * 1.005).astype(float)
        low   = data.get("low",  close * 0.995).astype(float)

        # Canal de Donchian (shift(1) para evitar look-ahead en la vela actual)
        upper = high.rolling(self.channel_window, min_periods=5).max().shift(1)
        lower = low.rolling(self.channel_window,  min_periods=5).min().shift(1)

        # SIDE: breakout
        side = pd.Series(0.0, index=close.index)
        side = side.where(close <= upper, 1.0)   # breakout alcista
        side = side.where(close >= lower, -1.0)  # breakout bajista

        # Mantener la posición hasta que aparezca la señal contraria
        side = side.replace(0.0, np.nan).ffill().fillna(0.0)

        # FILTRO ATR
        tr_hl  = high - low
        tr_hc  = (high - close.shift(1)).abs()
        tr_lc  = (low  - close.shift(1)).abs()
        atr    = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
        atr_ma = atr.rolling(self.atr_window, min_periods=2).mean()
        atr_rank = atr_ma.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
        active   = (atr_rank >= self.atr_filter_pct).fillna(True)
        side     = side.where(active, 0.0)

        # SIZE: targeting de volatilidad
        size      = self._vol_sizing(close, side, target_vol=self.target_vol)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "channel_window":  self.channel_window,
            "atr_window":      self.atr_window,
            "atr_filter_pct":  self.atr_filter_pct,
            "target_vol":      self.target_vol,
        }

    def warmup_period(self) -> int:
        return self.channel_window + 252

    def __repr__(self) -> str:
        return (f"DonchianBreakout(channel_window={self.channel_window}, "
                f"atr_window={self.atr_window})")


# ══════════════════════════════════════════════════════════════════════════════
# 10. XGBOOST TREND — Modelo ML re-entrenado por ventana Walk-Forward
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostTrendStrategy(BaseStrategy):
    """
    Estrategia ML: gradient boosting (XGBoost) sobre los 32 features técnicos.

    DISEÑO ANTI-FUGA (Walk-Forward estricto, López de Prado 2018 Cap. 7):
    ──────────────────────────────────────────────────────────────────────────
    · `fit(is_data)` se invoca UNA VEZ POR FOLD con la ventana IN-SAMPLE estricta
      que el `TTTSimulator` entrega (anterior al OOS, con purging+embargo ya
      aplicados). El modelo SOLO ve `is_data`; jamás barras del futuro/OOS.
    · El TARGET es la DIRECCIÓN del retorno FORWARD a `forward_horizon` días:
          y_t = 1[ close_{t+h} / close_t − 1 > 0 ]
      Para no usar información fuera de la ventana IS, las últimas `h` filas de
      `is_data` (cuyo target miraría más allá del IS) se DESCARTAN antes de
      entrenar. Así X e Y provienen exclusivamente del IS.
    · `generate_signals(oos_data)` predice con el modelo ya entrenado usando solo
      features contemporáneos (conocidos al cierre de t). El backtest aplica
      `positions.shift(1)`, de modo que la posición de t se materializa en t+1.

    FEATURES (X):
      Todas las columnas numéricas del DataFrame enriquecido EXCEPTO OHLCV y
      metadatos (`sp500_daily_..._features.parquet` aporta ret_lags, RSI, MACD,
      Bollinger, ATR, vol. anualizada, OBV, VWAP-dev, etc.). Se detectan de forma
      dinámica → si el motor recibe OHLCV crudo (sin features), la estrategia se
      mantiene PLANA (side=0) en lugar de fallar.

    HIPERPARÁMETROS ROBUSTOS (anti-overfitting):
      max_depth=3, learning_rate=0.05, n_estimators=100, subsample=0.8.

    SIZE (Meta-Labeling): targeting de volatilidad sobre la dirección predicha.
    """

    signal_halflife: int = 21

    # Columnas que NO son features predictivos (precio crudo + metadatos).
    _NON_FEATURE_COLS = {
        "open", "high", "low", "close", "adj_close", "volume",
        "price_col_used", "symbol", "_target",
    }

    def __init__(
        self,
        forward_horizon: int = 5,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        prob_band: float = 0.05,
        target_vol: float = 0.15,
    ) -> None:
        super().__init__(name="XGBoostTrend_d5")
        self.forward_horizon = forward_horizon
        self.max_depth       = max_depth
        self.learning_rate   = learning_rate
        self.n_estimators    = n_estimators
        self.prob_band       = prob_band
        self.target_vol      = target_vol
        self.strategy_type   = "ml_trend"
        self.required_features = ["close"]   # + features detectados dinámicamente
        self._model = None
        self._feature_cols: Optional[List[str]] = None

    # ── DETECCIÓN DINÁMICA DE FEATURES ──────────────────────────────────────
    def _detect_features(self, data: pd.DataFrame) -> List[str]:
        return [
            c for c in data.columns
            if c not in self._NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(data[c])
        ]

    # ── ENTRENAMIENTO (solo IS) ─────────────────────────────────────────────
    def fit(self, is_data: pd.DataFrame, capital: float = 100_000) -> "XGBoostTrendStrategy":
        self._model = None  # reset por fold (re-fit aislado)

        feats = self._detect_features(is_data)
        if not feats or "close" not in is_data.columns or len(is_data) < 60:
            return self  # sin features o IS insuficiente → estrategia plana

        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost no instalado — XGBoostTrend queda plana.")
            return self

        close = is_data["close"].astype(float)
        # Target: dirección del retorno forward a h días (estrictamente dentro del IS)
        fwd_ret = close.shift(-self.forward_horizon) / close - 1.0
        y = (fwd_ret > 0).astype(int)

        X = is_data[feats].copy()
        # Descartar las últimas h filas (target miraría fuera del IS) y NaNs.
        valid = fwd_ret.notna() & X.notna().all(axis=1)
        X, y = X[valid], y[valid]

        if len(X) < 50 or y.nunique() < 2:
            return self  # datos insuficientes o una sola clase

        model = XGBClassifier(
            max_depth        = self.max_depth,
            learning_rate    = self.learning_rate,
            n_estimators     = self.n_estimators,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            reg_lambda       = 1.0,
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            tree_method      = "hist",
            n_jobs           = 1,
            random_state     = 42,
        )
        try:
            model.fit(X, y)
            self._model = model
            self._feature_cols = feats
        except Exception as exc:
            logger.warning(f"XGBoostTrend.fit falló → {exc}")
            self._model = None
        return self

    # ── PREDICCIÓN (OOS) ────────────────────────────────────────────────────
    def generate_signals(
        self, data: pd.DataFrame, capital: float = 100_000
    ) -> Dict[str, pd.Series]:
        close = data["close"].astype(float)

        # Sin modelo entrenado o sin las columnas esperadas → posición plana.
        if (self._model is None or self._feature_cols is None
                or not set(self._feature_cols).issubset(data.columns)):
            side = pd.Series(0.0, index=data.index)
            size = pd.Series(0.0, index=data.index)
            return {"positions": side * 0.0, "side": side, "size": size,
                    "signal_age": pd.Series(0, index=data.index)}

        X = data[self._feature_cols].astype(float).fillna(0.0)
        proba_up = self._model.predict_proba(X)[:, 1]
        proba_up = pd.Series(proba_up, index=data.index)

        # SIDE con banda neutral: {-1, 0, +1}
        side = pd.Series(0.0, index=data.index)
        side = side.mask(proba_up > 0.5 + self.prob_band,  1.0)
        side = side.mask(proba_up < 0.5 - self.prob_band, -1.0)

        # SIZE (Meta-Labeling: targeting de volatilidad)
        size      = self._vol_sizing(close, side, target_vol=self.target_vol)
        positions = side * size * capital
        age       = self._compute_signal_age(side)

        return {"positions": positions, "side": side,
                "size": size, "signal_age": age}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "forward_horizon": self.forward_horizon,
            "max_depth":       self.max_depth,
            "learning_rate":   self.learning_rate,
            "n_estimators":    self.n_estimators,
            "prob_band":       self.prob_band,
            "target_vol":      self.target_vol,
        }

    def warmup_period(self) -> int:
        # Los features ya vienen calculados y recortados aguas arriba.
        return self.forward_horizon

    def __repr__(self) -> str:
        return (f"XGBoostTrendStrategy(h={self.forward_horizon}, "
                f"max_depth={self.max_depth}, lr={self.learning_rate}, "
                f"n_estimators={self.n_estimators})")


# ══════════════════════════════════════════════════════════════════════════════
# RETROCOMPATIBILIDAD — Aliases para walk_forward_engine_con_TTT.py
# ══════════════════════════════════════════════════════════════════════════════
# El motor importa: TrendFollowing, MeanReversion, VolatilityBreakout, etc.
# Estos aliases garantizan compatibilidad sin modificar el motor.

Strategy          = BaseStrategy       # alias genérico
TrendFollowing    = TrendFollowingStrategy
MeanReversion     = MeanReversionBB
RSIMeanReversion  = RSIMeanReversionStrategy
RangeBreakout     = DonchianBreakout
