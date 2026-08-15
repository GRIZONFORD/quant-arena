# =============================================================================
# FILE: quant_arena/backtesting/motor.py
# Motor de simulación histórica walk-forward estrictamente causal
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quant_arena.backtesting.crowding import ArenaCrowding
from quant_arena.backtesting.position_risk import TakeProfitATRRule, calcular_atr_pct, retornos_periodo_con_riesgo
from quant_arena.backtesting.risk_overlay import RiskOverlay
from quant_arena.core.abstracciones import AbstractJuez, MetricasResultado
from quant_arena.core.excepciones import SizingError
from quant_arena.core.interfaces_riesgo import AbstractPositionSizer
from quant_arena.metricas.filtros import KalmanSignalFilter
from quant_arena.metricas.performance_metrics import PerformanceMetrics
from quant_arena.zoo.base_estrategia import ZooManager

logger = logging.getLogger(__name__)


# =============================================================================
# Contenedor de resultados
# =============================================================================

@dataclass
class ResultadoBacktest:
    """
    Contenedor completo de resultados del backtest walk-forward.

    Toda la información temporal está en frecuencia diaria para las series
    de retornos y en frecuencia de rebalanceo para métricas y pesos del Juez.
    """

    retornos_meta:          pd.Series                   # Retorno diario del meta-portafolio
    retornos_estrategias:   pd.DataFrame                # [fecha x estrategia] retornos diarios
    pesos_juez:             pd.DataFrame                # [fecha_rebalanceo x estrategia]
    pesos_portafolio:       Dict[str, pd.DataFrame]     # {estrategia: [fecha_rebalanceo x ticker]}
    metricas_rolling:       pd.DataFrame                # [fecha_rebalanceo x estrategia] bruto
    metricas_kalman:        pd.DataFrame                # [fecha_rebalanceo x estrategia] filtrado
    fechas_rebalanceo:      pd.DatetimeIndex
    metrica_usada:          str

    # ------------------------------------------------------------------
    # Propiedades derivadas de análisis
    # ------------------------------------------------------------------

    def equity_curve(self) -> pd.Series:
        """Curva de riqueza acumulada del meta-portafolio (NAV normalizado a 1)."""
        return (1.0 + self.retornos_meta.fillna(0.0)).cumprod()

    def equity_curves_estrategias(self) -> pd.DataFrame:
        """Curvas de riqueza individuales de cada estrategia del Zoo."""
        return (1.0 + self.retornos_estrategias.fillna(0.0)).cumprod()

    def pesos_portafolio_diarios(
        self,
        nombre: str,
        fechas_diarias: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Expande los pesos de una estrategia (frecuencia de rebalanceo) a
        frecuencia diaria mediante forward-fill.

        Los pesos del rebalanceo en t_k se aplican a todos los días (t_k, t_{k+1}].

        Args:
            nombre:         Nombre de la estrategia.
            fechas_diarias: Índice diario destino (ej. retornos_meta.index).

        Returns:
            DataFrame [fechas_diarias x ticker] con pesos forward-filled.
        """
        snap = self.pesos_portafolio.get(nombre, pd.DataFrame())
        if snap.empty:
            return pd.DataFrame(index=fechas_diarias)
        union_idx = snap.index.union(fechas_diarias)
        return snap.reindex(union_idx).ffill().reindex(fechas_diarias)

    def resumen_estadistico(
        self,
        metricas: PerformanceMetrics,
        benchmark: pd.Series,
    ) -> pd.DataFrame:
        """
        Tabla de KPIs finales para el meta-portafolio y cada estrategia.

        Returns:
            DataFrame [estrategia x metrica] con todos los KPIs institucionales.
        """
        filas: List[Dict] = []

        def _calcular_fila(nombre: str, retornos: pd.Series) -> Optional[Dict]:
            ret = retornos.dropna()
            bench = benchmark.reindex(ret.index).dropna()
            ret_al, bench_al = ret.align(bench, join='inner')
            if len(ret_al) < 20:
                return None
            try:
                m = metricas.calcular_todas(ret_al, bench_al)
                return {'estrategia': nombre, **m.to_dict()}
            except ValueError:
                return None

        fila_meta = _calcular_fila('META_PORTFOLIO', self.retornos_meta)
        if fila_meta:
            filas.append(fila_meta)

        for col in self.retornos_estrategias.columns:
            fila = _calcular_fila(col, self.retornos_estrategias[col])
            if fila:
                filas.append(fila)

        if not filas:
            return pd.DataFrame()

        return pd.DataFrame(filas).set_index('estrategia').round(4)


# =============================================================================
# Motor principal
# =============================================================================

class BacktestEngine:
    """
    Motor de simulación histórica walk-forward estrictamente causal.

    Orquesta el ciclo completo en cada fecha de rebalanceo t_k:

        1. Acumula retornos del período (t_{k-1}, t_k] con pesos de t_{k-1}.
        2. Calcula la métrica rodante sobre retornos acumulados hasta t_k.
        3. Aplica KalmanSignalFilter al historial de métricas hasta t_k.
        4. Registra la señal filtrada en TTTJuez y actualiza habilidades.
        5. Obtiene pesos meta del Juez para el período siguiente.
        6. Genera nuevos pesos de portafolio (ticker-level) con datos hasta t_k.

    Garantía de causalidad absoluta:
        En ningún paso se accede a datos con índice de fecha posterior a t_k.
        Esta garantía opera en tres niveles:
          (a) Filtrado del DataFrame antes de llamar a generar_señales().
          (b) Retornos del período k usan SOLO pesos generados en t_{k-1}.
          (c) TTTJuez recibe solo métricas calculadas con historial hasta t_k.
    """

    def __init__(
        self,
        zoo: ZooManager,
        metricas: PerformanceMetrics,
        juez: AbstractJuez,
        datos: pd.DataFrame,
        benchmark: pd.Series,
        kalman_config: Optional[Dict[str, float]] = None,
        ventana_metricas: int = 63,
        metrica_ranking: str = 'sharpe',
        risk_overlay: Optional[RiskOverlay] = None,
        position_sizer: Optional[AbstractPositionSizer] = None,
        cap_bruto_exposicion: float = 1.0,
        tp_rule: Optional[TakeProfitATRRule] = None,
        datos_ohlc: Optional[Dict[str, pd.DataFrame]] = None,
        atr_window: int = 14,
        crowding: Optional[ArenaCrowding] = None,
        aum_total: float = 1.0,
    ) -> None:
        """
        Args:
            zoo:              ZooManager con las estrategias activas del Zoo.
            metricas:         Calculadora de KPIs (PerformanceMetrics).
            juez:             Juez TTT para estimación de habilidad latente.
            datos:            DataFrame de precios ajustados [fecha x ticker].
                              Debe tener DatetimeIndex con frecuencia diaria.
            benchmark:        Serie de retornos diarios del S&P 500.
                              Debe compartir el índice de fechas con `datos`.
            kalman_config:    Parámetros opcionales del KalmanSignalFilter:
                              {'Q': float, 'R': float, 'P0': float}.
            ventana_metricas: Días hábiles de la ventana rodante para métricas.
                              También es el mínimo de observaciones para activar TTT.
            metrica_ranking:  Campo de MetricasResultado para ranking en TTT
                              (ej. 'sharpe', 'sortino', 'information_ratio').
            risk_overlay:     RiskOverlay opcional (target-vol + trailing stop
                              global) aplicado a la exposición de CADA
                              estrategia entre rebalanceos, usando el track
                              record propio de esa estrategia (causal: solo
                              el fold recién cerrado). None (default) = motor
                              sin gestión de riesgo, comportamiento idéntico
                              al de antes de esta integración.
            position_sizer:   AbstractPositionSizer opcional (ej.
                              KellyBayesianSizer) que reemplaza
                              `TTTJuez.pesos_asignacion()` (pesos que suman 1)
                              por exposición absoluta justificada por el edge
                              realizado y la incertidumbre del posterior TTT.
                              None (default) = se usa `pesos_asignacion()`
                              sin cambios.
            cap_bruto_exposicion: Cota de exposición bruta agregada (suma de
                              |peso| sobre todas las estrategias) cuando
                              `position_sizer` está activo. Renormaliza si se
                              excede. Ignorado si `position_sizer` es None.
            tp_rule:          TakeProfitATRRule opcional para recorte parcial
                              de exposición intra-período. Requiere
                              `datos_ohlc`. None (default) = retornos de
                              período vectorizados sin path-dependency
                              (idéntico al comportamiento previo).
            datos_ohlc:       {ticker: DataFrame OHLCV en minúsculas} para el
                              cálculo causal de ATR% de entrada (ver
                              `position_risk.calcular_atr_pct`). Solo se usa
                              si `tp_rule` no es None.
            atr_window:       Ventana del ATR (Wilder) para `tp_rule`.
            crowding:         ArenaCrowding opcional (§1.4): erosiona la
                              exposición de cada estrategia según cuánto de
                              su posición cae en tickers con alta demanda
                              AGREGADA de capital de toda la arena (no solo
                              su propia participación individual) — el
                              acoplamiento que convierte el ranking TTT en
                              una arena con interacción real entre
                              estrategias. None (default) = sin cambios.
            aum_total:        AUM total del meta-portafolio, en unidades
                              monetarias consistentes con el `adv_por_ticker`
                              de `crowding`. Ignorado si `crowding` es None.
        """
        if len(zoo) == 0:
            raise ValueError(
                "ZooManager está vacío. Agrega estrategias con zoo.agregar() antes de ejecutar."
            )
        if cap_bruto_exposicion <= 0.0:
            raise ValueError(f"cap_bruto_exposicion={cap_bruto_exposicion} debe ser > 0.")

        self._zoo = zoo
        self._metricas = metricas
        self._juez = juez
        self._datos = datos.sort_index()
        self._benchmark = benchmark.sort_index()
        self._ventana = ventana_metricas
        self._metrica_ranking = metrica_ranking

        self._risk_overlay = risk_overlay
        self._position_sizer = position_sizer
        self._cap_bruto_exposicion = cap_bruto_exposicion
        self._tp_rule = tp_rule
        self._datos_ohlc = datos_ohlc or {}
        self._atr_window = atr_window
        self._crowding = crowding
        self._aum_total = aum_total

        cfg = kalman_config or {}
        self._kalman = KalmanSignalFilter(
            Q=cfg.get('Q', 1e-4),
            R=cfg.get('R', 1e-2),
            P0=cfg.get('P0', 1.0),
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def ejecutar_walk_forward(
        self,
        fecha_inicio: pd.Timestamp,
        fecha_fin: pd.Timestamp,
        frecuencia_rebalanceo: Union[str, int] = 21,
    ) -> ResultadoBacktest:
        """
        Ejecuta la simulación walk-forward sobre el período [fecha_inicio, fecha_fin].

        Args:
            fecha_inicio:          Primera fecha de rebalanceo. Debe existir
                                   (o ser aproximable) en el índice de `datos`.
            fecha_fin:             Última fecha del backtest (inclusive).
            frecuencia_rebalanceo:
                - int → cada N días hábiles del calendario de datos
                        (ej. 21 ≈ mensual, 63 ≈ trimestral).
                - str → offset de Pandas (ej. 'ME'=fin de mes, 'QE'=trimestral,
                        'W-FRI'=viernes semanal).

        Returns:
            ResultadoBacktest con retornos, pesos y métricas históricas.
        """
        nombres: List[str] = self._zoo.nombres()
        n_estrategias: int = len(nombres)

        # ── 1. Fechas de rebalanceo ───────────────────────────────────────────
        fechas_rebalanceo: pd.DatetimeIndex = self._generar_fechas_rebalanceo(
            fecha_inicio, fecha_fin, frecuencia_rebalanceo
        )
        if len(fechas_rebalanceo) < 2:
            raise ValueError(
                f"Solo {len(fechas_rebalanceo)} rebalanceo(s) en el rango solicitado. "
                "Se necesitan al menos 2. Amplía el rango o reduce la frecuencia."
            )

        logger.info(
            "BacktestEngine.ejecutar_walk_forward | "
            f"estrategias={nombres} | rebalanceos={len(fechas_rebalanceo)} | "
            f"[{fecha_inicio.date()} → {fecha_fin.date()}] | "
            f"ventana={self._ventana} | metrica='{self._metrica_ranking}'"
        )

        if self._risk_overlay is not None:
            self._risk_overlay.reset()

        # ── 2. Inicialización ─────────────────────────────────────────────────

        # Pesos de portafolio vigentes (ticker-level) para el siguiente período
        # Se inicializan como equal-weight sobre el universo de cada estrategia
        pesos_vigentes: Dict[str, pd.Series] = {
            n: self._pesos_iniciales(n) for n in nombres
        }

        # Pesos meta del Juez (estrategia-level); equal-weight hasta que TTT tenga datos
        pesos_juez_actuales: Dict[str, float] = {n: 1.0 / n_estrategias for n in nombres}

        # Acumuladores de retornos: lista de (fecha, retorno_diario) por estrategia
        listas_ret: Dict[str, List[Tuple[pd.Timestamp, float]]] = {n: [] for n in nombres}

        # Historial de la métrica ruidosa (bruta) en cada fecha de rebalanceo
        hist_rolling: Dict[str, List[Tuple[pd.Timestamp, float]]] = {n: [] for n in nombres}

        # Historial de la métrica filtrada por Kalman
        hist_kalman: Dict[str, List[Tuple[pd.Timestamp, float]]] = {n: [] for n in nombres}

        # Snapshots de pesos del Juez en cada rebalanceo
        pesos_juez_snapshots: List[Tuple[pd.Timestamp, Dict[str, float]]] = []

        # Snapshots de pesos de portafolio (ticker-level) en cada rebalanceo
        pesos_port_snapshots: Dict[str, Dict[pd.Timestamp, pd.Series]] = {n: {} for n in nombres}

        # Último día de trading disponible ANTES de fecha_inicio (base para pct_change)
        fecha_base: pd.Timestamp = self._ultimo_dia_trading_antes(fecha_inicio)

        # ── 3. Loop walk-forward ──────────────────────────────────────────────
        for idx, fecha_corte in enumerate(fechas_rebalanceo):

            # ── 3a. Acumular retornos del período (t_{k-1}, t_k] ─────────────
            #        Usando pesos generados en t_{k-1} (ya vigentes al entrar al loop).
            #        Causal: pesos_vigentes NO contienen información de t_k.
            fecha_desde: pd.Timestamp = (
                fechas_rebalanceo[idx - 1] if idx > 0 else fecha_base
            )

            for nombre in nombres:
                ret_periodo = self._retornos_periodo_efectivo(
                    pesos=pesos_vigentes[nombre],
                    fecha_desde=fecha_desde,
                    fecha_hasta=fecha_corte,
                )
                listas_ret[nombre].extend(
                    zip(ret_periodo.index.tolist(), ret_periodo.tolist())
                )

            # ── 3b. Métrica rodante + Kalman + TTT (solo con datos suficientes) ──
            n_obs: int = min(len(listas_ret[n]) for n in nombres)

            if n_obs >= self._ventana + 5:

                metricas_periodo_raw: Dict[str, MetricasResultado] = {}
                ret_ventana_reciente: Dict[str, pd.Series] = {}

                # Calcular métrica rodante para cada estrategia
                for nombre in nombres:
                    ret_acc = self._serie_desde_lista(listas_ret[nombre])
                    bench_acc = self._benchmark.reindex(ret_acc.index).dropna()
                    ret_al, bench_al = ret_acc.align(bench_acc, join='inner')

                    if len(ret_al) < self._ventana:
                        hist_rolling[nombre].append((fecha_corte, np.nan))
                        metricas_periodo_raw[nombre] = MetricasResultado()
                        continue

                    ret_ventana_reciente[nombre] = ret_al.iloc[-self._ventana:]

                    # Último valor de la ventana rodante = señal actual (ruidosa)
                    try:
                        serie_roll = self._metricas.calcular_rolling(
                            ret_al, bench_al,
                            ventana=self._ventana,
                            metrica=self._metrica_ranking,
                        )
                        valor_bruto = float(serie_roll.iloc[-1]) if not serie_roll.empty else np.nan
                    except Exception as exc:
                        logger.warning(f"[{nombre}] calcular_rolling falló en {fecha_corte.date()}: {exc}")
                        valor_bruto = np.nan

                    hist_rolling[nombre].append((fecha_corte, valor_bruto))

                    # Métricas completas para contexto (no solo la de ranking)
                    try:
                        m_full = self._metricas.calcular_todas(ret_al, bench_al)
                    except ValueError:
                        m_full = MetricasResultado()
                    metricas_periodo_raw[nombre] = m_full

                # ── 3c. Aplicar Kalman al historial de métricas acumulado ───────
                #        La serie de observaciones brutas hasta t_k se filtra causalmente.
                metricas_para_ttt: Dict[str, MetricasResultado] = {}

                for nombre in nombres:
                    m_base = metricas_periodo_raw[nombre]
                    obs_hist = hist_rolling[nombre]

                    if len(obs_hist) >= 2:
                        fechas_h, vals_h = zip(*obs_hist)
                        serie_bruta = pd.Series(
                            list(vals_h),
                            index=pd.DatetimeIndex(list(fechas_h)),
                            name=nombre,
                            dtype=float,
                        )
                        # Filtro Kalman causal: usa solo observaciones <= t_k
                        serie_filt = self._kalman.filtrar(serie_bruta)
                        valor_filtrado = (
                            float(serie_filt.iloc[-1])
                            if not serie_filt.empty and np.isfinite(serie_filt.iloc[-1])
                            else float(vals_h[-1]) if np.isfinite(vals_h[-1]) else np.nan
                        )
                    else:
                        valor_filtrado = float(obs_hist[-1][1]) if obs_hist else np.nan

                    hist_kalman[nombre].append((fecha_corte, valor_filtrado))

                    # MetricasResultado con el campo de ranking reemplazado por el
                    # valor Kalman-filtrado (limpio) para registrar en TTT
                    d = m_base.to_dict()
                    d[self._metrica_ranking] = valor_filtrado
                    metricas_para_ttt[nombre] = MetricasResultado(**d)

                # ── 3d. Registrar período en TTTJuez y actualizar ────────────────
                #        tiempo en días desde época Unix (consistente con API TTT)
                tiempo_dias: float = fecha_corte.timestamp() / 86400.0

                try:
                    self._juez.registrar_periodo(
                        metricas_para_ttt, tiempo_dias, self._metrica_ranking
                    )
                    self._juez.actualizar()

                    if self._position_sizer is not None:
                        pesos_juez_actuales = self._pesos_via_sizer(
                            self._position_sizer, nombres, ret_ventana_reciente, fecha_corte
                        )
                    else:
                        pesos_juez_actuales = self._juez.pesos_asignacion()

                    top_nombre = max(
                        pesos_juez_actuales, key=lambda k: pesos_juez_actuales[k], default=None
                    )
                    if top_nombre is not None:
                        logger.info(
                            f"[{fecha_corte.date()}] TTT actualizado | "
                            f"top={top_nombre} ({pesos_juez_actuales[top_nombre]:.1%})"
                        )
                except Exception as exc:
                    logger.warning(
                        f"[{fecha_corte.date()}] TTTJuez.actualizar() falló: {exc}. "
                        "Manteniendo pesos anteriores."
                    )

            # ── 3e. Guardar snapshot de pesos del Juez para este rebalanceo ────
            pesos_juez_snapshots.append((fecha_corte, dict(pesos_juez_actuales)))

            # ── 3f. Generar nuevas señales para el SIGUIENTE período ───────────
            #        CAUSAL: datos filtrados hasta fecha_corte inclusive.
            nuevas_señales = self._zoo.generar_señales_todas(self._datos, fecha_corte)

            factores_crowding: Dict[str, float] = {}
            if self._crowding is not None:
                factores_crowding = self._crowding.factores_decaimiento(
                    pesos_por_estrategia=nuevas_señales,
                    capital_por_estrategia=pesos_juez_actuales,
                    aum_total=self._aum_total,
                )

            for nombre, pesos in nuevas_señales.items():
                if self._risk_overlay is not None:
                    pesos = self._escalar_por_riesgo(
                        risk_overlay=self._risk_overlay,
                        nombre=nombre,
                        pesos=pesos,
                        lista_ret_estrategia=listas_ret[nombre],
                        fecha_desde=fecha_desde,
                        fecha_hasta=fecha_corte,
                    )
                if self._crowding is not None:
                    pesos = pesos * factores_crowding.get(nombre, 1.0)
                pesos_vigentes[nombre] = pesos
                pesos_port_snapshots[nombre][fecha_corte] = pesos

        # ── 4. Consolidar resultados ──────────────────────────────────────────
        return self._consolidar(
            nombres=nombres,
            listas_ret=listas_ret,
            pesos_juez_snapshots=pesos_juez_snapshots,
            pesos_port_snapshots=pesos_port_snapshots,
            hist_rolling=hist_rolling,
            hist_kalman=hist_kalman,
            fechas_rebalanceo=fechas_rebalanceo,
        )

    # ------------------------------------------------------------------
    # Métodos privados — cálculo de retornos
    # ------------------------------------------------------------------

    def _pesos_iniciales(self, nombre: str) -> pd.Series:
        """Equal-weight sobre el universo de la estrategia como prior neutral."""
        universo = self._zoo.obtener(nombre).universo
        if not universo:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(universo), index=universo, dtype=float)

    def _ultimo_dia_trading_antes(self, fecha: pd.Timestamp) -> pd.Timestamp:
        """
        Retorna el último día de trading en self._datos estrictamente anterior a `fecha`.
        Si no existe, retorna el primer día disponible.
        """
        prev_dias = self._datos.index[self._datos.index < fecha]
        return prev_dias[-1] if len(prev_dias) > 0 else self._datos.index[0]

    def _retornos_periodo(
        self,
        pesos: pd.Series,
        fecha_desde: pd.Timestamp,
        fecha_hasta: pd.Timestamp,
    ) -> pd.Series:
        """
        Retornos diarios del portafolio para el período (fecha_desde, fecha_hasta].

        Incluye fecha_desde en el slice de precios solo como base de pct_change
        (su retorno no aparece en la salida). Todos los activos con peso ~0 son
        descartados antes del cálculo para evitar ruido numérico.

        Causalidad: los pesos deben haber sido generados en fecha_desde o antes.

        Args:
            pesos:       pd.Series [ticker → peso] generados en fecha_desde.
            fecha_desde: Fecha base (inclusive para pct_change, excluida del output).
            fecha_hasta: Última fecha del período (inclusive en el output).

        Returns:
            pd.Series de retornos diarios indexados en (fecha_desde, fecha_hasta].
            Serie vacía si no hay datos o activos válidos.
        """
        # Activos con peso significativo y presentes en los datos
        activos = [
            a for a in pesos.index
            if a in self._datos.columns and abs(pesos.get(a, 0.0)) > 1e-10
        ]

        # Fechas de trading en el período, incluyendo fecha_desde como base
        fechas_slice = self._datos.index[
            (self._datos.index >= fecha_desde) & (self._datos.index <= fecha_hasta)
        ]

        if len(fechas_slice) < 2:
            # Período vacío o de un solo día: no hay retorno computable
            return pd.Series(dtype=float)

        if not activos:
            # Sin posiciones: retorno cero para cada día del período
            return pd.Series(0.0, index=fechas_slice[1:], dtype=float)

        # Slice de precios vectorizado
        precios: pd.DataFrame = self._datos.loc[fechas_slice, activos]

        # pct_change() es O(T×N); iloc[1:] elimina la fila NaN del primer día
        retornos_activos: pd.DataFrame = precios.pct_change().iloc[1:]

        # Dot product vectorizado: retorno_portafolio[t] = pesos · retornos[t]
        pesos_validos = pesos[activos]
        return (retornos_activos * pesos_validos).sum(axis=1)

    def _retornos_periodo_efectivo(
        self,
        pesos: pd.Series,
        fecha_desde: pd.Timestamp,
        fecha_hasta: pd.Timestamp,
    ) -> pd.Series:
        """
        Despacha entre el cálculo vectorizado estándar (`_retornos_periodo`)
        y la variante path-dependent con Take-Profit por ATR
        (`retornos_periodo_con_riesgo`), según si `self._tp_rule` está
        configurado. Sin `tp_rule`, es exactamente `_retornos_periodo`
        (no-regresión).
        """
        if self._tp_rule is None or not self._datos_ohlc:
            return self._retornos_periodo(pesos, fecha_desde, fecha_hasta)

        atr_pct = self._atr_pct_activos(pesos, fecha_desde)
        return retornos_periodo_con_riesgo(
            pesos=pesos,
            datos=self._datos,
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
            atr_pct=atr_pct,
            tp_rule=self._tp_rule,
        )

    def _atr_pct_activos(
        self,
        pesos: pd.Series,
        fecha_desde: pd.Timestamp,
    ) -> Dict[str, float]:
        """ATR%(fecha_desde) causal por activo, para los tickers con OHLC disponible."""
        return {
            ticker: calcular_atr_pct(self._datos_ohlc[ticker], fecha_desde, self._atr_window)
            for ticker in pesos.index
            if ticker in self._datos_ohlc
        }

    def _escalar_por_riesgo(
        self,
        risk_overlay: RiskOverlay,
        nombre: str,
        pesos: pd.Series,
        lista_ret_estrategia: List[Tuple[pd.Timestamp, float]],
        fecha_desde: pd.Timestamp,
        fecha_hasta: pd.Timestamp,
    ) -> pd.Series:
        """
        Escala los pesos ticker-level recién generados por el factor de
        `RiskOverlay` derivado del track record PROPIO de la estrategia en
        el fold recién cerrado (fecha_desde, fecha_hasta] — causal: solo usa
        retornos ya realizados al momento de fecha_hasta.

        `RiskOverlay` mantiene estado persistente por `strategy_name` entre
        llamadas sucesivas (un fold por rebalanceo), acumulando equity y HWM
        globales a través de todo el walk-forward.

        La volatilidad realizada se calcula sobre la serie COMPLETA acumulada
        (no solo el fold): un fold de ~21 días no tiene lookback suficiente
        para su propia ventana rodante, así que se precomputa con historial
        completo y se recorta al fold vía `market_data[vol_col]` — evita que
        `RiskOverlay._annualized_vol` recaiga en su fallback fold-local (que
        produciría NaN casi todo el fold y anularía la exposición).
        """
        if not lista_ret_estrategia:
            return pesos

        serie = self._serie_desde_lista(lista_ret_estrategia)
        fold = serie[(serie.index > fecha_desde) & (serie.index <= fecha_hasta)]
        if fold.empty:
            return pesos

        vol_completa = serie.rolling(
            risk_overlay.vol_window, min_periods=risk_overlay.min_periods
        ).std() * np.sqrt(252)

        base_weight = pd.Series(1.0, index=fold.index)
        market_data = pd.DataFrame({risk_overlay.vol_col: vol_completa.reindex(fold.index)})
        escalado = risk_overlay.compute_risk_weight(
            base_weight, fold, market_data, strategy_name=nombre
        )
        factor = float(escalado.iloc[-1]) if not escalado.empty and np.isfinite(escalado.iloc[-1]) else 1.0
        return pesos * factor

    def _pesos_via_sizer(
        self,
        position_sizer: AbstractPositionSizer,
        nombres: List[str],
        ret_ventana_reciente: Dict[str, pd.Series],
        fecha_corte: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Sustituye `TTTJuez.pesos_asignacion()` (pesos que suman 1) por
        exposición absoluta calculada con `self._position_sizer` (ej.
        `KellyBayesianSizer`): μ_edge y σ_retornos se estiman empíricamente
        de la ventana reciente de retornos realizados de cada estrategia;
        σ_skill viene del posterior TTT (`habilidades_latentes()`), sumada
        opcionalmente a la incertidumbre de régimen de la propia estrategia
        (§1.5 — ver `_incertidumbre_regimen_extra`).

        Renormaliza contra `self._cap_bruto_exposicion` si la suma cruda la
        excede — cota de seguridad válida para cualquier sizer, no solo Kelly.
        """
        habilidades = self._juez.habilidades_latentes()

        crudo: Dict[str, float] = {}
        for nombre in nombres:
            ret_reciente = ret_ventana_reciente.get(nombre)
            if ret_reciente is None or ret_reciente.empty:
                crudo[nombre] = 0.0
                continue

            mu_edge = float(ret_reciente.mean())
            sigma_retornos = float(ret_reciente.std())
            _, sigma_skill_ttt = habilidades.get(nombre, (0.0, self._juez_sigma_prior()))
            sigma_skill = sigma_skill_ttt + self._incertidumbre_regimen_extra(nombre, fecha_corte)

            try:
                crudo[nombre] = position_sizer.exposicion(
                    mu_edge=mu_edge,
                    sigma_retornos=sigma_retornos,
                    sigma_skill=sigma_skill,
                )
            except SizingError:
                crudo[nombre] = 0.0

        bruto = sum(crudo.values())
        if bruto > self._cap_bruto_exposicion:
            factor = self._cap_bruto_exposicion / bruto
            crudo = {k: v * factor for k, v in crudo.items()}

        return crudo

    @staticmethod
    def _juez_sigma_prior() -> float:
        """
        σ_skill de respaldo para una estrategia sin entrada en
        `habilidades_latentes()` (aún no compitió). Un valor alto penaliza
        fuertemente vía el denominador de Kelly — coherente con "sin
        convicción, sin exposición" (ver H4 en el plan de diseño).
        """
        return 10.0

    def _incertidumbre_regimen_extra(self, nombre: str, fecha_corte: pd.Timestamp) -> float:
        """
        Hook opcional (duck-typing, no forma parte de `AbstractStrategy` —
        forzarlo violaría ISP: la mayoría de estrategias no tienen noción de
        "régimen"): si la estrategia expone `incertidumbre_regimen(log_rets)`
        (ej. `HMMGARCHStrategy`, §1.5), se calcula sobre los log-retornos
        causales del primer ticker de su universo y se suma a σ_skill_TTT en
        `_pesos_via_sizer`.

        Nota de escala: `incertidumbre_regimen` vive en [0, 1−1/K] (escala
        de probabilidad); σ_skill_TTT vive en la escala interna de TTT. La
        suma directa es, igual que `kappa_skill` en `KellyBayesianSizer`,
        una aproximación honesta que requiere calibración empírica, no una
        equivalencia teórica exacta.

        Returns 0.0 (sin efecto) si la estrategia no expone el método, no
        tiene universo válido en `self._datos`, o el cálculo falla — nunca
        interrumpe el walk-forward por un fallo de este hook opcional.
        """
        try:
            estrategia = self._zoo.obtener(nombre)
        except KeyError:
            return 0.0

        metodo = getattr(estrategia, "incertidumbre_regimen", None)
        if not callable(metodo):
            return 0.0

        universo = getattr(estrategia, "universo", [])
        if not universo or universo[0] not in self._datos.columns:
            return 0.0

        try:
            precios = self._datos.loc[self._datos.index <= fecha_corte, universo[0]].astype(float)
            log_rets = np.log(precios / precios.shift(1)).dropna().values
            if len(log_rets) < 30:
                return 0.0
            return float(metodo(log_rets))
        except Exception:
            return 0.0

    @staticmethod
    def _serie_desde_lista(
        lista: List[Tuple[pd.Timestamp, float]],
    ) -> pd.Series:
        """Convierte [(fecha, valor), ...] en pd.Series con DatetimeIndex."""
        if not lista:
            return pd.Series(dtype=float)
        fechas, valores = zip(*lista)
        return pd.Series(
            list(valores),
            index=pd.DatetimeIndex(list(fechas)),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Métodos privados — generación de fechas
    # ------------------------------------------------------------------

    def _generar_fechas_rebalanceo(
        self,
        fecha_inicio: pd.Timestamp,
        fecha_fin: pd.Timestamp,
        frecuencia: Union[str, int],
    ) -> pd.DatetimeIndex:
        """
        Genera fechas de rebalanceo dentro de [fecha_inicio, fecha_fin],
        snapeadas al último día de trading disponible en self._datos.

        Modo int: selecciona cada N-ésimo día del calendario de trading.
        Modo str: genera fechas con el offset Pandas y snap al día de trading
                  más cercano anterior o igual (conservador, causal).
        """
        dias_disponibles: pd.DatetimeIndex = self._datos.index[
            (self._datos.index >= fecha_inicio) & (self._datos.index <= fecha_fin)
        ]

        if len(dias_disponibles) == 0:
            raise ValueError(
                f"No hay datos de precios entre {fecha_inicio.date()} y {fecha_fin.date()}. "
                "Verifica que self._datos cubra este período."
            )

        if isinstance(frecuencia, int):
            # Cada N-ésimo día de trading: stride directo sobre el índice
            return dias_disponibles[::frecuencia]

        # Frecuencia Pandas: snap al último día de trading <= fecha calendario
        fechas_calendario: pd.DatetimeIndex = pd.date_range(
            start=fecha_inicio, end=fecha_fin, freq=frecuencia
        )
        snapped: List[pd.Timestamp] = []
        for fecha_cal in fechas_calendario:
            candidatos = dias_disponibles[dias_disponibles <= fecha_cal]
            if len(candidatos) > 0:
                snapped.append(candidatos[-1])

        if not snapped:
            raise ValueError(
                f"La frecuencia '{frecuencia}' no generó ninguna fecha válida de rebalanceo "
                f"entre {fecha_inicio.date()} y {fecha_fin.date()}."
            )

        return pd.DatetimeIndex(sorted(set(snapped)))

    # ------------------------------------------------------------------
    # Métodos privados — consolidación de resultados
    # ------------------------------------------------------------------

    def _consolidar(
        self,
        nombres: List[str],
        listas_ret: Dict[str, List[Tuple[pd.Timestamp, float]]],
        pesos_juez_snapshots: List[Tuple[pd.Timestamp, Dict[str, float]]],
        pesos_port_snapshots: Dict[str, Dict[pd.Timestamp, pd.Series]],
        hist_rolling: Dict[str, List[Tuple[pd.Timestamp, float]]],
        hist_kalman: Dict[str, List[Tuple[pd.Timestamp, float]]],
        fechas_rebalanceo: pd.DatetimeIndex,
    ) -> ResultadoBacktest:
        """
        Ensambla todas las estructuras acumuladas en un ResultadoBacktest cohesivo.

        Alineación de índices:
          - retornos_estrategias/meta: unión de días con retornos calculados.
          - pesos_juez/metricas: indexados por fecha de rebalanceo (baja frecuencia).
          - pesos_portafolio: snapshots en fechas de rebalanceo por estrategia.
        """

        # --- Retornos diarios individuales por estrategia ---
        retornos_df = pd.DataFrame(
            {n: self._serie_desde_lista(listas_ret[n]) for n in nombres}
        ).sort_index()

        # --- Meta-portafolio: pesos TTT aplicados por período ---
        retornos_meta = self._ensamblar_meta(retornos_df, pesos_juez_snapshots)

        # --- Pesos Juez en fechas de rebalanceo ---
        if pesos_juez_snapshots:
            fechas_j, dicts_j = zip(*pesos_juez_snapshots)
            pesos_juez_df = pd.DataFrame(
                list(dicts_j), index=pd.DatetimeIndex(list(fechas_j))
            ).sort_index()
            pesos_juez_df.index.name = 'fecha_rebalanceo'
        else:
            pesos_juez_df = pd.DataFrame()

        # --- Pesos portafolio: snapshots de rebalanceo por estrategia ---
        pesos_portafolio_dfs: Dict[str, pd.DataFrame] = {}
        for nombre in nombres:
            snap = pesos_port_snapshots.get(nombre, {})
            if snap:
                pesos_portafolio_dfs[nombre] = (
                    pd.DataFrame(snap)
                    .T.sort_index()
                    .rename_axis('fecha_rebalanceo')
                )
            else:
                pesos_portafolio_dfs[nombre] = pd.DataFrame()

        # --- Métricas en fechas de rebalanceo ---
        def _hist_a_df(hist: Dict[str, List[Tuple[pd.Timestamp, float]]]) -> pd.DataFrame:
            series_dict = {}
            for nombre in nombres:
                puntos = hist.get(nombre, [])
                if puntos:
                    fechas_h, vals_h = zip(*puntos)
                    series_dict[nombre] = pd.Series(
                        list(vals_h), index=pd.DatetimeIndex(list(fechas_h))
                    )
            if not series_dict:
                return pd.DataFrame()
            df = pd.DataFrame(series_dict).sort_index()
            df.index.name = 'fecha_rebalanceo'
            return df

        metricas_rolling_df = _hist_a_df(hist_rolling)
        metricas_kalman_df  = _hist_a_df(hist_kalman)

        return ResultadoBacktest(
            retornos_meta=retornos_meta,
            retornos_estrategias=retornos_df,
            pesos_juez=pesos_juez_df,
            pesos_portafolio=pesos_portafolio_dfs,
            metricas_rolling=metricas_rolling_df,
            metricas_kalman=metricas_kalman_df,
            fechas_rebalanceo=fechas_rebalanceo,
            metrica_usada=self._metrica_ranking,
        )

    def _ensamblar_meta(
        self,
        retornos_estrategias: pd.DataFrame,
        pesos_juez_snapshots: List[Tuple[pd.Timestamp, Dict[str, float]]],
    ) -> pd.Series:
        """
        Construye la serie de retornos diarios del meta-portafolio.

        Para cada período (t_k, t_{k+1}], aplica los pesos del Juez determinados
        en t_k. Esta asignación es CAUSAL: los pesos se conocen antes del inicio
        del período en que se aplican.

        Si un rebalanceo no generó actualización TTT (warm-up), se usan los
        pesos equal-weight que estaban vigentes en ese momento.

        Renormalización: `TTTJuez.pesos_asignacion()` garantiza suma=1, así
        que si faltan columnas (estrategia sin retorno ese día) se renormaliza
        para no perder exposición por una ausencia accidental. Con
        `self._position_sizer` activo (ej. KellyBayesianSizer) los pesos son
        EXPOSICIÓN ABSOLUTA por diseño (pueden sumar < 1 a propósito — ver
        H4/§1.2): renormalizar aquí destruiría exactamente la propiedad que
        el sizer implementa, así que se omite.

        Args:
            retornos_estrategias: DataFrame [fecha x estrategia].
            pesos_juez_snapshots: [(fecha_rebalanceo, {estrategia: peso}), ...]
                                  Ordenados cronológicamente.

        Returns:
            pd.Series de retornos diarios del meta-portafolio.
        """
        if not pesos_juez_snapshots or retornos_estrategias.empty:
            return pd.Series(dtype=float)

        renormalizar = self._position_sizer is None

        salida = pd.Series(np.nan, index=retornos_estrategias.index, dtype=float)
        n = len(pesos_juez_snapshots)

        for i, (fecha_k, pesos_dict) in enumerate(pesos_juez_snapshots):
            # Período de aplicación: (t_k, t_{k+1}]
            if i + 1 < n:
                fecha_sig = pesos_juez_snapshots[i + 1][0]
                mascara = (
                    (retornos_estrategias.index > fecha_k) &
                    (retornos_estrategias.index <= fecha_sig)
                )
            else:
                mascara = retornos_estrategias.index > fecha_k

            fechas_p = retornos_estrategias.index[mascara]
            if len(fechas_p) == 0:
                continue

            ret_p = retornos_estrategias.loc[fechas_p]
            cols = [c for c in pesos_dict if c in ret_p.columns]
            if not cols:
                salida.loc[fechas_p] = 0.0
                continue

            pesos_s = pd.Series({c: pesos_dict[c] for c in cols})
            total = pesos_s.sum()
            if renormalizar and total > 1e-10:
                pesos_s = pesos_s / total  # renormalizar si hay estrategias ausentes

            # Dot product vectorizado: retorno_meta[t] = sum_i(w_i * r_i[t])
            salida.loc[fechas_p] = (ret_p[cols] * pesos_s).sum(axis=1).values

        return salida.dropna()
