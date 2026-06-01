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

from quant_arena.core.abstracciones import AbstractJuez, MetricasResultado
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
        """
        if len(zoo) == 0:
            raise ValueError(
                "ZooManager está vacío. Agrega estrategias con zoo.agregar() antes de ejecutar."
            )

        self._zoo = zoo
        self._metricas = metricas
        self._juez = juez
        self._datos = datos.sort_index()
        self._benchmark = benchmark.sort_index()
        self._ventana = ventana_metricas
        self._metrica_ranking = metrica_ranking

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
                ret_periodo = self._retornos_periodo(
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

                # Calcular métrica rodante para cada estrategia
                for nombre in nombres:
                    ret_acc = self._serie_desde_lista(listas_ret[nombre])
                    bench_acc = self._benchmark.reindex(ret_acc.index).dropna()
                    ret_al, bench_al = ret_acc.align(bench_acc, join='inner')

                    if len(ret_al) < self._ventana:
                        hist_rolling[nombre].append((fecha_corte, np.nan))
                        metricas_periodo_raw[nombre] = MetricasResultado()
                        continue

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
                    pesos_juez_actuales = self._juez.pesos_asignacion()

                    top_nombre = max(pesos_juez_actuales, key=pesos_juez_actuales.get)
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
            for nombre, pesos in nuevas_señales.items():
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

    @staticmethod
    def _ensamblar_meta(
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

        Args:
            retornos_estrategias: DataFrame [fecha x estrategia].
            pesos_juez_snapshots: [(fecha_rebalanceo, {estrategia: peso}), ...]
                                  Ordenados cronológicamente.

        Returns:
            pd.Series de retornos diarios del meta-portafolio.
        """
        if not pesos_juez_snapshots or retornos_estrategias.empty:
            return pd.Series(dtype=float)

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
            if total > 1e-10:
                pesos_s = pesos_s / total  # renormalizar si hay estrategias ausentes

            # Dot product vectorizado: retorno_meta[t] = sum_i(w_i * r_i[t])
            salida.loc[fechas_p] = (ret_p[cols] * pesos_s).sum(axis=1).values

        return salida.dropna()
