#!/usr/bin/env python3
# =============================================================================
# FILE: quant_arena/scripts/run_backtest.py
# Orquestador maestro del ecosistema quant_arena — arquitectura POO estricta
# =============================================================================
"""
Pipeline de simulación walk-forward completo para el Zoo de estrategias.

Clases:
    PipelineConfig     Configuración declarativa del experimento.
    DataLoader         Carga y valida el dataset histórico del S&P 500.
    StrategyLoader     Descubre, prueba e instancia las estrategias disponibles.
    JuezBuilder        Construye y opcionalmente calibra el TTTJuez.
    ResultsExporter    Persiste resultados en CSV y PDF tear-sheet.
    SimulationPipeline Orquestador maestro que coordina el ciclo completo.

Ejecución mínima (desde la raíz del proyecto):
    python quant_arena/scripts/run_backtest.py

Ejecución parametrizada:
    python quant_arena/scripts/run_backtest.py \\
        --inicio 2010-01-01 --fin 2022-12-31 \\
        --frecuencia 21 --metrica sharpe --salida ./resultados

Ver todas las opciones:
    python quant_arena/scripts/run_backtest.py --help
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Resuelve imports tanto al ejecutar como script directo como como módulo ──
_RAIZ = Path(__file__).resolve().parent.parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from quant_arena.backtesting.motor import BacktestEngine, ResultadoBacktest
from quant_arena.juez.ttt_juez import TTTJuez
from quant_arena.metricas.performance_metrics import PerformanceMetrics
from quant_arena.resultados.visualizador import ReporteCuantitativo
from quant_arena.zoo.base_estrategia import RegistroZoo, ZooManager


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quant_arena.run_backtest")


# =============================================================================
# PipelineConfig — contrato declarativo del experimento
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Parámetros completos del experimento de simulación.
    Todos los campos tienen defaults operativos; el pipeline se puede lanzar
    sin modificar ninguno de ellos si el entorno está correctamente instalado.
    """

    # ── Datos ────────────────────────────────────────────────────────────────
    ruta_parquet: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent
            / "data"
            / "sp500_daily_1997_to_today.parquet"
        )
    )
    columna_precio: str = "Close"

    # ── Rango temporal del backtest ───────────────────────────────────────────
    fecha_inicio: str = "2005-01-01"
    fecha_fin: str = "2023-12-31"
    frecuencia_rebalanceo: int = 21     # días hábiles (≈ mensual)

    # ── Métricas y ranking ────────────────────────────────────────────────────
    ventana_metricas: int = 63          # días hábiles (≈ trimestral)
    metrica_ranking: str = "sharpe"     # campo de MetricasResultado

    # ── Juez TTT ──────────────────────────────────────────────────────────────
    ttt_sigma: float = 1.6              # incertidumbre del prior de habilidad
    ttt_gamma: float = 0.036            # drift del proceso de Wiener entre períodos
    ttt_metodo_asignacion: str = "mu_sobre_sigma"
    calibrar_hiperparametros: bool = False  # True → OptimizadorTTT antes del run

    # ── Filtro de Kalman ──────────────────────────────────────────────────────
    kalman_Q: float = 1e-4              # varianza del proceso (ruido de modelo)
    kalman_R: float = 1e-2              # varianza de medición (ruido de observación)

    # ── Estrategias ───────────────────────────────────────────────────────────
    estrategias_habilitadas: Optional[List[str]] = None  # None = todas disponibles
    ventana_probe: int = 200            # observaciones mínimas para el probe

    # ── Salidas ───────────────────────────────────────────────────────────────
    directorio_salida: Path = field(
        default_factory=lambda: Path.cwd() / "resultados_backtest"
    )
    generar_pdf: bool = True
    guardar_csv: bool = True
    mostrar_graficos: bool = False

    # ──────────────────────────────────────────────────────────────────────────

    def validar(self) -> None:
        """
        Valida la coherencia interna de la configuración antes de arrancar.

        Raises:
            FileNotFoundError: Si el archivo Parquet no existe.
            ValueError:        Si los parámetros temporales o de métrica son inválidos.
        """
        if not self.ruta_parquet.exists():
            raise FileNotFoundError(
                f"Parquet no encontrado: {self.ruta_parquet}\n"
                "Genera el dataset con: python quant_arena/scripts/fetch_sp500.py"
            )
        t0 = pd.Timestamp(self.fecha_inicio)
        t1 = pd.Timestamp(self.fecha_fin)
        if t0 >= t1:
            raise ValueError(
                f"fecha_inicio ({self.fecha_inicio}) debe ser anterior a "
                f"fecha_fin ({self.fecha_fin})."
            )
        metricas_validas = {
            "sharpe", "sortino", "calmar", "information_ratio", "alpha_tstat"
        }
        if self.metrica_ranking not in metricas_validas:
            raise ValueError(
                f"metrica_ranking='{self.metrica_ranking}' inválida. "
                f"Opciones: {sorted(metricas_validas)}"
            )
        if self.ventana_metricas < 20:
            raise ValueError(
                f"ventana_metricas={self.ventana_metricas} < 20 (mínimo estadístico)."
            )


# =============================================================================
# DataLoader — carga y preparación del dataset histórico
# =============================================================================

class DataLoader:
    """
    Carga el dataset histórico del S&P 500 desde el archivo Parquet configurado.

    Responsabilidades:
        - Leer y validar la estructura del Parquet.
        - Extraer la columna de precios de cierre (ajustados).
        - Construir la serie de retornos diarios del benchmark.
        - El recorte temporal queda delegado al BacktestEngine para preservar
          el período de warm-up de las métricas rolling.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def cargar(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Lee el Parquet y retorna el par (datos, benchmark).

        Returns:
            datos:     DataFrame [fecha x columna_precio] con precios ajustados.
            benchmark: pd.Series de retornos diarios del S&P 500.

        Raises:
            FileNotFoundError: Si el Parquet no existe.
            KeyError:          Si columna_precio no está en el Parquet.
        """
        logger.info(f"Cargando dataset: {self._config.ruta_parquet.name}")

        raw: pd.DataFrame = pd.read_parquet(self._config.ruta_parquet)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()

        col = self._config.columna_precio
        if col not in raw.columns:
            raise KeyError(
                f"columna_precio='{col}' no encontrada en el Parquet. "
                f"Columnas disponibles: {list(raw.columns)}"
            )

        # `datos` contiene solo la columna de precio; el BacktestEngine accede
        # únicamente a columnas presentes en el universo de cada estrategia.
        datos: pd.DataFrame = raw[[col]].dropna()

        # Benchmark: retornos diarios de la misma serie de precio de cierre.
        benchmark: pd.Series = datos[col].pct_change().dropna()
        benchmark.name = "SP500_Close"

        logger.info(
            f"Dataset listo | filas={len(datos):,} | "
            f"rango={datos.index[0].date()} → {datos.index[-1].date()} | "
            f"columna='{col}'"
        )
        return datos, benchmark


# =============================================================================
# StrategyLoader — descubrimiento, probe y construcción del Zoo
# =============================================================================

class StrategyLoader:
    """
    Descubre las estrategias del Zoo, las instancia y filtra las operativas.

    Ciclo de vida:
        1. RegistroZoo.autodescubrir() importa todos los módulos del Zoo.
           Los decoradores @RegistroZoo.registrar() pueblan el registro global.
        2. _probe() genera señales sobre una muestra mínima de datos históricos.
           Estrategias con ImportError o excepción numérica se excluyen.
        3. ZooManager se construye únicamente con las estrategias que pasaron.

    La separación entre «descubrir» y «probar» garantiza que una dependencia
    de DL faltante (p.ej. torch) no bloquea la ejecución de las estrategias
    de ML clásico que sí están instaladas.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._fallidos: Dict[str, str] = {}

    @property
    def fallidos(self) -> Dict[str, str]:
        """Mapa {nombre_estrategia: motivo_de_fallo} para diagnóstico."""
        return dict(self._fallidos)

    def construir_zoo(
        self,
        datos_probe: pd.DataFrame,
        universo: List[str],
    ) -> ZooManager:
        """
        Construye un ZooManager con las estrategias disponibles y operativas.

        Args:
            datos_probe: Slice de datos históricos para el probe de señales.
            universo:    Lista de tickers (columnas de datos) asignada a cada
                         estrategia como universo invertible.

        Returns:
            ZooManager poblado con instancias que superaron el probe.

        Raises:
            RuntimeError: Si ninguna estrategia supera el probe.
        """
        logger.info("Autodescubriendo estrategias del Zoo...")
        RegistroZoo.autodescubrir()
        registradas = RegistroZoo.listar()
        logger.info(f"Estrategias en registro: {registradas}")

        nombres_objetivo: List[str] = (
            self._config.estrategias_habilitadas
            if self._config.estrategias_habilitadas is not None
            else registradas
        )

        zoo = ZooManager()
        fecha_corte_probe: pd.Timestamp = datos_probe.index[-1]

        for nombre in nombres_objetivo:
            if nombre not in registradas:
                self._fallidos[nombre] = "No registrada en RegistroZoo"
                logger.warning(f"  [SKIP] {nombre} — no registrada")
                continue

            try:
                klass = RegistroZoo.obtener_clase(nombre)
                instancia = klass(universo=universo)

                # Probe: señales sobre muestra histórica mínima
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _ = instancia.generar_señales(datos_probe, fecha_corte_probe)

                zoo.agregar(instancia)
                logger.info(f"  [OK]   {nombre}")

            except ImportError as exc:
                motivo = f"ImportError: {exc}"
                self._fallidos[nombre] = motivo
                logger.warning(f"  [SKIP] {nombre} — dependencia faltante: {exc}")
            except Exception as exc:
                motivo = f"{type(exc).__name__}: {exc}"
                self._fallidos[nombre] = motivo
                logger.warning(f"  [SKIP] {nombre} — {motivo}")

        if len(zoo) == 0:
            raise RuntimeError(
                "Ninguna estrategia disponible tras el probe. "
                "Instala las dependencias con: pip install -e '.[ml,dl,nlp]'"
            )

        logger.info(
            f"Zoo listo | operativas={len(zoo)} | omitidas={len(self._fallidos)}"
        )
        return zoo


# =============================================================================
# JuezBuilder — construcción y calibración opcional del TTTJuez
# =============================================================================

class JuezBuilder:
    """
    Construye la instancia del Juez TTT con parámetros calibrados o por defecto.

    Si config.calibrar_hiperparametros=True, ejecuta OptimizadorTTT (L-BFGS-B)
    sobre datos sintéticos para ajustar sigma y gamma antes del backtest.
    Los fallos de calibración se degradan silenciosamente a los defaults.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def construir(self) -> TTTJuez:
        """
        Retorna un TTTJuez configurado y listo para recibir períodos de competencia.
        """
        sigma = self._config.ttt_sigma
        gamma = self._config.ttt_gamma

        if self._config.calibrar_hiperparametros:
            sigma, gamma = self._calibrar(sigma, gamma)

        juez = TTTJuez(
            sigma=sigma,
            gamma=gamma,
            modo_causal_estricto=True,
        )
        logger.info(f"TTTJuez listo | sigma={sigma:.4f} | gamma={gamma:.4f}")
        return juez

    def _calibrar(
        self, sigma_0: float, gamma_0: float
    ) -> Tuple[float, float]:
        """
        Ajusta sigma/gamma sobre datos históricos sintéticos, delegando en
        `TTTJuez.calibrar_hiperparametros()` (que a su vez usa
        `OptimizadorTTT`, L-BFGS-B) en vez de reimplementar el mismo cálculo.
        """
        try:
            from quant_arena.core.abstracciones import MetricasResultado

            logger.info("Calibrando hiperparámetros TTT (L-BFGS-B)...")
            rng = np.random.default_rng(42)
            n_jugadores = 4
            jugadores = [f"strat_{i}" for i in range(n_jugadores)]
            juez_temp = TTTJuez(sigma=sigma_0, gamma=gamma_0)

            for epoca in range(1, 25):
                metricas_ep = {
                    j: MetricasResultado(sharpe=float(rng.normal(0.4, 0.35)))
                    for j in jugadores
                }
                juez_temp.registrar_periodo(metricas_ep, tiempo=float(epoca))

            resultado = juez_temp.calibrar_hiperparametros()

            sigma_opt = resultado["sigma_optimo"]
            gamma_opt = resultado["gamma_optimo"]
            logger.info(
                f"Calibración completada | "
                f"sigma {sigma_0:.4f} → {sigma_opt:.4f} | "
                f"gamma {gamma_0:.4f} → {gamma_opt:.4f}"
            )
            return sigma_opt, gamma_opt

        except Exception as exc:
            logger.warning(
                f"Calibración fallida ({exc}). "
                f"Usando defaults: sigma={sigma_0}, gamma={gamma_0}"
            )
            return sigma_0, gamma_0


# =============================================================================
# ResultsExporter — persistencia de artefactos de salida
# =============================================================================

class ResultsExporter:
    """
    Exporta los resultados del backtest a archivos CSV y un PDF tear-sheet.

    Estructura de salida en config.directorio_salida/:
        retornos_meta.csv           Serie de retornos diarios del meta-portafolio.
        retornos_estrategias.csv    Retornos por estrategia (fecha x estrategia).
        pesos_juez.csv              Pesos TTT en fechas de rebalanceo.
        metricas_rolling.csv        Métricas brutas (ruidosas) en cada rebalanceo.
        metricas_kalman.csv         Métricas filtradas por Kalman en cada rebalanceo.
        resumen_estadistico.csv     KPIs finales anualizados por estrategia.
        tearsheet.pdf               Tear-sheet visual institucional (7 paneles).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._dir = config.directorio_salida
        self._dir.mkdir(parents=True, exist_ok=True)

    def exportar(
        self,
        resultado: ResultadoBacktest,
        juez: TTTJuez,
        benchmark: pd.Series,
        metricas: PerformanceMetrics,
    ) -> None:
        """Ejecuta la exportación completa según la configuración activa."""
        if self._config.guardar_csv:
            self._exportar_csv(resultado, benchmark, metricas)
        if self._config.generar_pdf:
            self._exportar_pdf(resultado, benchmark, juez)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _exportar_csv(
        self,
        resultado: ResultadoBacktest,
        benchmark: pd.Series,
        metricas: PerformanceMetrics,
    ) -> None:
        archivos: Dict[str, object] = {
            "retornos_meta.csv":        resultado.retornos_meta.rename("meta_portfolio"),
            "retornos_estrategias.csv": resultado.retornos_estrategias,
            "pesos_juez.csv":           resultado.pesos_juez,
            "metricas_rolling.csv":     resultado.metricas_rolling,
            "metricas_kalman.csv":      resultado.metricas_kalman,
        }
        for nombre_archivo, obj in archivos.items():
            ruta = self._dir / nombre_archivo
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(ruta)
            elif isinstance(obj, pd.Series):
                obj.to_csv(ruta, header=True)
            logger.info(f"  CSV → {nombre_archivo}")

        try:
            resumen = resultado.resumen_estadistico(metricas, benchmark)
            if not resumen.empty:
                ruta_res = self._dir / "resumen_estadistico.csv"
                resumen.to_csv(ruta_res)
                logger.info("  CSV → resumen_estadistico.csv")
                _imprimir_resumen(resumen)
        except Exception as exc:
            logger.warning(f"Resumen estadístico no generado: {exc}")

    def _exportar_pdf(
        self,
        resultado: ResultadoBacktest,
        benchmark: pd.Series,
        juez: TTTJuez,
    ) -> None:
        try:
            import matplotlib.pyplot as plt

            reporte = ReporteCuantitativo(
                resultado, benchmark=benchmark, juez=juez
            )
            ruta_pdf = self._dir / "tearsheet.pdf"
            reporte.generar_tearsheet(str(ruta_pdf))
            logger.info(f"  PDF → tearsheet.pdf")

            if self._config.mostrar_graficos:
                plt.show()

        except Exception as exc:
            logger.warning(f"Tear-sheet PDF no generado: {exc}")


# =============================================================================
# SimulationPipeline — orquestador maestro
# =============================================================================

class SimulationPipeline:
    """
    Orquestador maestro del ecosistema quant_arena.

    Coordina cuatro fases secuenciales con responsabilidades bien delimitadas:

        Fase 1 — Datos       DataLoader carga y valida el Parquet histórico.
        Fase 2 — Componentes StrategyLoader, JuezBuilder y PerformanceMetrics.
        Fase 3 — Ejecución   BacktestEngine.ejecutar_walk_forward().
        Fase 4 — Exportación ResultsExporter persiste CSV y PDF.

    Patrón de uso mínimo::

        config = PipelineConfig(
            fecha_inicio="2010-01-01",
            fecha_fin="2022-12-31",
            estrategias_habilitadas=["momentum_126d", "xgboost_trend"],
        )
        pipeline = SimulationPipeline(config)
        resultado = pipeline.ejecutar()

    El objeto ``resultado`` (ResultadoBacktest) queda almacenado en
    ``pipeline.resultado`` para uso posterior en notebooks o análisis ad-hoc.
    """

    def __init__(self, config: PipelineConfig) -> None:
        config.validar()
        self._config = config
        self._resultado:  Optional[ResultadoBacktest] = None
        self._juez:       Optional[TTTJuez] = None
        self._metricas:   Optional[PerformanceMetrics] = None
        self._benchmark:  Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def ejecutar(self) -> ResultadoBacktest:
        """
        Ejecuta el pipeline de punta a punta y retorna el ResultadoBacktest.

        Raises:
            FileNotFoundError: Si el Parquet no existe.
            RuntimeError:      Si ninguna estrategia está disponible.
            ValueError:        Si la configuración es incoherente.
        """
        t_global = time.perf_counter()
        _encabezado("quant_arena — Pipeline Walk-Forward")

        datos, benchmark     = self._fase_datos()
        zoo, metricas, juez  = self._fase_componentes(datos, benchmark)
        resultado            = self._fase_ejecucion(zoo, metricas, juez, datos, benchmark)
        self._fase_exportacion(resultado, juez, benchmark, metricas)

        elapsed = time.perf_counter() - t_global
        logger.info(f"Pipeline completado en {elapsed:.1f} s")
        _encabezado("FIN")

        self._resultado = resultado
        self._juez      = juez
        self._metricas  = metricas
        self._benchmark = benchmark
        return resultado

    @property
    def resultado(self) -> Optional[ResultadoBacktest]:
        """Último ResultadoBacktest producido. None si aún no se ejecutó."""
        return self._resultado

    @property
    def juez(self) -> Optional[TTTJuez]:
        """Instancia del TTTJuez usada en el último run."""
        return self._juez

    # ------------------------------------------------------------------
    # Fases internas (un método por responsabilidad)
    # ------------------------------------------------------------------

    def _fase_datos(self) -> Tuple[pd.DataFrame, pd.Series]:
        loader = DataLoader(self._config)
        return loader.cargar()

    def _fase_componentes(
        self,
        datos: pd.DataFrame,
        benchmark: pd.Series,
    ) -> Tuple[ZooManager, PerformanceMetrics, TTTJuez]:
        universo = list(datos.columns)

        n_probe = min(self._config.ventana_probe, len(datos))
        datos_probe = datos.iloc[:n_probe]

        strategy_loader = StrategyLoader(self._config)
        zoo = strategy_loader.construir_zoo(datos_probe, universo)

        if strategy_loader.fallidos:
            logger.warning("Estrategias omitidas (dependencias faltantes):")
            for nombre, motivo in strategy_loader.fallidos.items():
                logger.warning(f"    {nombre}: {motivo}")

        metricas = PerformanceMetrics(tasa_libre_riesgo_anual=0.02)

        juez_builder = JuezBuilder(self._config)
        juez = juez_builder.construir()

        return zoo, metricas, juez

    def _fase_ejecucion(
        self,
        zoo: ZooManager,
        metricas: PerformanceMetrics,
        juez: TTTJuez,
        datos: pd.DataFrame,
        benchmark: pd.Series,
    ) -> ResultadoBacktest:
        engine = BacktestEngine(
            zoo=zoo,
            metricas=metricas,
            juez=juez,
            datos=datos,
            benchmark=benchmark,
            kalman_config={
                "Q": self._config.kalman_Q,
                "R": self._config.kalman_R,
            },
            ventana_metricas=self._config.ventana_metricas,
            metrica_ranking=self._config.metrica_ranking,
        )

        logger.info(
            f"Iniciando walk-forward | "
            f"[{self._config.fecha_inicio} → {self._config.fecha_fin}] | "
            f"frecuencia={self._config.frecuencia_rebalanceo}d | "
            f"estrategias={zoo.nombres()}"
        )

        resultado = engine.ejecutar_walk_forward(
            fecha_inicio=pd.Timestamp(self._config.fecha_inicio),
            fecha_fin=pd.Timestamp(self._config.fecha_fin),
            frecuencia_rebalanceo=self._config.frecuencia_rebalanceo,
        )

        logger.info(
            f"Walk-forward completado | "
            f"días_meta={len(resultado.retornos_meta)} | "
            f"rebalanceos={len(resultado.fechas_rebalanceo)}"
        )
        return resultado

    def _fase_exportacion(
        self,
        resultado: ResultadoBacktest,
        juez: TTTJuez,
        benchmark: pd.Series,
        metricas: PerformanceMetrics,
    ) -> None:
        logger.info(f"Exportando resultados → {self._config.directorio_salida}")
        exporter = ResultsExporter(self._config)
        exporter.exportar(resultado, juez, benchmark, metricas)


# =============================================================================
# Utilidades internas
# =============================================================================

def _encabezado(titulo: str) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"  {titulo}")
    logger.info(sep)


def _imprimir_resumen(resumen: pd.DataFrame) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RESUMEN ESTADÍSTICO DEL BACKTEST")
    print(sep)
    print(resumen.to_string())
    print(f"{sep}\n")


# =============================================================================
# CLI — interfaz de línea de comandos
# =============================================================================

def _parse_args() -> PipelineConfig:
    """Parsea los argumentos de la CLI y retorna un PipelineConfig configurado."""
    parser = argparse.ArgumentParser(
        description="quant_arena — Backtest walk-forward del Zoo de estrategias",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inicio", default="2005-01-01",
        help="Fecha de inicio del backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fin", default="2023-12-31",
        help="Fecha de fin del backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--frecuencia", type=int, default=21,
        help="Días hábiles entre rebalanceos",
    )
    parser.add_argument(
        "--ventana", type=int, default=63,
        help="Ventana de métricas rolling (días hábiles)",
    )
    parser.add_argument(
        "--metrica", default="sharpe",
        choices=["sharpe", "sortino", "calmar", "information_ratio", "alpha_tstat"],
        help="Métrica de ranking para el Juez TTT",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.6,
        help="TTT: desv. estándar del prior de habilidad",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.036,
        help="TTT: velocidad de cambio de habilidad inter-período",
    )
    parser.add_argument(
        "--calibrar", action="store_true",
        help="Ejecutar OptimizadorTTT (L-BFGS-B) para ajustar sigma/gamma",
    )
    parser.add_argument(
        "--salida", default="./resultados_backtest",
        help="Directorio de salida para CSV y PDF",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="No generar el tear-sheet PDF",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="No exportar archivos CSV",
    )
    parser.add_argument(
        "--mostrar", action="store_true",
        help="Mostrar gráficos interactivos al finalizar",
    )
    parser.add_argument(
        "--estrategias", nargs="+", default=None,
        metavar="NOMBRE",
        help=(
            "Subconjunto de estrategias a incluir. "
            "Ej: --estrategias momentum_126d xgboost_trend olps_rmr"
        ),
    )

    args = parser.parse_args()

    return PipelineConfig(
        fecha_inicio=args.inicio,
        fecha_fin=args.fin,
        frecuencia_rebalanceo=args.frecuencia,
        ventana_metricas=args.ventana,
        metrica_ranking=args.metrica,
        ttt_sigma=args.sigma,
        ttt_gamma=args.gamma,
        calibrar_hiperparametros=args.calibrar,
        directorio_salida=Path(args.salida),
        generar_pdf=not args.no_pdf,
        guardar_csv=not args.no_csv,
        mostrar_graficos=args.mostrar,
        estrategias_habilitadas=args.estrategias,
    )


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    config = _parse_args()
    pipeline = SimulationPipeline(config)
    pipeline.ejecutar()
