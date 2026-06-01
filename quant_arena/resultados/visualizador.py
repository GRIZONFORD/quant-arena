# =============================================================================
# FILE: quant_arena/resultados/visualizador.py
# Capa de presentación visual institucional — Tear Sheet cuantitativo
# =============================================================================
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns

from quant_arena.backtesting.motor import ResultadoBacktest
from quant_arena.metricas.performance_metrics import PerformanceMetrics


# =============================================================================
# Constantes de estilo institucional
# =============================================================================

_COLOR_META:      str = '#0D3B8C'   # azul marino profundo (meta-portafolio)
_COLOR_BENCH:     str = '#757575'   # gris medio (benchmark)
_COLOR_DD:        str = '#C62828'   # rojo oscuro (drawdown)
_COLOR_DD_FILL:   str = '#FFCDD2'   # rojo claro (relleno drawdown)
_COLOR_BRUTO:     str = '#90A4AE'   # gris azulado (señal ruidosa)
_COLOR_KALMAN:    str = '#37474F'   # gris oscuro (señal filtrada)

_FIGSIZE_WIDE:    Tuple[float, float] = (16.0, 9.0)
_FIGSIZE_TALL:    Tuple[float, float] = (14.0, 8.0)
_FIGSIZE_MEDIUM:  Tuple[float, float] = (14.0, 5.5)
_DPI_PANTALLA:    int = 110
_DPI_PDF:         int = 200

# Paleta de estrategias: tab10 de seaborn, accesible y distinguible
_PALETA_ZOO = sns.color_palette('tab10', n_colors=10)

# Estilo matplotlib: prueba versiones nuevas y antiguas de seaborn-styles
_ESTILO_CANDIDATOS = [
    'seaborn-v0_8-whitegrid',
    'seaborn-whitegrid',
    'ggplot',
]
_ESTILO_ACTIVO = next(
    (s for s in _ESTILO_CANDIDATOS if s in plt.style.available),
    'default',
)


# =============================================================================
# Funciones utilitarias de cálculo (stateless, module-level)
# =============================================================================

def _nav(retornos: pd.Series) -> pd.Series:
    """NAV acumulado normalizado a 1.0 en la primera observación."""
    nav = (1.0 + retornos.fillna(0.0)).cumprod()
    if len(nav) > 0:
        nav = nav / nav.iloc[0]
    return nav


def _drawdown_serie(nav: pd.Series) -> pd.Series:
    """
    Calcula el drawdown relativo al pico histórico.
    Resultado en [-1, 0]: valores negativos representan caídas.
    """
    pico = nav.cummax()
    dd = (nav - pico) / pico.replace(0.0, np.nan)
    return dd.fillna(0.0)


def _formatear_fechas(ax: plt.Axes) -> None:
    """Aplica formato de fecha legible al eje x."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)


def _anotar_max_dd(ax: plt.Axes, dd: pd.Series) -> None:
    """Anota el punto de máximo drawdown con una flecha."""
    if dd.empty or dd.min() == 0:
        return
    idx_min = dd.idxmin()
    val_min = dd.min()
    ax.annotate(
        f'MDD {val_min:.1%}',
        xy=(idx_min, val_min),
        xytext=(15, 10),
        textcoords='offset points',
        fontsize=7.5,
        color=_COLOR_DD,
        arrowprops=dict(arrowstyle='->', color=_COLOR_DD, lw=0.8),
    )


# =============================================================================
# Clase principal
# =============================================================================

class ReporteCuantitativo:
    """
    Generador de reportes visuales institucionales para resultados de backtest.

    Encapsula toda la lógica de ploteo. Cada método devuelve una ``Figure``
    independiente sin mostrarla ni cerrarla — la gestión del ciclo de vida
    es responsabilidad del llamador o de ``generar_tearsheet()``.

    Ejemplo mínimo::

        reporte = ReporteCuantitativo(resultado, benchmark=sp500_ret, juez=ttt)
        fig = reporte.plot_equity_curve()
        fig.savefig('equity.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        reporte.generar_tearsheet('informe.pdf')
    """

    def __init__(
        self,
        resultado: ResultadoBacktest,
        benchmark: Optional[pd.Series] = None,
        juez: Optional[object] = None,
        dpi: int = _DPI_PANTALLA,
    ) -> None:
        """
        Args:
            resultado:  Objeto ``ResultadoBacktest`` producido por ``BacktestEngine``.
            benchmark:  Serie de retornos diarios del S&P 500 (opcional).
                        Si se provee, se añade a la curva de equity como referencia.
            juez:       Instancia de ``TTTJuez`` para graficar curvas de aprendizaje.
                        Requerido solo por ``plot_ttt_learning_curves()``.
            dpi:        Resolución de las figuras (pantalla: 110, PDF: 200).
        """
        self._r    = resultado
        self._bench = benchmark
        self._juez  = juez
        self._dpi   = dpi

        cols = list(resultado.retornos_estrategias.columns)
        self._colores: Dict[str, tuple] = {
            nombre: _PALETA_ZOO[i % 10] for i, nombre in enumerate(cols)
        }

    # ------------------------------------------------------------------
    # 1. Equity Curve + Underwater
    # ------------------------------------------------------------------

    def plot_equity_curve(
        self,
        log_scale: bool = False,
        mostrar_estrategias: bool = True,
        mostrar_tabla_kpi: bool = True,
    ) -> Figure:
        """
        Panel superior: curvas de riqueza acumuladas (NAV normalizado a 1).
        Panel inferior: gráfico underwater (drawdown del meta-portafolio).

        Args:
            log_scale:             Si True, escala logarítmica en eje Y del NAV.
            mostrar_estrategias:   Si True, superpone las curvas individuales del Zoo.
            mostrar_tabla_kpi:     Si True, añade una caja de texto con KPIs clave.

        Returns:
            Figure de matplotlib con dos subplots (equity arriba, drawdown abajo).
        """
        with plt.style.context(_ESTILO_ACTIVO):
            fig = plt.figure(figsize=_FIGSIZE_WIDE, dpi=self._dpi, tight_layout=False)
            gs  = GridSpec(3, 1, figure=fig, hspace=0.05,
                           height_ratios=[3.5, 1, 0.01])
            ax_eq = fig.add_subplot(gs[0])
            ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

            nav_meta = _nav(self._r.retornos_meta)

            # ── Estrategias individuales (fondo, tenues) ──────────────────────
            if mostrar_estrategias and not self._r.retornos_estrategias.empty:
                navs_e = self._r.equity_curves_estrategias()
                for col in navs_e.columns:
                    ax_eq.plot(
                        navs_e.index, navs_e[col],
                        color=self._colores.get(col, 'gray'),
                        lw=0.9, alpha=0.30, zorder=2,
                    )

            # ── Benchmark ────────────────────────────────────────────────────
            if self._bench is not None and not self._bench.empty:
                nav_b = _nav(self._bench.reindex(nav_meta.index).fillna(0.0))
                ax_eq.plot(
                    nav_b.index, nav_b.values,
                    color=_COLOR_BENCH, lw=1.4, ls='--',
                    label='S&P 500', zorder=3,
                )

            # ── Meta-portafolio (protagonista) ────────────────────────────────
            ax_eq.plot(
                nav_meta.index, nav_meta.values,
                color=_COLOR_META, lw=2.2, label='Meta-Portfolio', zorder=4,
            )

            # ── Drawdown ──────────────────────────────────────────────────────
            dd = _drawdown_serie(nav_meta)
            ax_dd.fill_between(
                dd.index, dd.values, 0.0,
                color=_COLOR_DD_FILL, alpha=0.9, zorder=2,
            )
            ax_dd.plot(dd.index, dd.values, color=_COLOR_DD, lw=0.9, zorder=3)
            _anotar_max_dd(ax_dd, dd)
            ax_dd.set_ylabel('Drawdown', fontsize=9)
            ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax_dd.set_ylim(min(dd.min() * 1.15, -0.01), 0.01)

            # ── Leyenda dinámica (estrategias) ────────────────────────────────
            if mostrar_estrategias and not self._r.retornos_estrategias.empty:
                for col in self._r.retornos_estrategias.columns:
                    ax_eq.plot([], [], color=self._colores.get(col, 'gray'),
                               lw=1.5, alpha=0.7, label=col)

            # ── KPI caja de texto ─────────────────────────────────────────────
            if mostrar_tabla_kpi and len(self._r.retornos_meta) >= 20:
                self._caja_kpi(ax_eq, nav_meta, self._bench)

            # ── Estética ──────────────────────────────────────────────────────
            if log_scale:
                ax_eq.set_yscale('log')
                ax_eq.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda y, _: f'{y:.1f}×')
                )
            else:
                ax_eq.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda y, _: f'{y:.2f}')
                )

            ax_eq.set_ylabel('NAV (base 1)', fontsize=10)
            ax_eq.set_title(
                'Curva de Capital — Meta-Portafolio vs. Benchmark',
                fontsize=13, fontweight='bold', pad=10,
            )
            ax_eq.legend(loc='upper left', fontsize=8.5, framealpha=0.8)
            ax_eq.tick_params(labelbottom=False)

            _formatear_fechas(ax_dd)
            fig.align_ylabels([ax_eq, ax_dd])

        return fig

    # ------------------------------------------------------------------
    # 2. Curvas de Aprendizaje TTT (μ ± σ)
    # ------------------------------------------------------------------

    def plot_ttt_learning_curves(self) -> Figure:
        """
        Evolución temporal de la habilidad latente del Juez para cada estrategia.

        Muestra ``μ`` como línea sólida y la banda de confianza ``[μ−σ, μ+σ]``
        como área semitransparente. Requiere que ``juez`` haya sido pasado al
        constructor.

        Returns:
            Figure de matplotlib con un panel por cada grupo de estrategias.

        Raises:
            ValueError: Si no se pasó un TTTJuez al constructor.
        """
        if self._juez is None:
            raise ValueError(
                "plot_ttt_learning_curves() requiere pasar juez=<TTTJuez> al constructor."
            )

        curvas_df: pd.DataFrame = self._juez.exportar_curvas_df()

        if curvas_df.empty:
            warnings.warn("TTTJuez no tiene curvas de aprendizaje. ¿Se llamó actualizar()?")
            fig, ax = plt.subplots(figsize=_FIGSIZE_MEDIUM, dpi=self._dpi)
            ax.text(0.5, 0.5, 'Sin datos de curvas TTT',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        estrategias = curvas_df['estrategia'].unique().tolist()
        n = len(estrategias)

        with plt.style.context(_ESTILO_ACTIVO):
            fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=self._dpi)

            for i, nombre in enumerate(estrategias):
                grupo = curvas_df[curvas_df['estrategia'] == nombre].copy()
                # tiempo = días desde época Unix → convertir a fechas
                grupo['fecha'] = pd.to_datetime(
                    grupo['tiempo'] * 86_400, unit='s'
                ).dt.normalize()
                grupo = grupo.sort_values('fecha')

                color = _PALETA_ZOO[i % 10]
                ax.plot(
                    grupo['fecha'], grupo['mu'],
                    color=color, lw=2.0, label=nombre, zorder=3,
                )
                ax.fill_between(
                    grupo['fecha'], grupo['banda_inf'], grupo['banda_sup'],
                    color=color, alpha=0.15, zorder=2,
                )

            ax.axhline(0, color='#90A4AE', lw=0.8, ls='--', zorder=1)
            ax.set_ylabel('Habilidad latente μ', fontsize=10)
            ax.set_title(
                'Curvas de Aprendizaje TTT — μ ± 1σ por Estrategia',
                fontsize=13, fontweight='bold',
            )
            ax.legend(loc='best', fontsize=9, framealpha=0.8)
            _formatear_fechas(ax)

        return fig

    # ------------------------------------------------------------------
    # 3. Asignación Dinámica del Juez (stacked area)
    # ------------------------------------------------------------------

    def plot_asignacion_dinamica(self) -> Figure:
        """
        Stacked area chart que muestra la rotación de capital del Juez entre
        las estrategias del Zoo en cada fecha de rebalanceo.

        Returns:
            Figure de matplotlib con un gráfico de área apilada.
        """
        pesos = self._r.pesos_juez

        if pesos is None or pesos.empty:
            warnings.warn("pesos_juez está vacío: TTT no actualizó durante el backtest.")
            fig, ax = plt.subplots(figsize=_FIGSIZE_MEDIUM, dpi=self._dpi)
            ax.text(0.5, 0.5, 'Sin datos de asignación del Juez',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        pesos = pesos.fillna(0.0)
        # Ordenar estrategias por asignación media (mayor → base del stack)
        orden = pesos.mean().sort_values(ascending=False).index.tolist()
        pesos = pesos[orden]

        colores = [_PALETA_ZOO[i % 10] for i in range(len(orden))]

        with plt.style.context(_ESTILO_ACTIVO):
            fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=self._dpi)

            ax.stackplot(
                pesos.index,
                [pesos[col].values for col in orden],
                labels=orden,
                colors=colores,
                alpha=0.82,
            )

            # Líneas de rebalanceo
            for fecha in pesos.index:
                ax.axvline(fecha, color='white', lw=0.4, alpha=0.5, zorder=0)

            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_ylim(0, 1.01)
            ax.set_ylabel('Asignación de capital', fontsize=10)
            ax.set_title(
                'Asignación Dinámica del Juez TTT por Estrategia',
                fontsize=13, fontweight='bold',
            )
            ax.legend(
                loc='upper left', fontsize=8.5,
                framealpha=0.85, ncol=min(len(orden), 4),
            )
            _formatear_fechas(ax)

        return fig

    # ------------------------------------------------------------------
    # 4. Señal Kalman vs. Rolling bruta
    # ------------------------------------------------------------------

    def plot_señal_kalman(self) -> Figure:
        """
        Compara la señal de métrica ruidosa (rolling) con la señal filtrada
        por el Filtro de Kalman para cada estrategia del Zoo.

        Permite evaluar visualmente el efecto del suavizado sobre el ruido
        de las estimaciones periódicas.

        Returns:
            Figure de matplotlib con una sub-figura por estrategia.
        """
        rolling = self._r.metricas_rolling
        kalman  = self._r.metricas_kalman

        if rolling.empty:
            fig, ax = plt.subplots(figsize=_FIGSIZE_MEDIUM, dpi=self._dpi)
            ax.text(0.5, 0.5, 'Sin datos de métricas rolling',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        estrategias = rolling.columns.tolist()
        n = len(estrategias)
        ncols = min(n, 2)
        nrows = int(np.ceil(n / ncols))

        with plt.style.context(_ESTILO_ACTIVO):
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(14.0, 4.0 * nrows),
                dpi=self._dpi,
                squeeze=False,
            )
            fig.suptitle(
                f'Señal {self._r.metrica_usada.capitalize()} — '
                f'Rolling (ruidoso) vs. Kalman (filtrado)',
                fontsize=13, fontweight='bold', y=1.01,
            )

            for idx, nombre in enumerate(estrategias):
                ax = axes[idx // ncols][idx % ncols]
                color = _PALETA_ZOO[idx % 10]

                serie_raw = rolling[nombre].dropna()
                ax.plot(
                    serie_raw.index, serie_raw.values,
                    color=_COLOR_BRUTO, lw=0.9, ls='--',
                    marker='o', markersize=3, alpha=0.7,
                    label='Rolling (bruto)',
                )

                if nombre in kalman.columns:
                    serie_k = kalman[nombre].dropna()
                    ax.plot(
                        serie_k.index, serie_k.values,
                        color=color, lw=2.0, label='Kalman (filtrado)',
                    )

                ax.axhline(0, color='#90A4AE', lw=0.6, ls=':')
                ax.set_title(nombre, fontsize=10, fontweight='bold')
                ax.set_ylabel(self._r.metrica_usada, fontsize=8)
                ax.legend(fontsize=7.5)
                _formatear_fechas(ax)

            # Ocultar subplots vacíos
            for idx in range(n, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # 5. Tear Sheet — orquestador PDF
    # ------------------------------------------------------------------

    def generar_tearsheet(
        self,
        ruta_salida: Union[str, Path],
        log_scale: bool = False,
        incluir_curvas_ttt: bool = True,
    ) -> Path:
        """
        Ensambla todas las gráficas en un archivo PDF multipágina de calidad
        institucional. Cierra cada figura inmediatamente después de guardarla
        para evitar fugas de memoria (crítico en loops de backtesting).

        Args:
            ruta_salida:          Ruta del archivo PDF de salida.
            log_scale:            Si True, escala logarítmica en el equity curve.
            incluir_curvas_ttt:   Si True y hay un juez disponible, añade la
                                  página de curvas de aprendizaje.

        Returns:
            Objeto ``Path`` con la ruta del PDF generado.
        """
        ruta = Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # Construir lista de (titulo_página, callable)
        paginas: List[Tuple[str, object]] = [
            ('Curva de Capital',         lambda: self.plot_equity_curve(log_scale=log_scale)),
            ('Asignación Dinámica',      lambda: self.plot_asignacion_dinamica()),
            ('Señal Kalman vs. Rolling', lambda: self.plot_señal_kalman()),
        ]
        if incluir_curvas_ttt and self._juez is not None:
            paginas.append(
                ('Curvas de Aprendizaje TTT', lambda: self.plot_ttt_learning_curves())
            )

        with PdfPages(str(ruta)) as pdf:
            for titulo, fn_plot in paginas:
                try:
                    fig = fn_plot()
                    pdf.savefig(fig, dpi=_DPI_PDF, bbox_inches='tight')
                except Exception as exc:
                    warnings.warn(f"Página '{titulo}' no pudo generarse: {exc}")
                finally:
                    # CRÍTICO: siempre cerrar para liberar memoria
                    try:
                        plt.close(fig)
                    except Exception:
                        pass

            # Metadatos del PDF
            info = pdf.infodict()
            info['Title']   = 'quant_arena — Tear Sheet'
            info['Subject'] = 'Walk-Forward Backtest Report'
            info['Author']  = 'quant_arena v1.0'

        return ruta

    # ------------------------------------------------------------------
    # Utilidades privadas de anotación
    # ------------------------------------------------------------------

    @staticmethod
    def _caja_kpi(
        ax: plt.Axes,
        nav: pd.Series,
        bench: Optional[pd.Series],
    ) -> None:
        """
        Añade una caja de texto con KPIs clave en la esquina superior derecha.
        Calcula métricas directamente desde el NAV sin necesitar PerformanceMetrics.
        """
        n = len(nav)
        retornos = nav.pct_change().dropna()
        if len(retornos) < 5:
            return

        cagr = nav.iloc[-1] ** (252.0 / n) - 1.0
        vol  = retornos.std(ddof=1) * np.sqrt(252)
        sr   = (retornos.mean() / retornos.std(ddof=1)) * np.sqrt(252) if retornos.std() > 0 else 0
        dd   = _drawdown_serie(nav).min()

        lines = [
            f'CAGR:   {cagr:+.1%}',
            f'Vol:    {vol:.1%}',
            f'Sharpe: {sr:.2f}',
            f'MDD:    {dd:.1%}',
        ]
        texto = '\n'.join(lines)

        ax.text(
            0.985, 0.96, texto,
            transform=ax.transAxes,
            fontsize=8.5, family='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='white', edgecolor='#BDBDBD',
                alpha=0.88,
            ),
        )
