#!/usr/bin/env bash
# Renderiza las 27 escenas en orden. Uso:
#   ./render_all.sh draft   -> calidad baja, rápido, para iterar
#   ./render_all.sh final   -> calidad alta 1080p60, para el congreso
set -euo pipefail
cd "$(dirname "$0")"

MODO="${1:-draft}"
if [ "$MODO" = "final" ]; then
    FLAG="-qh"   # high quality, 1080p60
else
    FLAG="-ql"   # low quality, rápido para revisar timing/contenido
fi

declare -A ESCENAS=(
    [escenas/escena_01_apertura.py]=Apertura
    [escenas/escena_02_split_sharpe.py]=SplitSharpe
    [escenas/escena_03_ranking_tabla.py]=RankingTabla
    [escenas/escena_04_tres_problemas.py]=TresProblemas
    [escenas/escena_05_ttt_titulo.py]=TTTTitulo
    [escenas/escena_06_ajedrez.py]=Ajedrez
    [escenas/escena_07_gaussianas.py]=Gaussianas
    [escenas/escena_08_ep_flecha.py]=EPFlecha
    [escenas/escena_09_mu_sobre_sigma.py]=MuSobreSigma
    [escenas/escena_10_kalman.py]=Kalman
    [escenas/escena_11_adapter.py]=Adapter
    [escenas/escena_12_pregunta_supuestos.py]=PreguntaSupuestos
    [escenas/escena_13_tabla_normalidad.py]=TablaNormalidad
    [escenas/escena_14_tabla_estacionariedad_k.py]=TablaKOptimo
    [escenas/escena_15_garch_dist_t.py]=GarchDistT
    [escenas/escena_16_texto_riesgo.py]=TextoRiesgo
    [escenas/escena_17_equity_riskoverlay.py]=EquityRiskOverlay
    [escenas/escena_18_kelly_ecuacion.py]=KellyEcuacion
    [escenas/escena_19_bug_fix.py]=BugFix
    [escenas/escena_20_kappa_nota.py]=KappaNota
    [escenas/escena_21_ajedrez_pool.py]=AjedrezPool
    [escenas/escena_22_crowding_ecuacion.py]=CrowdingEcuacion
    [escenas/escena_23_feedback_loop.py]=FeedbackLoop
    [escenas/escena_24_tabla_resumen.py]=TablaResumen
    [escenas/escena_25_limitaciones.py]=Limitaciones
    [escenas/escena_26_cierre.py]=Cierre
    [escenas/escena_27_logo.py]=Logo
)

# Orden explícito (los diccionarios bash no garantizan orden de inserción)
ORDEN=(
    escena_00_glosario:Glosario
    escena_01_apertura:Apertura
    escena_02_split_sharpe:SplitSharpe
    escena_03_ranking_tabla:RankingTabla
    escena_04_tres_problemas:TresProblemas
    escena_05_ttt_titulo:TTTTitulo
    escena_06_ajedrez:Ajedrez
    escena_07_gaussianas:Gaussianas
    escena_08_ep_flecha:EPFlecha
    escena_09_mu_sobre_sigma:MuSobreSigma
    escena_10_kalman:Kalman
    escena_11_adapter:Adapter
    escena_12_pregunta_supuestos:PreguntaSupuestos
    escena_12b_hmm_garch_glosa:HMMGarchGlosa
    escena_13_tabla_normalidad:TablaNormalidad
    escena_14_tabla_estacionariedad_k:TablaKOptimo
    escena_15_garch_dist_t:GarchDistT
    escena_16_texto_riesgo:TextoRiesgo
    escena_17_equity_riskoverlay:EquityRiskOverlay
    escena_18_kelly_ecuacion:KellyEcuacion
    escena_19_bug_fix:BugFix
    escena_20_kappa_nota:KappaNota
    escena_21_ajedrez_pool:AjedrezPool
    escena_22_crowding_ecuacion:CrowdingEcuacion
    escena_23_feedback_loop:FeedbackLoop
    escena_24_tabla_resumen:TablaResumen
    escena_25_limitaciones:Limitaciones
    escena_26_cierre:Cierre
    escena_27_logo:Logo
)

rm -f lista_escenas.txt
for par in "${ORDEN[@]}"; do
    archivo="${par%%:*}"
    clase="${par##*:}"
    echo ">>> Renderizando ${clase} (${archivo}.py) [${MODO}]"
    manim $FLAG "escenas/${archivo}.py" "${clase}"
    ruta_mp4=$(find media -path "*${clase}.mp4" | sort | tail -1)
    echo "file '$(realpath "$ruta_mp4")'" >> lista_escenas.txt
done

echo ""
echo "Listo. lista_escenas.txt generado con las ${#ORDEN[@]} escenas en orden."
echo "Para unirlas: ./concat_escenas.sh"
