#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-DNABERT2_5.6}
DATA_DIR=${DATA_DIR:-Data/processed_data}
STAT_ROOT=${STAT_ROOT:-outputs/motif_attention_stats}
DISCOVERY_ROOT=${DISCOVERY_ROOT:-${EXP2_DISCOVERY_ROOT:-${RECOVERY_ROOT:-${EXP2_RECOVERY_ROOT:-outputs/motif_discovery}}}}
TF_LIST_CSV=${TF_LIST_CSV:-CTCF,FOXA1,GATA1,GATA4,JUN,LDB1,MEF2A,MYC,NRF1,SPI1,USF2,YY1}

python -m exp2_attention.visualization.visualize_attention   --all_tfs   --model_type "${MODEL_TYPE}"   --aggregation_mode max_head   --input_type threshold   --data_source motif

python -m exp2_attention.visualization.plot_motif_score_heatmaps   --results_dir "${STAT_ROOT}"   --output_dir "${STAT_ROOT}/visualize/heatmap"   --tf_name CTCF

python -m exp2_attention.discovery.plot_all_discovered_vs_jaspar_heatmaps   --heatmap_dir "${DISCOVERY_ROOT}/heatmaps"   --tfs "${TF_LIST_CSV}"   --out_prefix "${DISCOVERY_ROOT}/heatmaps/all_tfs_discovered_vs_jaspar_heatmap"
