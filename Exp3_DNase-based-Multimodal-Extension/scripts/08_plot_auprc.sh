#!/usr/bin/env bash
# ============================================================
# 08_plot_auprc.sh
#
# Generate sequence-only vs. sequence+DNase AUPRC comparison
# plots (2×3 subplots, one per TF). Reads result JSON files
# from the original training output directory structure.
#
# Usage:
#   bash scripts/08_plot_auprc.sh
#   bash scripts/08_plot_auprc.sh /path/to/DNase_implement /output/dir
#
# Positional arguments:
#   $1  results_root  – root containing scripts/finetune/outputs/
#                       (default: original DNase_implement repo)
#   $2  output_dir    – where to save PDF figures (default: ./figures)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"

RESULTS_ROOT="${1:-/dataset/zjn_zjj/DLM/10_21_previous_work/Data_associated_with_Graduation/DNase_implement}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/figures}"

echo "Results root : $RESULTS_ROOT"
echo "Output dir   : $OUTPUT_DIR"

$CONDA_RUN python -m exp3_dnase.analysis.plot_auprc \
    --results_root "$RESULTS_ROOT" \
    --output_dir   "$OUTPUT_DIR"

echo "Plots saved to: $OUTPUT_DIR"
