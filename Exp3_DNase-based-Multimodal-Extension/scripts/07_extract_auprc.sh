#!/usr/bin/env bash
# ============================================================
# 07_extract_auprc.sh
#
# Aggregate AUPRC values from all trained models into a summary CSV.
#
# Usage:
#   bash scripts/07_extract_auprc.sh
#   bash scripts/07_extract_auprc.sh /custom/outputs/root
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"
OUTPUTS_DIR="${1:-${SCRIPT_DIR}/outputs}"

echo "Extracting AUPRC from: $OUTPUTS_DIR"

$CONDA_RUN python -m exp3_dnase.analysis.extract_auprc \
    --outputs_dir "$OUTPUTS_DIR"

echo "Done. Summary CSV written to: ${OUTPUTS_DIR}/auprc_summary_model_rows.csv"
