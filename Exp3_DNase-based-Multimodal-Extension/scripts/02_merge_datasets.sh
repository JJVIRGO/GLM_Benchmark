#!/usr/bin/env bash
# ============================================================
# 02_merge_datasets.sh
#
# Merge per-cell-line HDF5 datasets into cross-cell training/
# validation splits and copy the GM12878 test set.
#
# Source cell lines : K562, HepG2, Lung
# Test cell line    : GM12878 (held-out)
# Output            : DATASET_DIR/imbalanced/{train,val,test}/
#
# Environment variables:
#   DATA_ROOT    – root of the Data directory
#   DATASET_DIR  – output directory (default: DATA_ROOT/dataset/imbalanced)
#   CONDA_RUN    – conda run prefix
#
# Usage:
#   bash scripts/02_merge_datasets.sh
#   bash scripts/02_merge_datasets.sh CTCF   # single TF
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data}"
export DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/dataset/imbalanced}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"

PEAK_ARG=""
if [ $# -ge 1 ]; then
    PEAK_ARG="--peak_type $1"
fi

echo "Merging cross-cell datasets ..."
echo "  Source: K562, HepG2, Lung"
echo "  Test  : GM12878"
echo "  Output: $DATASET_DIR"

$CONDA_RUN python -m exp3_dnase.preprocess.merge_cross_cell_datasets \
    --mode all \
    --source_cell_lines "K562,HepG2,Lung" \
    --output_dir "$DATASET_DIR" \
    $PEAK_ARG

echo "Merge complete. Dataset directory:"
ls -lh "$DATASET_DIR"
