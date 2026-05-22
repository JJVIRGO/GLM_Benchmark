#!/usr/bin/env bash
# ============================================================
# 01_create_datasets.sh
#
# Build per-cell-line HDF5 datasets (sequence + DNase signal + labels).
# Runs in two phases:
#   1. precompute – extract sequences and DNase signals for each cell
#   2. create     – attach per-TF labels to the precomputed base data
#
# Environment variables:
#   DATA_ROOT       – root of the Data directory
#   CONDA_RUN       – conda run prefix (default: conda run -n glm_hf)
#
# Usage:
#   bash scripts/01_create_datasets.sh
#   bash scripts/01_create_datasets.sh K562 CTCF   # single cell + TF
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"

CELL_LINES=("GM12878" "HepG2" "K562" "Lung")
TF_TYPES=("BRD4" "CTCF" "EZH2" "GABPA" "POLR2A" "USF2")

if [ $# -ge 1 ]; then CELL_LINES=("$1"); fi
if [ $# -ge 2 ]; then TF_TYPES=("$2"); fi

PYTHON="$CONDA_RUN python -m exp3_dnase.preprocess.create_peak_datasets"

for CELL in "${CELL_LINES[@]}"; do
    for SPLIT in train test; do
        echo "==> Precomputing $CELL ($SPLIT) ..."
        $PYTHON --mode precompute --cell_line "$CELL" --train_or_test "$SPLIT"
    done

    for TF in "${TF_TYPES[@]}"; do
        echo "==> Creating train/val datasets for $CELL / $TF ..."
        $PYTHON --mode create --cell_line "$CELL" --peak_type "$TF" --train_or_test train
        echo "==> Creating test dataset for $CELL / $TF ..."
        $PYTHON --mode create --cell_line "$CELL" --peak_type "$TF" --train_or_test test
    done
done

echo "Dataset creation complete."
