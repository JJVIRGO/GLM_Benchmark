#!/usr/bin/env bash
# ============================================================
# 03_finetune_basic.sh
#
# Fine-tune the from-scratch CNN-LSTM baseline model on DNase +
# sequence data for a single TF.
#
# Usage:
#   bash scripts/03_finetune_basic.sh CTCF
#   bash scripts/03_finetune_basic.sh BRD4 --batch_size 32 --num_epochs 10
#
# Environment variables:
#   DATASET_DIR  – path to imbalanced HDF5 dataset root
#   CONDA_RUN    – conda run prefix
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export DATASET_DIR="${DATASET_DIR:-/dataset/zjn_zjj/DLM/10_21_previous_work/Data_associated_with_Graduation/DNase_implement/Data/dataset/imbalanced}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <peak_type> [extra args...]" >&2; exit 1
fi
PEAK_TYPE="$1"; shift

# Defaults
BATCH_SIZE=16
LEARNING_RATE="1e-3"
NUM_EPOCHS=15
HIDDEN_DIM=256
GPU=0
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${PEAK_TYPE}/Basic/256d_unfrozen"

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --num_epochs)   NUM_EPOCHS="$2";   shift 2 ;;
        --hidden_dim)   HIDDEN_DIM="$2";   shift 2 ;;
        --gpu)          GPU="$2";          shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TRAIN_DATA="${DATASET_DIR}/train/${PEAK_TYPE}_train_merged.h5"
VAL_DATA="${DATASET_DIR}/val/${PEAK_TYPE}_val_merged.h5"
TEST_DATA="${DATASET_DIR}/test/${PEAK_TYPE}_test_GM12878.h5"

for f in "$TRAIN_DATA" "$VAL_DATA" "$TEST_DATA"; do
    [ -f "$f" ] || { echo "Error: $f not found" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"
echo "=== Basic CNN-LSTM fine-tuning | TF=$PEAK_TYPE | output=$OUTPUT_DIR ==="

CUDA_VISIBLE_DEVICES=$GPU $CONDA_RUN python -m exp3_dnase.finetune.finetune_basic \
    --train_data_path "$TRAIN_DATA"  \
    --val_data_path   "$VAL_DATA"    \
    --test_data_path  "$TEST_DATA"   \
    --peak_type       "$PEAK_TYPE"   \
    --output_dir      "$OUTPUT_DIR"  \
    --hidden_dim      "$HIDDEN_DIM"  \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size  $((BATCH_SIZE * 2)) \
    --learning_rate   "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS"   \
    --dataloader_pin_memory false    \
    --report_to none                 \
    --logging_steps 100              \
    --save_steps 500                 \
    --eval_steps 500

echo "=== Done. Results: $OUTPUT_DIR/results/ ==="
