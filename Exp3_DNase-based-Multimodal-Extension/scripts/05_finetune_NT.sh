#!/usr/bin/env bash
# ============================================================
# 05_finetune_NT.sh
#
# Fine-tune Nucleotide Transformer v2 500M (multi-species) with
# DNase signal fusion for a single TF.
#
# Usage:
#   bash scripts/05_finetune_NT.sh CTCF
#   bash scripts/05_finetune_NT.sh BRD4 --batch_size 32
#
# Environment variables:
#   DATASET_DIR  – imbalanced HDF5 dataset root
#   NT_PATH      – NT model path or HuggingFace ID
#   CONDA_RUN    – conda run prefix
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_DIR="${DATASET_DIR:-/dataset/zjn_zjj/DLM/10_21_previous_work/Data_associated_with_Graduation/DNase_implement/Data/dataset/imbalanced}"
NT_PATH="${NT_PATH:-InstaDeepAI/nucleotide-transformer-v2-500m-multi-species}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <peak_type> [options...]" >&2; exit 1
fi
PEAK_TYPE="$1"; shift

BATCH_SIZE=64
LEARNING_RATE="2e-5"
NUM_EPOCHS=3
FREEZE_BACKBONE=true
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --learning_rate)  LEARNING_RATE="$2";  shift 2 ;;
        --num_epochs)     NUM_EPOCHS="$2";     shift 2 ;;
        --freeze_backbone) FREEZE_BACKBONE="$2"; shift 2 ;;
        --gpu)            GPU="$2";            shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

FREEZE_STR="frozen"; [ "$FREEZE_BACKBONE" = "false" ] && FREEZE_STR="unfrozen"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${PEAK_TYPE}/NT/NT_${FREEZE_STR}}"

TRAIN_DATA="${DATASET_DIR}/train/${PEAK_TYPE}_train_merged.h5"
VAL_DATA="${DATASET_DIR}/val/${PEAK_TYPE}_val_merged.h5"
TEST_DATA="${DATASET_DIR}/test/${PEAK_TYPE}_test_GM12878.h5"

for f in "$TRAIN_DATA" "$VAL_DATA" "$TEST_DATA"; do
    [ -f "$f" ] || { echo "Error: $f not found" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"
echo "=== NT fine-tuning | TF=$PEAK_TYPE | frozen=$FREEZE_BACKBONE ==="

CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false \
$CONDA_RUN python -m exp3_dnase.finetune.finetune_NT \
    --model_name "NT"                \
    --model_path "$NT_PATH"          \
    --train_data_path "$TRAIN_DATA"  \
    --val_data_path   "$VAL_DATA"    \
    --test_data_path  "$TEST_DATA"   \
    --peak_type       "$PEAK_TYPE"   \
    --output_dir      "$OUTPUT_DIR"  \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size  $((BATCH_SIZE * 2)) \
    --learning_rate   "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS"   \
    --freeze_backbone "$FREEZE_BACKBONE" \
    --fp16 false                     \
    --dataloader_pin_memory false    \
    --report_to none                 \
    --logging_steps 100              \
    --save_steps 500                 \
    --eval_steps 500

echo "=== Done. Results: $OUTPUT_DIR/results/ ==="
