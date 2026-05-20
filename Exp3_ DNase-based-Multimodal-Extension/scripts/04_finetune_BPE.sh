#!/usr/bin/env bash
# ============================================================
# 04_finetune_BPE.sh
#
# Fine-tune BPE-tokenised DNA language models (DNABERT-2, GROVER,
# GENA-LM-BERT, GENA-LM-BigBird) with DNase signal fusion.
#
# Usage:
#   bash scripts/04_finetune_BPE.sh CTCF --model_name GROVER
#   bash scripts/04_finetune_BPE.sh BRD4 --model_name DNABERT2 --batch_size 64
#
# Environment variables:
#   DATASET_DIR        – imbalanced HDF5 dataset root
#   MODEL_FILES        – root of local model files
#   GFM_MODEL_USE      – path to GENA_LM source repo (for GENA_LM models)
#   CONDA_RUN          – conda run prefix
#   DNABERT2_PATH      – override DNABERT-2 model path
#   GROVER_PATH        – override GROVER model path
#   GENA_LM_BERT_PATH  – override GENA-LM-BERT model path
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export DATASET_DIR="${DATASET_DIR:-/dataset/zjn_zjj/DLM/10_21_previous_work/Data_associated_with_Graduation/DNase_implement/Data/dataset/imbalanced}"
MODEL_FILES="${MODEL_FILES:-/dataset/zjn_zjj/DLM/GFM_model_files}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n glm_hf}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <peak_type> [--model_name MODEL] [options...]" >&2; exit 1
fi
PEAK_TYPE="$1"; shift

# Defaults
MODEL_NAME="GROVER"
BATCH_SIZE=64
LEARNING_RATE="2e-5"
NUM_EPOCHS=3
FREEZE_BACKBONE=true
MAX_LENGTH=1002
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)     MODEL_NAME="$2";     shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --learning_rate)  LEARNING_RATE="$2";  shift 2 ;;
        --num_epochs)     NUM_EPOCHS="$2";     shift 2 ;;
        --freeze_backbone) FREEZE_BACKBONE="$2"; shift 2 ;;
        --max_length)     MAX_LENGTH="$2";     shift 2 ;;
        --gpu)            GPU="$2";            shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Resolve model path
case "$MODEL_NAME" in
    DNABERT2)
        MODEL_PATH="${DNABERT2_PATH:-${MODEL_FILES}/DNABERT-2-117M}"
        ;;
    GROVER)
        MODEL_PATH="${GROVER_PATH:-${MODEL_FILES}/GROVER}"
        ;;
    GENA_LM_BERT)
        MODEL_PATH="${GENA_LM_BERT_PATH:-${MODEL_FILES}/GENA_LM_BERT}"
        ;;
    GENA_LM_BigBird)
        MODEL_PATH="${GENA_LM_BIGBIRD_PATH:-${MODEL_FILES}/GENA_LM_BigBird}"
        ;;
    *)
        echo "Error: unsupported model '$MODEL_NAME'. Choices: DNABERT2 | GROVER | GENA_LM_BERT | GENA_LM_BigBird" >&2
        exit 1
        ;;
esac

FREEZE_STR="frozen"
[ "$FREEZE_BACKBONE" = "false" ] && FREEZE_STR="unfrozen"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${PEAK_TYPE}/BPE/${MODEL_NAME}_${FREEZE_STR}}"

TRAIN_DATA="${DATASET_DIR}/train/${PEAK_TYPE}_train_merged.h5"
VAL_DATA="${DATASET_DIR}/val/${PEAK_TYPE}_val_merged.h5"
TEST_DATA="${DATASET_DIR}/test/${PEAK_TYPE}_test_GM12878.h5"

for f in "$TRAIN_DATA" "$VAL_DATA" "$TEST_DATA"; do
    [ -f "$f" ] || { echo "Error: $f not found" >&2; exit 1; }
done
[ -d "$MODEL_PATH" ] || { echo "Error: model path not found: $MODEL_PATH" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
echo "=== BPE fine-tuning | TF=$PEAK_TYPE | model=$MODEL_NAME | frozen=$FREEZE_BACKBONE ==="
echo "    model path : $MODEL_PATH"
echo "    output dir : $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false \
$CONDA_RUN python -m exp3_dnase.finetune.finetune_BPE \
    --model_name      "$MODEL_NAME"     \
    --model_path      "$MODEL_PATH"     \
    --train_data_path "$TRAIN_DATA"     \
    --val_data_path   "$VAL_DATA"       \
    --test_data_path  "$TEST_DATA"      \
    --peak_type       "$PEAK_TYPE"      \
    --output_dir      "$OUTPUT_DIR"     \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size  $((BATCH_SIZE * 2)) \
    --learning_rate   "$LEARNING_RATE"  \
    --num_train_epochs "$NUM_EPOCHS"    \
    --freeze_backbone "$FREEZE_BACKBONE" \
    --model_max_length "$MAX_LENGTH"    \
    --fp16 false                        \
    --dataloader_pin_memory false       \
    --report_to none                    \
    --logging_steps 100                 \
    --save_steps 300                    \
    --eval_steps 300

echo "=== Done. Results: $OUTPUT_DIR/results/ ==="
