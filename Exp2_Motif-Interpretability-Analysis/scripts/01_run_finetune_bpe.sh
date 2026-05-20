#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-dnabert2}
TF_NAME=${TF_NAME:?Set TF_NAME, e.g. CTCF}
DATA_ROOT=${DATA_ROOT:-Data/processed_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/finetune}
RUN_NAME=${RUN_NAME:-${MODEL_TYPE}_${TF_NAME}}
MODEL_PATH_ARG=()
if [[ -n "${MODEL_NAME_OR_PATH:-}" ]]; then
  MODEL_PATH_ARG=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
fi

python -m exp2_attention.finetune.train_bpe   --model_type "${MODEL_TYPE}"   "${MODEL_PATH_ARG[@]}"   --data_path "${DATA_ROOT}/${TF_NAME}"   --output_dir "${OUTPUT_ROOT}/${MODEL_TYPE}/motif_${TF_NAME}"   --run_name "${RUN_NAME}"   "$@"
