#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false

MODEL_TYPE=${MODEL_TYPE:-DNABERT-2}
OUTPUT_MODEL_NAME=${OUTPUT_MODEL_NAME:-DNABERT2_5.6}
TF_LIST=${TF_LIST:-"CTCF FOXA1 GATA1 GATA4 JUN LDB1 MEF2A MYC NRF1 SPI1 USF2 YY1"}
DATA_DIR=${DATA_DIR:-Data/processed_data}
MODEL_ROOT=${MODEL_ROOT:-GLM_weights}
DISCOVERY_ROOT=${DISCOVERY_ROOT:-${EXP2_DISCOVERY_ROOT:-${RECOVERY_ROOT:-${EXP2_RECOVERY_ROOT:-outputs/motif_discovery}}}}
MODISCO_BIN=${MODISCO_BIN:-modisco}
MEME_DB=${MEME_DB:-Data/meme/JASPAR2024_CORE_non-redundant_pfms_meme.txt}
MAX_SEQLETS=${MAX_SEQLETS:-5000}
WINDOW=${WINDOW:-200}
TOP_MATCHES=${TOP_MATCHES:-3}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ROOT}/DNABERT-2-117M}

mkdir -p "${DISCOVERY_ROOT}/predict_true/${OUTPUT_MODEL_NAME}/attention"

for TF in ${TF_LIST}; do
  echo "================ Discovery ${OUTPUT_MODEL_NAME}/${TF} ================"
  train_csv="${DATA_DIR}/${TF}/train.csv"
  pred_csv="${DISCOVERY_ROOT}/predict_true/${OUTPUT_MODEL_NAME}/${TF}_train_true.csv"
  attention_file="${DISCOVERY_ROOT}/predict_true/${OUTPUT_MODEL_NAME}/attention/${TF}_attention_weight.parquet"
  attention_dir="${DISCOVERY_ROOT}/predict_true/${OUTPUT_MODEL_NAME}/attention/${TF}_attention_weight"
  files_dir="${DISCOVERY_ROOT}/predict_true/${OUTPUT_MODEL_NAME}/tfmodisco/${TF}/files"
  report_dir="${DISCOVERY_ROOT}/predict_true/${OUTPUT_MODEL_NAME}/tfmodisco/${TF}/report_cwm"
  mkdir -p "${files_dir}" "${report_dir}"

  if [[ ! -s "${train_csv}" ]]; then
    echo "[WARN] Missing train.csv for ${TF}; skip"
    continue
  fi

  if [[ ! -s "${pred_csv}" ]]; then
    python -m exp2_attention.discovery.predict_true       --tf_name "${TF}"       --model_type "${MODEL_TYPE}"       --output_model_name "${OUTPUT_MODEL_NAME}"       --gfm_root "${MODEL_ROOT}"       --input_path "${train_csv}"       --output_path "${pred_csv}"       --batch_size 256
  fi

  if [[ ! -d "${attention_dir}" ]]; then
    python -m exp2_attention.attention.extract_bpe_attention       --tf_name "${TF}"       --model_type "${MODEL_TYPE}"       --output_model_name "${OUTPUT_MODEL_NAME}"       --gfm_root "${MODEL_ROOT}"       --input_path "${pred_csv}"       --output_path "${attention_file}"       --batch_size 128       --write_chunk_size 512
  fi

  if [[ ! -s "${files_dir}/ohe1.npz" || ! -s "${files_dir}/hypscores1.npz" ]]; then
    python -m exp2_attention.discovery.build_tfmodisco_inputs_bpe       --tf_name "${TF}"       --model_type "${MODEL_TYPE}"       --output_model_name "${OUTPUT_MODEL_NAME}"       --csv_path "${pred_csv}"       --attn_dir "${attention_dir}"       --tokenizer_path "${TOKENIZER_PATH}"       --out_dir "${files_dir}"       --center none       --fake_negative       --fake_neg_base A       --fake_neg_scale 0.01
  fi

  modisco_h5="${files_dir}/modisco_results.h5"
  cwm_meme="${files_dir}/modisco_results.CWM-PFM.meme"
  if [[ ! -s "${modisco_h5}" ]]; then
    "${MODISCO_BIN}" motifs -s "${files_dir}/ohe1.npz" -a "${files_dir}/hypscores1.npz"       -n "${MAX_SEQLETS}" -o "${modisco_h5}" -w "${WINDOW}" -v || true
  fi
  if [[ -s "${modisco_h5}" ]]; then
    "${MODISCO_BIN}" meme -i "${modisco_h5}" -t CWM-PFM -o "${cwm_meme}" -q || true
    if [[ -s "${cwm_meme}" && -s "${MEME_DB}" ]]; then
      "${MODISCO_BIN}" report -i "${modisco_h5}" -o "${report_dir}" -s "${report_dir}"         -m "${MEME_DB}" -n "${TOP_MATCHES}" -l || true
    fi
    python -m exp2_attention.discovery.summarize_modisco_h5       -i "${modisco_h5}"       -o "${report_dir}/patterns_summary.csv"       -m "${MEME_DB}"       -n "${TOP_MATCHES}"       --query_matrix CWM_PFM       --model "${OUTPUT_MODEL_NAME}"       --tf "${TF}" || true
  fi
done
