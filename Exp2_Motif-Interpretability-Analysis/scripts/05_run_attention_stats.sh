#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-DNABERT-2}
OUTPUT_MODEL_NAME=${OUTPUT_MODEL_NAME:-${MODEL_TYPE}}
TF_LIST=${TF_LIST:-"CTCF FOXA1 GATA1 GATA4 JUN MEF2A MYC NRF1 SPI1 USF2 YY1"}
DATA_DIR=${DATA_DIR:-Data/processed_data}
MODEL_ROOT=${MODEL_ROOT:-GLM_weights}
STAT_ROOT=${STAT_ROOT:-outputs/motif_attention_stats}
BATCH_SIZE=${BATCH_SIZE:-128}

for TF in ${TF_LIST}; do
  model_dir="${DATA_DIR}/${TF}/${OUTPUT_MODEL_NAME}"
  mkdir -p "${model_dir}" "${STAT_ROOT}/${TF}/${OUTPUT_MODEL_NAME}"
  mapping_csv="${model_dir}/motif_mapping_${OUTPUT_MODEL_NAME}_threshold.csv"
  if [[ ! -s "${mapping_csv}" && -s "${DATA_DIR}/${TF}/${MODEL_TYPE}/motif_mapping_${MODEL_TYPE}_threshold.csv" ]]; then
    cp "${DATA_DIR}/${TF}/${MODEL_TYPE}/motif_mapping_${MODEL_TYPE}_threshold.csv" "${mapping_csv}"
  fi
  if [[ ! -s "${mapping_csv}" ]]; then
    echo "[WARN] Missing motif mapping for ${TF}/${OUTPUT_MODEL_NAME}; skip"
    continue
  fi
  attention_file="${model_dir}/attention_scores_${OUTPUT_MODEL_NAME}_original.parquet"
  attention_dir="${model_dir}/attention_scores_${OUTPUT_MODEL_NAME}_original"
  python -m exp2_attention.attention.extract_bpe_attention     --tf_name "${TF}"     --model_type "${MODEL_TYPE}"     --output_model_name "${OUTPUT_MODEL_NAME}"     --gfm_root "${MODEL_ROOT}"     --input_path "${mapping_csv}"     --output_path "${attention_file}"     --batch_size "${BATCH_SIZE}"     --write_chunk_size 512
  python -m exp2_attention.attention.build_attention_scores_from_parquet     --tf_name "${TF}"     --model_type "${OUTPUT_MODEL_NAME}"     --processed_data_dir "${DATA_DIR}"     --mapping_csv "${mapping_csv}"     --attention_dir "${attention_dir}"     --overwrite
  python -m exp2_attention.stats.run_motif_attention_stats     --tf_name "${TF}"     --model_type "${MODEL_TYPE}"     --output_model_name "${OUTPUT_MODEL_NAME}"     --attention_path "${attention_dir}"     --mapping_csv "${mapping_csv}"     --out_dir "${STAT_ROOT}/${TF}/${OUTPUT_MODEL_NAME}"     --num_random 1000     --seed 42     --aggregate mean     --alternative greater     --mode optimized     --progress log     --log_every 500
done
