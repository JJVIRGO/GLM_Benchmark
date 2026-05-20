#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-Data/processed_data}
MODEL_ROOT=${MODEL_ROOT:-GLM_weights}
TF_LIST=${TF_LIST:-"CTCF FOXA1 GATA1 GATA4 JUN MEF2A MYC NRF1 SPI1 USF2 YY1"}
BPE_MODELS=${BPE_MODELS:-"DNABERT-2 GENA_LM_BERT GROVER"}
RUN_NT=${RUN_NT:-1}

for TF in ${TF_LIST}; do
  fasta_file="${DATA_DIR}/${TF}/${TF}_positive_sequences.fasta"
  fimo_file="${DATA_DIR}/${TF}/fimo_results/fimo_filtered.tsv"
  if [[ ! -s "${fasta_file}" || ! -s "${fimo_file}" ]]; then
    echo "[WARN] Missing FASTA or filtered FIMO for ${TF}; skip"
    continue
  fi
  for MODEL in ${BPE_MODELS}; do
    mkdir -p "${DATA_DIR}/${TF}/${MODEL}"
    python -m exp2_attention.mapping.map_bpe_threshold       --fasta_file "${fasta_file}"       --fimo_file "${fimo_file}"       --result_df_path "${DATA_DIR}/${TF}/${MODEL}/motif_mapping.csv"       --model_type "${MODEL}"       --model_root "${MODEL_ROOT}"       --threshold_mode
  done
  if [[ "${RUN_NT}" = "1" ]]; then
    mkdir -p "${DATA_DIR}/${TF}/NT"
    python -m exp2_attention.mapping.map_nt_threshold       --tf_name "${TF}"       --processed_data_dir "${DATA_DIR}"       --threshold_mode
  fi
done
