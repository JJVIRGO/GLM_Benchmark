#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-Data/processed_data}
MEME_DIR=${MEME_DIR:-Data/meme}
TF_LIST=${TF_LIST:-"CTCF FOXA1 GATA1 GATA4 JUN MEF2A MYC NRF1 SPI1 USF2 YY1"}

for TF in ${TF_LIST}; do
  input_csv="${DATA_DIR}/${TF}/${TF}_positive_sequences.csv"
  fasta_file="${DATA_DIR}/${TF}/${TF}_positive_sequences.fasta"
  output_dir="${DATA_DIR}/${TF}/fimo_results"
  meme_files=("${MEME_DIR}/${TF}"*.meme)
  if [[ ! -s "${input_csv}" ]]; then
    echo "[WARN] Missing positive CSV for ${TF}: ${input_csv}"
    continue
  fi
  if [[ ! -e "${meme_files[0]}" ]]; then
    echo "[WARN] Missing MEME files for ${TF} in ${MEME_DIR}"
    continue
  fi
  python -m exp2_attention.fimo.extract_motif_position     --input_csv "${input_csv}"     --input_meme "${meme_files[@]}"     --output_dir "${output_dir}"     --fasta_file "${fasta_file}"
  python -m exp2_attention.fimo.filter_fimo_results     "${output_dir}/fimo.tsv"     --output "${output_dir}/fimo_filtered.tsv"
done
