#!/usr/bin/env bash
set -euo pipefail

PEAKS_DIR=${PEAKS_DIR:-Data/original_peaks}
REFERENCE_DIR=${REFERENCE_DIR:-Data/reference}
OUTPUT_DIR=${OUTPUT_DIR:-Data/processed_data}
TF_LIST=${TF_LIST:-"CTCF FOXA1 GATA1 GATA4 JUN LDB1 MEF2A MYC NRF1 SPI1 USF2 YY1"}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-1500000}
NUM_PROCESSES=${NUM_PROCESSES:-8}
SEED=${SEED:-42}

python -m exp2_attention.preprocess.build_tfbs_dataset   --peaks-dir "${PEAKS_DIR}"   --reference-dir "${REFERENCE_DIR}"   --output-dir "${OUTPUT_DIR}"   --tf-list ${TF_LIST}   --total-samples "${TOTAL_SAMPLES}"   --num-processes "${NUM_PROCESSES}"   --seed "${SEED}"

python -m exp2_attention.preprocess.split_dataset   --base-dir "${OUTPUT_DIR}"   --tf-list ${TF_LIST}
