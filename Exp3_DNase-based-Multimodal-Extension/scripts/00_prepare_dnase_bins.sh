#!/usr/bin/env bash
# ============================================================
# 00_prepare_dnase_bins.sh
#
# Generate DNase-peak-centric 1000 bp bins for each cell line.
# Bins are centered on DNase narrowPeak summits, stripped of
# ENCODE hg38 blacklist regions, then split by chromosome into
# train (chr1/11/13/19) and test (chr12) sets.
#
# Environment variables (override defaults):
#   DATA_ROOT   – path to the Data directory
#   REFERENCE_DIR – path to directory containing hg38-blacklist.v2.bed
#
# Usage:
#   bash scripts/00_prepare_dnase_bins.sh
#   bash scripts/00_prepare_dnase_bins.sh GM12878   # single cell line
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$SCRIPT_DIR")/Data}"
REFERENCE_DIR="${REFERENCE_DIR:-${DATA_ROOT}/original_data/reference}"
IN_DIR="${DATA_ROOT}/original_data"
OUT_DIR_BASE="${DATA_ROOT}/processed_data/DNase_bin"

CELL_LINES=("GM12878" "HepG2" "K562" "Lung")
if [ $# -ge 1 ]; then
    CELL_LINES=("$1")
fi

PREPARE_SCRIPT="$(dirname "$SCRIPT_DIR")/src/exp3_dnase/preprocess/prepare_dnase_bins_helper.sh"
# Fallback: use bundled awk-based implementation below if helper is absent
if [ ! -f "$PREPARE_SCRIPT" ]; then
    PREPARE_SCRIPT="$SCRIPT_DIR/_prepare_bins_inline.sh"
fi

WINDOW=1000
HALF=$((WINDOW / 2))
BLACKLIST="${REFERENCE_DIR}/hg38-blacklist.v2.bed"

if [ ! -f "$BLACKLIST" ]; then
    echo "Error: blacklist not found at $BLACKLIST (set REFERENCE_DIR)" >&2
    exit 1
fi

for CELL in "${CELL_LINES[@]}"; do
    DNASE_BED="${IN_DIR}/${CELL}/DNase_peaks.bed"
    if [ ! -f "$DNASE_BED" ]; then
        echo "Warning: $DNASE_BED not found – skipping $CELL" >&2
        continue
    fi

    OUTDIR="${OUT_DIR_BASE}/${CELL}"
    mkdir -p "$OUTDIR"

    TMP_ALL="${OUTDIR}/_tmp_all.bed"
    TMP_CLEAN="${OUTDIR}/_tmp_clean.bed"

    echo "[$(date '+%T')] $CELL – generating ${WINDOW}bp summit-centred windows ..."
    awk -v half="$HALF" 'BEGIN{OFS="\t"} {
        center=$2+$10; start=center-half; end=center+half;
        if(start>0) print $1, start, end
    }' "$DNASE_BED" > "$TMP_ALL"

    echo "[$(date '+%T')] $CELL – removing blacklist ..."
    bedtools intersect -a "$TMP_ALL" -b "$BLACKLIST" -v > "$TMP_CLEAN"

    echo "[$(date '+%T')] $CELL – splitting by chromosome ..."
    awk '$1 ~ /^(chr1|chr11|chr13|chr19)$/' "$TMP_CLEAN" > "${OUTDIR}/train_dnase_bins.bed"
    awk '$1 ~ /^chr12$/'                    "$TMP_CLEAN" > "${OUTDIR}/test_dnase_bins.bed"

    rm -f "$TMP_ALL" "$TMP_CLEAN"
    echo "[$(date '+%T')] $CELL done. train=$(wc -l < "${OUTDIR}/train_dnase_bins.bed")  test=$(wc -l < "${OUTDIR}/test_dnase_bins.bed")"
done

echo "All cell lines processed."
