#!/usr/bin/env python
# coding: utf-8

"""
Build per-cell-line HDF5 datasets containing DNA sequences, DNase-seq signals,
and binary TF-binding labels for cross-cell-type prediction experiments.

Usage:
  # Step 1 – precompute base data (sequences + DNase signals) for one cell line
  python -m exp3_dnase.preprocess.create_peak_datasets \
      --mode precompute --cell_line K562 --train_or_test train

  # Step 2 – create per-TF labeled datasets from the precomputed base
  python -m exp3_dnase.preprocess.create_peak_datasets \
      --mode create --cell_line K562 --peak_type CTCF --train_or_test train

Environment variables (override defaults):
  DATA_ROOT   – root of the Data directory  (default: ./Data)
"""

import os
import sys
import argparse
import subprocess
import warnings

import h5py
import numpy as np
import pandas as pd
import pyBigWig
from Bio import SeqIO
from tqdm import tqdm

warnings.filterwarnings("ignore")

BIN_SIZE = 1000

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _data_root() -> str:
    return os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Data"))


def get_cell_line_info():
    """Return validated mapping of cell lines to their BigWig and peak files."""
    base_dir = _data_root()

    cell_info = {
        "GM12878": {"peaks": ["BRD4", "CTCF", "EZH2", "GABPA", "POLR2A", "USF2"], "bigwig": "ENCFF960FMM.bigWig"},
        "HepG2":   {"peaks": ["BRD4", "CTCF", "EZH2", "GABPA", "GCN5", "POLR2A", "USF2"], "bigwig": "ENCFF113VII.bigWig"},
        "K562":    {"peaks": ["BRD4", "CTCF", "EZH2", "GABPA", "GCN5", "POLR2A", "USF2"], "bigwig": "ENCFF972GVB.bigWig"},
        "Lung":    {"peaks": ["BRD4", "CTCF", "EZH2", "GABPA", "POLR2A", "USF2"], "bigwig": "ENCFF868ZIM.bigWig"},
    }

    validated = {}
    for cell, info in cell_info.items():
        original_dir  = os.path.join(base_dir, "original_data", cell)
        processed_dir = os.path.join(base_dir, "processed_data", cell)

        bigwig_path = os.path.join(original_dir, info["bigwig"])
        if not os.path.exists(bigwig_path):
            print(f"Warning: BigWig not found for {cell}: {bigwig_path}")
            continue

        valid_peaks = []
        for peak in info["peaks"]:
            peak_file = os.path.join(processed_dir, f"{peak}_peaks.bed")
            if os.path.exists(peak_file):
                valid_peaks.append(peak)
            else:
                print(f"Warning: peak file missing for {cell}/{peak}: {peak_file}")

        if valid_peaks:
            validated[cell] = {"peaks": valid_peaks, "bigwig": bigwig_path}

    return validated


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_background_bins(cell_line: str, train_or_test: str) -> pd.DataFrame:
    base_dir = _data_root()
    bins_file = os.path.join(base_dir, "processed_data", "DNase_bin", cell_line, f"{train_or_test}_dnase_bins.bed")

    if not os.path.exists(bins_file):
        raise FileNotFoundError(
            f"DNase bins not found: {bins_file}. Run scripts/00_prepare_dnase_bins.sh first."
        )

    bins_df = pd.read_csv(bins_file, sep="\t", header=None, names=["chrom", "start", "end"])
    standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    return bins_df[bins_df["chrom"].isin(standard_chroms)]


def load_peak_regions(cell_line: str, peak_type: str) -> pd.DataFrame:
    base_dir = _data_root()
    peak_file = os.path.join(base_dir, "processed_data", cell_line, f"{peak_type}_peaks.bed")

    if not os.path.exists(peak_file):
        raise FileNotFoundError(f"Peak file not found: {peak_file}")

    peaks_df = pd.read_csv(peak_file, sep="\t", header=None, names=["chrom", "start", "end", "peak_id", "score"])
    standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    return peaks_df[peaks_df["chrom"].isin(standard_chroms)]


def load_precomputed_data(cell_line: str, train_or_test: str):
    base_dir = _data_root()
    base_file = os.path.join(base_dir, "precomputed", cell_line, f"{cell_line}_{train_or_test}_base.h5")

    if not os.path.exists(base_file):
        raise FileNotFoundError(
            f"Precomputed data not found: {base_file}. Run --mode precompute first."
        )

    print(f"Loading precomputed data from {base_file}...")
    with h5py.File(base_file, "r") as hf:
        chroms          = [c.decode("utf-8") for c in hf["chroms"][:]]
        starts          = hf["starts"][:]
        ends            = hf["ends"][:]
        sequences_bytes = hf["sequences"][:]
        dnase_signals   = hf["dnase_signals"][:]

    sequences = [s.tobytes().decode("ascii") for s in sequences_bytes]
    bins_df = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    print(f"Loaded {len(bins_df):,} precomputed samples.")
    return bins_df, sequences, dnase_signals


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def create_labels_for_bins(bins_df: pd.DataFrame, peaks_df: pd.DataFrame, overlap_threshold: float = 0.5):
    """Label each 1000 bp bin as positive if it overlaps ≥ overlap_threshold of any TF peak."""
    print(f"Creating labels for {len(bins_df):,} bins ({len(peaks_df):,} peaks, threshold={overlap_threshold}) ...")

    bins_bed = "/tmp/_exp3_bins.bed"
    peaks_bed = "/tmp/_exp3_peaks.bed"
    intersect_file = "/tmp/_exp3_intersect.bed"

    try:
        bins_df.to_csv(bins_bed, sep="\t", header=False, index=False)
        peaks_df.to_csv(peaks_bed, sep="\t", header=False, index=False)

        subprocess.run(
            f"bedtools intersect -a {bins_bed} -b {peaks_bed} -wo > {intersect_file}",
            shell=True, check=True,
        )

        positive_bins = set()
        if os.path.exists(intersect_file) and os.path.getsize(intersect_file) > 0:
            overlap_df = pd.read_csv(
                intersect_file, sep="\t", header=None,
                names=["bin_chrom", "bin_start", "bin_end",
                       "peak_chrom", "peak_start", "peak_end",
                       "peak_id", "peak_score", "overlap_len"],
            )
            peak_lengths = {
                (r.chrom, r.start, r.end): r.end - r.start
                for _, r in peaks_df.iterrows()
            }
            for _, row in overlap_df.iterrows():
                peak_key = (row["peak_chrom"], row["peak_start"], row["peak_end"])
                if peak_key in peak_lengths:
                    if row["overlap_len"] / peak_lengths[peak_key] >= overlap_threshold:
                        positive_bins.add((row["bin_chrom"], row["bin_start"], row["bin_end"]))

        labels = [
            1 if (r.chrom, r.start, r.end) in positive_bins else 0
            for _, r in tqdm(bins_df.iterrows(), total=len(bins_df), desc="Assigning labels")
        ]
        print(f"Positive bins: {sum(labels):,} / {len(labels):,}")
        return labels

    finally:
        for f in [bins_bed, peaks_bed, intersect_file]:
            if os.path.exists(f):
                os.remove(f)


# ---------------------------------------------------------------------------
# Sequence and signal extraction
# ---------------------------------------------------------------------------

def extract_sequences(bins_df: pd.DataFrame, genome_fasta: str, n_filter_threshold: float = 1.0):
    print(f"Extracting sequences for {len(bins_df):,} regions (N-filter threshold: {n_filter_threshold}) ...")

    temp_bed   = "/tmp/_exp3_extract.bed"
    temp_fasta = "/tmp/_exp3_sequences.fa"

    try:
        bins_df[["chrom", "start", "end"]].to_csv(temp_bed, sep="\t", header=False, index=False)
        subprocess.run(
            f"bedtools getfasta -fi {genome_fasta} -bed {temp_bed} -fo {temp_fasta} -name",
            shell=True, check=True, stderr=subprocess.PIPE,
        )

        sequences = []
        valid_indices = []
        filtered = 0

        with open(temp_fasta) as handle:
            for i, record in enumerate(tqdm(SeqIO.parse(handle, "fasta"), desc="Filtering N sequences")):
                seq = str(record.seq).upper()
                if "N" in seq:
                    filtered += 1
                    continue
                sequences.append(seq)
                valid_indices.append(i)

        print(f"Valid sequences: {len(sequences):,} (filtered {filtered:,} containing N)")
        return sequences, valid_indices

    finally:
        for f in [temp_bed, temp_fasta]:
            if os.path.exists(f):
                os.remove(f)


def extract_dnase_signals(bins_df: pd.DataFrame, bigwig_file: str, filtered_indices=None):
    print(f"Extracting DNase signals from {bigwig_file} ...")

    if filtered_indices is not None:
        bins_to_use = bins_df.iloc[filtered_indices].reset_index(drop=True)
    else:
        bins_to_use = bins_df

    try:
        bw = pyBigWig.open(bigwig_file)
        signals = []
        for _, row in tqdm(bins_to_use.iterrows(), total=len(bins_to_use), desc="DNase signals"):
            chrom, start, end = str(row["chrom"]), int(row["start"]), int(row["end"])
            try:
                vals = np.nan_to_num(bw.values(chrom, start, end), nan=0.0).astype(np.float32)
                expected = end - start
                if len(vals) != expected:
                    padded = np.zeros(expected, dtype=np.float32)
                    padded[: min(len(vals), expected)] = vals[: expected]
                    vals = padded
                signals.append(vals)
            except Exception as e:
                print(f"Signal error {chrom}:{start}-{end}: {e}")
                signals.append(np.zeros(end - start, dtype=np.float32))
        bw.close()
        return signals

    except Exception as e:
        print(f"Error opening BigWig {bigwig_file}: {e}")
        n = len(filtered_indices) if filtered_indices is not None else len(bins_df)
        return [np.zeros(BIN_SIZE, dtype=np.float32)] * n


# ---------------------------------------------------------------------------
# HDF5 saving
# ---------------------------------------------------------------------------

def _save_h5(path: str, bins_df: pd.DataFrame, sequences, dnase_signals, meta_attrs: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as hf:
        meta = hf.create_group("metadata")
        for k, v in meta_attrs.items():
            meta.attrs[k] = v

        hf.create_dataset("chroms",        data=bins_df["chrom"].values.astype("S"),        compression="gzip")
        hf.create_dataset("starts",        data=bins_df["start"].values.astype(np.int32),   compression="gzip")
        hf.create_dataset("ends",          data=bins_df["end"].values.astype(np.int32),     compression="gzip")
        hf.create_dataset("sequences",     data=np.array([s.encode("ascii") for s in sequences], dtype=f"S{BIN_SIZE}"), compression="gzip")
        hf.create_dataset("dnase_signals", data=np.array(dnase_signals, dtype=np.float32), compression="gzip")

        if "label" in bins_df.columns:
            hf.create_dataset("labels", data=bins_df["label"].values.astype(np.int8), compression="gzip")

    print(f"Saved {len(bins_df):,} samples to {path}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def precompute_and_save_base_data(cell_line: str, bigwig_file: str, train_or_test: str, n_filter_threshold: float = 1.0):
    base_dir = _data_root()
    print(f"\n{'='*50}\nPrecomputing {cell_line} ({train_or_test})\n{'='*50}")

    bins_df = load_background_bins(cell_line, train_or_test)
    print(f"Background bins: {len(bins_df):,}")

    genome_fasta = os.path.join(base_dir, "processed_data", "genome", f"{train_or_test}_chr.fa")
    if not os.path.exists(genome_fasta):
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta}. Run prepare_genome_bins.sh first.")

    sequences, valid_indices = extract_sequences(bins_df, genome_fasta, n_filter_threshold)
    if not valid_indices:
        print("No valid sequences. Aborting.")
        return False

    filtered_bins  = bins_df.iloc[valid_indices].reset_index(drop=True)
    dnase_signals  = extract_dnase_signals(filtered_bins, bigwig_file)
    output_path    = os.path.join(base_dir, "precomputed", cell_line, f"{cell_line}_{train_or_test}_base.h5")

    _save_h5(output_path, filtered_bins, sequences, dnase_signals, meta_attrs={
        "cell_line": cell_line, "bin_size": BIN_SIZE,
        "total_samples": len(filtered_bins), "train_or_test": train_or_test,
    })
    return True


def create_dataset_from_precomputed(
    cell_line: str, peak_type: str, train_or_test: str,
    val_ratio: float = 0.1, overlap_threshold: float = 0.5, random_state: int = 42,
):
    base_dir = _data_root()
    print(f"\n{'='*50}\nCreating dataset for {cell_line}/{peak_type} ({train_or_test})\n{'='*50}")

    base_bins, base_seqs, base_dnase = load_precomputed_data(cell_line, train_or_test)
    peaks_df = load_peak_regions(cell_line, peak_type)
    labels   = create_labels_for_bins(base_bins, peaks_df, overlap_threshold)

    data = base_bins.copy()
    data["label"] = labels

    if train_or_test == "train":
        np.random.seed(random_state)
        pos_idx = data.index[data["label"] == 1].tolist()
        neg_idx = data.index[data["label"] == 0].tolist()

        val_pos = np.random.choice(pos_idx, max(1, int(len(pos_idx) * val_ratio)), replace=False).tolist()
        val_neg = np.random.choice(neg_idx, max(1, int(len(neg_idx) * val_ratio)), replace=False).tolist()
        tr_idx  = sorted(set(pos_idx + neg_idx) - set(val_pos + val_neg))
        va_idx  = val_pos + val_neg
        np.random.shuffle(tr_idx)
        np.random.shuffle(va_idx)

        for split, idx in [("train", tr_idx), ("val", va_idx)]:
            split_data  = data.iloc[idx].reset_index(drop=True)
            split_seqs  = [base_seqs[i] for i in idx]
            split_dnase = np.array(base_dnase, dtype=np.float32)[idx].tolist()
            out = os.path.join(base_dir, "dataset", cell_line, split, f"{cell_line}_{peak_type}_{split}.h5")
            _save_h5(out, split_data, split_seqs, split_dnase, meta_attrs={
                "cell_line": cell_line, "peak_type": peak_type, "bin_size": BIN_SIZE,
                "total_samples": len(split_data), "train_or_test": split,
                "num_positives": int((split_data["label"] == 1).sum()),
                "num_negatives": int((split_data["label"] == 0).sum()),
            })
    else:
        out = os.path.join(base_dir, "dataset", cell_line, "test", f"{cell_line}_{peak_type}_test.h5")
        _save_h5(out, data, base_seqs, base_dnase, meta_attrs={
            "cell_line": cell_line, "peak_type": peak_type, "bin_size": BIN_SIZE,
            "total_samples": len(data), "train_or_test": "test",
            "num_positives": int((data["label"] == 1).sum()),
            "num_negatives": int((data["label"] == 0).sum()),
        })
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build per-cell-type HDF5 datasets for TF binding prediction.")
    parser.add_argument("--mode",          choices=["precompute", "create"], required=True)
    parser.add_argument("--cell_line",     help="Cell line to process (default: all)")
    parser.add_argument("--peak_type",     help="TF peak type (used in create mode)")
    parser.add_argument("--train_or_test", choices=["train", "test"], default="train")
    parser.add_argument("--n_filter_threshold", type=float, default=1.0,
                        help="Sequences containing any N are filtered (default: 1.0)")
    parser.add_argument("--overlap_threshold",  type=float, default=0.5,
                        help="Min fraction of TF peak that must overlap bin for positive label (default: 0.5)")
    parser.add_argument("--val_ratio",          type=float, default=0.1,
                        help="Fraction of training data to use as validation (default: 0.1)")
    args = parser.parse_args()

    cell_info = get_cell_line_info()
    cells = {args.cell_line: cell_info[args.cell_line]} if args.cell_line else cell_info

    success = failed = 0
    for cell, info in tqdm(cells.items(), desc="Cell lines"):
        if args.mode == "precompute":
            ok = precompute_and_save_base_data(cell, info["bigwig"], args.train_or_test, args.n_filter_threshold)
        else:
            peaks = [args.peak_type] if args.peak_type else info["peaks"]
            ok_list = [
                create_dataset_from_precomputed(cell, peak, args.train_or_test, args.val_ratio, args.overlap_threshold)
                for peak in peaks
            ]
            ok = all(ok_list)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
