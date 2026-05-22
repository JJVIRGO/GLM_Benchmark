#!/usr/bin/env python
# coding: utf-8

"""
Merge per-cell-line HDF5 datasets into cross-cell-line training/validation sets
and copy the GM12878 test set as the held-out cross-cell-type test split.

Usage:
  python -m exp3_dnase.preprocess.merge_cross_cell_datasets --mode all
  python -m exp3_dnase.preprocess.merge_cross_cell_datasets --mode merge_train --peak_type CTCF

Environment variables:
  DATA_ROOT  – root of the Data directory (default: ./Data)
  DATASET_DIR – imbalanced dataset root (default: DATA_ROOT/dataset)
"""

import argparse
import os
import warnings

import h5py
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _dataset_dir() -> str:
    explicit = os.environ.get("DATASET_DIR")
    if explicit:
        return explicit
    data_root = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Data"))
    return os.path.join(data_root, "dataset")


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_datasets_for_peak(source_cell_lines, peak_type: str, data_type: str, output_dir: str) -> bool:
    base_dir = _dataset_dir()
    print(f"\n{'='*60}\nMerging {data_type} | {peak_type} from {source_cell_lines}\n{'='*60}")

    source_files = []
    for cell in source_cell_lines:
        path = os.path.join(base_dir, cell, data_type, f"{cell}_{peak_type}_{data_type}.h5")
        if os.path.exists(path):
            source_files.append((cell, path))
        else:
            print(f"  Warning: {path} not found – skipping")

    if not source_files:
        print(f"  Error: no source files for {peak_type}/{data_type}")
        return False

    all_chroms = []
    all_starts = []
    all_ends   = []
    all_labels = []
    all_seqs   = []
    all_dnase  = []
    total_pos  = 0

    for cell, path in tqdm(source_files, desc="Reading"):
        with h5py.File(path, "r") as hf:
            chroms = [c.decode("utf-8") for c in hf["chroms"][:]]
            labels = hf["labels"][:]
            seqs   = [s.tobytes().decode("ascii") for s in hf["sequences"][:]]

            all_chroms.extend(chroms)
            all_starts.extend(hf["starts"][:].tolist())
            all_ends.extend(hf["ends"][:].tolist())
            all_labels.extend(labels.tolist())
            all_seqs.extend(seqs)
            all_dnase.extend(hf["dnase_signals"][:].tolist())
            total_pos += int(np.sum(labels))

            n = len(labels)
            print(f"  {cell}: {n:,} samples, {int(np.sum(labels)):,} positive ({np.sum(labels)/n*100:.1f}%)")

    total = len(all_labels)
    print(f"\nMerged total: {total:,} samples, {total_pos:,} positive ({total_pos/total*100:.1f}%)")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{peak_type}_{data_type}_merged.h5")

    with h5py.File(out_path, "w") as hf:
        meta = hf.create_group("metadata")
        meta.attrs["peak_type"]         = peak_type
        meta.attrs["data_type"]         = data_type
        meta.attrs["source_cell_lines"] = ",".join(source_cell_lines)
        meta.attrs["total_samples"]     = total
        meta.attrs["num_positives"]     = total_pos
        meta.attrs["num_negatives"]     = total - total_pos
        meta.attrs["bin_size"]          = 1000

        hf.create_dataset("chroms",        data=np.array(all_chroms, dtype="S"),              compression="gzip")
        hf.create_dataset("starts",        data=np.array(all_starts, dtype=np.int32),         compression="gzip")
        hf.create_dataset("ends",          data=np.array(all_ends,   dtype=np.int32),         compression="gzip")
        hf.create_dataset("labels",        data=np.array(all_labels, dtype=np.int8),          compression="gzip")
        hf.create_dataset("sequences",     data=np.array([s.encode("ascii") for s in all_seqs], dtype="S1000"), compression="gzip")
        hf.create_dataset("dnase_signals", data=np.array(all_dnase, dtype=np.float32),        compression="gzip")

    print(f"Saved merged dataset: {out_path}")
    return True


def copy_gm12878_test(peak_type: str, output_dir: str) -> bool:
    base_dir    = _dataset_dir()
    source_path = os.path.join(base_dir, "GM12878", "test", f"GM12878_{peak_type}_test.h5")
    out_path    = os.path.join(output_dir, f"{peak_type}_test_GM12878.h5")

    if not os.path.exists(source_path):
        print(f"  Error: GM12878 test file not found: {source_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(source_path, "r") as src, h5py.File(out_path, "w") as dst:
        for key in src.keys():
            if isinstance(src[key], h5py.Group):
                src.copy(key, dst)
            else:
                dst.create_dataset(key, data=src[key][:], compression="gzip")

    with h5py.File(out_path, "r") as hf:
        n = len(hf["labels"][:])
        n_pos = int(np.sum(hf["labels"][:]))
    print(f"Copied GM12878 test for {peak_type}: {n:,} samples, {n_pos:,} positive → {out_path}")
    return True


def get_available_peaks(cell_lines, base_dir: str):
    peak_sets = []
    for cell in cell_lines:
        train_dir = os.path.join(base_dir, cell, "train")
        if not os.path.isdir(train_dir):
            continue
        peaks = set()
        for fname in os.listdir(train_dir):
            if fname.endswith("_train.h5"):
                parts = fname.split("_")
                if len(parts) >= 2:
                    peaks.add(parts[1])
        peak_sets.append(peaks)
    return sorted(set.intersection(*peak_sets)) if peak_sets else []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge cross-cell-line TF-binding datasets.")
    parser.add_argument("--mode", choices=["merge_train", "merge_val", "copy_test", "all"], default="all")
    parser.add_argument("--peak_type", help="Specific TF to process (default: all available)")
    parser.add_argument("--source_cell_lines", default="K562,Lung,HepG2",
                        help="Comma-separated training cell lines (default: K562,Lung,HepG2)")
    parser.add_argument("--output_dir",
                        help="Output root dir (default: DATASET_DIR/imbalanced)")
    args = parser.parse_args()

    source_lines = [c.strip() for c in args.source_cell_lines.split(",") if c.strip()]
    base_dir     = _dataset_dir()
    output_root  = args.output_dir or os.path.join(base_dir, "imbalanced")

    available = get_available_peaks(source_lines, base_dir)
    if not available:
        print(f"No common peaks found across {source_lines}. Check DATA_ROOT / dataset structure.")
        return

    peaks = [args.peak_type] if args.peak_type else available
    print(f"Source cell lines : {source_lines}")
    print(f"Peaks to process  : {peaks}")
    print(f"Output root       : {output_root}")

    ok = failed = 0
    for peak in peaks:
        results = []
        if args.mode in ("merge_train", "all"):
            results.append(merge_datasets_for_peak(source_lines, peak, "train", os.path.join(output_root, "train")))
        if args.mode in ("merge_val", "all"):
            results.append(merge_datasets_for_peak(source_lines, peak, "val", os.path.join(output_root, "val")))
        if args.mode in ("copy_test", "all"):
            results.append(copy_gm12878_test(peak, os.path.join(output_root, "test")))

        if all(results):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} peaks succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
