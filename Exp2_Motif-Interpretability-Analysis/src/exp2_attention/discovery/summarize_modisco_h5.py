#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


@dataclass
class SummarizeConfig:
    input_h5: str
    output_csv: str
    motif_db: Optional[str] = None
    top_n: int = 3
    trim_threshold: float = 0.3
    query_matrix: str = "PFM"  # PFM | CWM_PFM | hCWM_PFM
    model: Optional[str] = None
    tf: Optional[str] = None


def _iter_patterns(modisco_h5: str) -> List[Tuple[str, np.ndarray, np.ndarray, int]]:
    results: List[Tuple[str, np.ndarray, np.ndarray, int]] = []
    with h5py.File(modisco_h5, "r") as f:
        for group in ["pos_patterns", "neg_patterns"]:
            if group not in f:
                continue
            metacluster = f[group]

            def sort_key(kv):
                return int(kv[0].split("_")[-1])

            for pattern_name, pattern in sorted(metacluster.items(), key=sort_key):  # type: ignore
                tag = f"{group}.{pattern_name}"
                ppm = np.array(pattern["sequence"], copy=True)
                cwm = np.array(pattern["contrib_scores"], copy=True)
                n_seqlets = int(pattern["seqlets"]["n_seqlets"][0])
                results.append((tag, ppm, cwm, n_seqlets))
    return results


def _trim_by_cwm(cwm: np.ndarray, threshold_ratio: float) -> Tuple[int, int]:
    score = np.sum(np.abs(cwm), axis=1)
    if score.size == 0:
        return 0, 0
    cutoff = float(np.max(score)) * float(threshold_ratio)
    pass_idx = np.where(score >= cutoff)[0]
    if pass_idx.size == 0:
        return 0, cwm.shape[0]
    return int(np.min(pass_idx)), int(np.max(pass_idx) + 1)


def _compute_tomtom_lite(queries_ppm_T: List[np.ndarray], motif_db_path: str, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
    from memelite import tomtom  # type: ignore
    from memelite.io import read_meme  # type: ignore

    target_db = read_meme(motif_db_path)
    target_names = list(target_db.keys())
    target_pwms = list(target_db.values())

    pvals, scores, offsets, overlaps, strands, idxs = tomtom(queries_ppm_T, target_pwms, n_nearest=top_n)
    return pvals, idxs.astype(int)


def summarize(config: SummarizeConfig) -> pd.DataFrame:
    rows = []
    patterns = _iter_patterns(config.input_h5)

    query_ppms_T: List[np.ndarray] = []
    for tag, ppm, cwm, n_seqlets in patterns:
        start, end = _trim_by_cwm(cwm, config.trim_threshold)
        trimmed_len = max(0, end - start)

        avg_importance = float(np.mean(np.sum(np.abs(cwm), axis=1))) if cwm.size else float("nan")
        gc_content = float(np.mean(ppm[:, 1] + ppm[:, 2])) if ppm.size else float("nan")

        row = {
            "pattern": tag,
            "num_seqlets": n_seqlets,
            "length": int(ppm.shape[0]),
            "trim_start": int(start),
            "trim_end": int(end),
            "trim_len": int(trimmed_len),
            "gc_content": gc_content,
            "avg_importance": avg_importance,
            "matrix": config.query_matrix,
        }
        if config.model is not None:
            row["model"] = config.model
        if config.tf is not None:
            row["tf"] = config.tf
        rows.append(row)

        if config.motif_db:
            if config.query_matrix.upper() == "PFM":
                trimmed_ppm = ppm[start:end] if trimmed_len > 0 else ppm
                query_ppms_T.append(trimmed_ppm.T)  # shape (4, L)
            elif config.query_matrix.upper() == "CWM_PFM":
                # Softmax over nucleotides for each position to get probabilities
                cwm_slice = cwm[start:end] if trimmed_len > 0 else cwm
                # Numerical stability
                exp = np.exp(cwm_slice - np.max(cwm_slice, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                query_ppms_T.append(probs.T)
            elif config.query_matrix.upper() == "HCWM_PFM":
                # Not directly available in this summarizer; fall back to CWM_PFM semantics
                cwm_slice = cwm[start:end] if trimmed_len > 0 else cwm
                exp = np.exp(cwm_slice - np.max(cwm_slice, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                query_ppms_T.append(probs.T)
            else:
                raise ValueError(f"Unsupported query_matrix: {config.query_matrix}")

    df = pd.DataFrame(rows)

    if config.motif_db and len(query_ppms_T) > 0:
        try:
            pvals, idxs = _compute_tomtom_lite(query_ppms_T, config.motif_db, config.top_n)
            from memelite.io import read_meme  # type: ignore
            target_db = read_meme(config.motif_db)
            target_names = list(target_db.keys())

            # For each query (row), attach top-k matches
            top_cols: List[str] = []
            for j in range(config.top_n):
                match_col = f"match{j}"
                pval_col = f"pval{j}"
                top_cols.extend([match_col, pval_col])
                df[match_col] = [target_names[int(idxs[i, j])].strip() for i in range(idxs.shape[0])]
                df[pval_col] = [float(pvals[i, j]) for i in range(pvals.shape[0])]

        except Exception as e:
            # Fallback: no matches if tomtom-lite not available
            sys.stderr.write(f"[WARN] tomtom-lite failed: {e}\n")

    return df


def parse_args(argv: Optional[List[str]] = None) -> SummarizeConfig:
    p = argparse.ArgumentParser(description="Summarize TF-MoDISco H5 into a CSV with optional tomtom-lite matches.")
    p.add_argument("-i", "--input", dest="input_h5", required=True, help="Path to modisco_results.h5")
    p.add_argument("-o", "--output", dest="output_csv", required=True, help="Path to output CSV")
    p.add_argument("-m", "--meme_db", dest="motif_db", default=None, help="Motif database in MEME format for tomtom-lite")
    p.add_argument("-n", "--top_n", dest="top_n", type=int, default=3, help="Top-N matches to include")
    p.add_argument("--trim_threshold", type=float, default=0.3, help="Trim threshold as ratio of max |CWM| per position")
    p.add_argument("--query_matrix", type=str, choices=["PFM", "CWM_PFM", "HCWM_PFM"], default="PFM", help="Which matrix to use as query vs DB")
    p.add_argument("--model", type=str, default=None, help="Optional model name for output column")
    p.add_argument("--tf", type=str, default=None, help="Optional TF name for output column")
    args = p.parse_args(argv)
    return SummarizeConfig(
        input_h5=args.input_h5,
        output_csv=args.output_csv,
        motif_db=args.motif_db,
        top_n=args.top_n,
        trim_threshold=args.trim_threshold,
        query_matrix=args.query_matrix,
        model=args.model,
        tf=args.tf,
    )


def main() -> None:
    config = parse_args()
    df = summarize(config)
    df.to_csv(config.output_csv, index=False)
    print(f"Wrote: {config.output_csv}")


if __name__ == "__main__":
    main()


