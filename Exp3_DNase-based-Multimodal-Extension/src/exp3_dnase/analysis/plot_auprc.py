#!/usr/bin/env python
# coding: utf-8

"""
Generate a 2×3 subplot figure comparing sequence-only vs. sequence+DNase AUPRC
for each TF across all evaluated models.

Run:
  python -m exp3_dnase.analysis.plot_auprc \
      --results_root /path/to/DNase_implement \
      --output_dir   ./figures

The expected result directory layout mirrors the original training output structure:
  results_root/scripts/finetune/outputs/{tf}/BPE/GROVER_frozen/results/test_results.json
  results_root/scripts/finetune/without_DNase_5.1/outputs/{tf}/BPE/GROVER_sequence_only_frozen/results/test_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"]    = "sans-serif"
plt.rcParams["font.sans-serif"]= ["Arial"]
plt.rcParams["pdf.fonttype"]   = 42
plt.rcParams["ps.fonttype"]    = 42

TF_LIST = ["BRD4", "CTCF", "EZH2", "GABPA", "POLR2A", "USF2"]

MODEL_LABELS = {
    "Basic":           "Basic",
    "dnabert2_117m":   "DNABERT-2",
    "ntv2_500m_multi": "NTv2-500M",
    "gena_bigbird_t2t":"GENA-Bert",
    "grover":          "GROVER",
    "hyenadna_large_1m":"HyenaDNA",
}

MODEL_COLORS = {
    "Basic":           "#E75D5D",
    "dnabert2_117m":   "#3B6EA8",
    "ntv2_500m_multi": "#D95F02",
    "gena_bigbird_t2t":"#2A9D8F",
    "grover":          "#7A5195",
    "hyenadna_large_1m":"#6C757D",
}

MODEL_KEY_TO_PATTERN = {
    "Basic":           "Basic",
    "dnabert2_117m":   "DNABERT2",
    "grover":          "GROVER",
    "gena_bigbird_t2t":"GENA_LM_BERT",
    "ntv2_500m_multi": "NT",
    "hyenadna_large_1m":"Hyena",
}

SEQUENCE_ONLY_PATTERNS = {
    "Basic":      "scripts/finetune/without_DNase_5.1/outputs/{tf}/Basic/256d_sequence_only_unfrozen/results/test_results.json",
    "DNABERT2":   "scripts/finetune/without_DNase_5.1/outputs/{tf}/BPE/DNABERT2_sequence_only_frozen/results/test_results.json",
    "GROVER":     "scripts/finetune/without_DNase_5.1/outputs/{tf}/BPE/GROVER_sequence_only_frozen/results/test_results.json",
    "GENA_LM_BERT":"scripts/finetune/without_DNase_5.1/outputs/{tf}/BPE/GENA_LM_BERT_sequence_only_frozen/results/test_results.json",
    "NT":         "scripts/finetune/without_DNase_5.1/outputs/{tf}/NT/NT_sequence_only_frozen/results/test_results.json",
    "Hyena":      "scripts/finetune/without_DNase_5.1/outputs/{tf}/Hyena/hyena-small_sequence_only_frozen/results/test_results.json",
}

WITH_DNASE_PATTERNS = {
    "Basic":       "scripts/finetune/outputs/{tf}/Basic/256d_unfrozen/results/test_results.json",
    "DNABERT2":    "scripts/finetune/outputs/{tf}/BPE/DNABERT2_frozen/results/test_results.json",
    "GROVER":      "scripts/finetune/outputs/{tf}/BPE/GROVER_frozen/results/test_results.json",
    "GENA_LM_BERT":"scripts/finetune/outputs/{tf}/BPE/GENA_LM_BERT_frozen/results/test_results.json",
    "NT":          "scripts/finetune/outputs/{tf}/NT/NT_frozen/results/test_results.json",
    "Hyena":       "scripts/finetune/outputs/{tf}/Hyena/hyena-small_frozen/results/test_results.json",
}


def read_auprc(path: Path) -> float:
    if not path.exists():
        return float("nan")
    with path.open() as f:
        val = json.load(f).get("eval_auprc", float("nan"))
    return float(val)


def collect_matrix(root: Path, patterns: dict) -> np.ndarray:
    model_keys = list(MODEL_LABELS.keys())
    matrix = np.zeros((len(model_keys), len(TF_LIST)), dtype=float)
    for mi, mk in enumerate(model_keys):
        pk = MODEL_KEY_TO_PATTERN.get(mk, mk)
        pat = patterns.get(pk)
        for ti, tf in enumerate(TF_LIST):
            matrix[mi, ti] = read_auprc(root / pat.format(tf=tf)) if pat else float("nan")
    return matrix


def plot_figure(seq_only: np.ndarray, with_dnase: np.ndarray, ylabel: str,
                title: str, output_path: Path):
    model_keys = list(MODEL_LABELS.keys())
    labels = [MODEL_LABELS[k] for k in model_keys]
    colors = [MODEL_COLORS.get(k, "#777777") for k in model_keys]

    width = 0.12
    group_dist = 1.0
    group_centers = np.array([-group_dist / 2, group_dist / 2])
    offsets = (np.arange(len(model_keys)) - (len(model_keys) - 1) / 2) * width

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
    axes = axes.flatten()

    for ti, tf in enumerate(TF_LIST):
        ax = axes[ti]
        for mi in range(len(model_keys)):
            y = [seq_only[mi, ti], with_dnase[mi, ti]]
            bars = ax.bar(group_centers + offsets[mi], y, width=width,
                          label=labels[mi] if ti == 0 else None, color=colors[mi])
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(tf, fontsize=15, pad=10)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(["sequence-only", "sequence+DNase"], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(model_keys), frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=20, y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot sequence-only vs. sequence+DNase AUPRC comparison.")
    parser.add_argument("--results_root", required=True,
                        help="Root directory of the original DNase_implement repo (contains scripts/finetune/outputs/)")
    parser.add_argument("--output_dir", default="./figures",
                        help="Directory to save output PDF files")
    args = parser.parse_args()

    root = Path(args.results_root)
    out  = Path(args.output_dir)

    seq_only  = collect_matrix(root, SEQUENCE_ONLY_PATTERNS)
    with_dnase= collect_matrix(root, WITH_DNASE_PATTERNS)

    # Compute val matrices
    val_seq_only_pats  = {k: v.replace("test_results.json", "val_results.json") for k, v in SEQUENCE_ONLY_PATTERNS.items()}
    val_dnase_pats     = {k: v.replace("test_results.json", "val_results.json") for k, v in WITH_DNASE_PATTERNS.items()}
    val_seq_only   = collect_matrix(root, val_seq_only_pats)
    val_with_dnase = collect_matrix(root, val_dnase_pats)

    plot_figure(seq_only, with_dnase, "test AUPRC",
                "Test AUPRC: Sequence-only vs. Sequence+DNase",
                out / "auprc_transfer_test_6tf_subplots.pdf")

    plot_figure(val_seq_only, val_with_dnase, "val AUPRC",
                "Validation AUPRC: Sequence-only vs. Sequence+DNase",
                out / "auprc_transfer_val_6tf_subplots.pdf")


if __name__ == "__main__":
    main()
