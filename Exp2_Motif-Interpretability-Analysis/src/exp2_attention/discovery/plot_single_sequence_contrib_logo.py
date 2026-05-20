#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

try:
    import logomaker
except ImportError as exc:  # pragma: no cover - dependency check at runtime
    raise SystemExit("This script requires logomaker: pip install logomaker") from exc


BASES = ["A", "C", "G", "T"]
MODEL_ALIASES: Dict[str, str] = {
    "DNABERT2": "DNABERT2_5.6",
    "DNABERT-2": "DNABERT-2",
    "DNABERT2_5.6": "DNABERT2_5.6",
    "GENA": "GENA_LM_BERT",
    "GENA_LM_BERT": "GENA_LM_BERT",
    "GROVER": "GROVER",
    "NT": "NT",
}
MODEL_LABELS: Dict[str, str] = {
    "DNABERT2_5.6": "DNABERT2",
    "DNABERT-2": "DNABERT-2",
    "GENA_LM_BERT": "GENA",
    "GROVER": "GROVER",
    "NT": "NT",
}
ROW_COLORS = ["#b2182b", "#2166ac", "#1b7837", "#b8860b", "#542788", "#4d4d4d"]


@dataclass
class ModelData:
    model: str
    label: str
    ohe: np.ndarray
    contrib: np.ndarray


def resolve_model(name: str) -> str:
    key = name.strip()
    if key not in MODEL_ALIASES:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model '{name}'. Known names: {known}")
    return MODEL_ALIASES[key]


def load_npz_array(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path) as handle:
        if "arr_0" in handle.files:
            return np.asarray(handle["arr_0"])
        if len(handle.files) == 1:
            return np.asarray(handle[handle.files[0]])
        raise ValueError(f"Cannot choose array from {path}; keys={handle.files}")


def load_model_data(base_dir: str, model: str, tf_name: str) -> ModelData:
    files_dir = os.path.join(base_dir, "predict_true", model, "tfmodisco", tf_name, "files")
    ohe = load_npz_array(os.path.join(files_dir, "ohe1.npz"))
    contrib = load_npz_array(os.path.join(files_dir, "hypscores1.npz"))
    if ohe.shape != contrib.shape:
        raise ValueError(f"{model}/{tf_name}: ohe and contrib shapes differ: {ohe.shape} vs {contrib.shape}")
    if ohe.ndim != 3 or ohe.shape[1] != 4:
        raise ValueError(f"{model}/{tf_name}: expected arrays shaped [N,4,L], got {ohe.shape}")
    return ModelData(model=model, label=MODEL_LABELS.get(model, model), ohe=ohe, contrib=contrib)


def choose_sample_index(model_data: Sequence[ModelData], rank: int) -> int:
    min_n = min(data.contrib.shape[0] for data in model_data)
    score_parts = []
    for data in model_data:
        scores = np.max(np.abs(data.contrib[:min_n]), axis=(1, 2))
        max_score = float(np.max(scores))
        if max_score > 0:
            scores = scores / max_score
        score_parts.append(scores)
    scores = np.mean(np.vstack(score_parts), axis=0)
    rank = max(0, min(rank, scores.shape[0] - 1))
    return int(np.argsort(scores)[::-1][rank])


def sequence_from_ohe(ohe_4xl: np.ndarray) -> str:
    idx = np.argmax(ohe_4xl, axis=0)
    valid = ohe_4xl.sum(axis=0) > 0
    return "".join(BASES[i] if ok else "N" for i, ok in zip(idx, valid))


def window_from_center(length: int, center: int, width: Optional[int]) -> Tuple[int, int]:
    if width is None or width <= 0 or width >= length:
        return 0, length
    start = max(0, int(center) - width // 2)
    end = min(length, start + width)
    start = max(0, end - width)
    return start, end


def strongest_position_across_models(model_data: Sequence[ModelData], sample_index: int, length: int) -> int:
    per_pos_total = np.zeros(length, dtype=np.float64)
    for data in model_data:
        per_pos = np.max(np.abs(data.contrib[sample_index, :, :length]), axis=0).astype(np.float64)
        scale = float(np.max(per_pos))
        if scale > 0:
            per_pos_total += per_pos / scale
    return int(np.argmax(per_pos_total))


def contribution_matrix(
    ohe_4xl: np.ndarray,
    contrib_4xl: np.ndarray,
    start: int,
    end: int,
    normalize: str,
) -> pd.DataFrame:
    mat = (ohe_4xl[:, start:end] * contrib_4xl[:, start:end]).T.astype(np.float64)
    # 当前流程理论上应以非负贡献为主；这里裁剪负值以去除下半轴显示。
    mat = np.clip(mat, 0.0, None)
    if normalize == "maxabs":
        scale = float(np.max(np.abs(mat)))
        if scale > 0:
            mat = mat / scale
    return pd.DataFrame(mat, columns=BASES)


def upper_ylim_for_logo(mat: pd.DataFrame, normalize: str) -> float:
    values = mat.to_numpy(dtype=np.float64)
    pos_stack = np.clip(values, 0.0, None).sum(axis=1)
    y_max = float(np.max(pos_stack)) if pos_stack.size else 0.0
    if normalize == "maxabs":
        return max(1.25, 1.22 * max(y_max, 1e-6))
    return max(1.0, 1.22 * max(y_max, 1e-6))


def plot_single_sequence(
    model_data: Sequence[ModelData],
    tf_name: str,
    sample_index: int,
    start: int,
    end: int,
    out_prefix: str,
    normalize: str,
    dpi: int,
    title: Optional[str],
) -> None:
    n_rows = len(model_data)
    fig_h = max(1.75 * n_rows + 1.0, 3.0)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, fig_h),
        sharex=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.18},
    )
    axes_flat = axes[:, 0]

    for row_idx, (ax, data) in enumerate(zip(axes_flat, model_data)):
        mat = contribution_matrix(data.ohe[sample_index], data.contrib[sample_index], start, end, normalize)
        if float(np.max(np.abs(mat.values))) > 0:
            logo = logomaker.Logo(mat, ax=ax, color_scheme="classic")
            logo.style_spines(visible=False)
        else:
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_xlim(-0.5, len(mat) - 0.5)
        ax.set_ylim(0, upper_ylim_for_logo(mat, normalize))
        ax.tick_params(axis="both", labelsize=8, length=2, width=0.6)
        label_x = -0.03 if data.label == "GROVER" else -0.022
        ax.text(
            label_x,
            0.5,
            data.label,
            color=ROW_COLORS[row_idx % len(ROW_COLORS)],
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

    seq = sequence_from_ohe(model_data[0].ohe[sample_index])
    seq_slice = seq[start:end]
    tick_step = 20 if len(seq_slice) <= 220 else 100
    tick_positions = list(range(0, len(seq_slice), tick_step))
    axes_flat[-1].set_xticks(tick_positions)
    axes_flat[-1].set_xticklabels([str(start + pos) for pos in tick_positions])
    axes_flat[-1].set_xlabel("Sequence position", fontsize=10)
    fig.text(0.02, 0.5, "Profile contribution scores", rotation=90, va="center", fontsize=12)

    norm_tag = "normalized" if normalize == "maxabs" else "raw"
    fig_title = title or f"{tf_name} sample {sample_index} contribution scores ({norm_tag})"
    fig.suptitle(fig_title, y=0.98, fontsize=13)
    fig.subplots_adjust(left=0.18, right=0.995, top=0.9, bottom=0.16)

    png_path = f"{out_prefix}.png"
    pdf_path = f"{out_prefix}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    fasta_path = f"{out_prefix}.window.fa"
    with open(fasta_path, "w", encoding="utf-8") as handle:
        handle.write(f">{tf_name}|sample={sample_index}|window={start}-{end}\n")
        handle.write(seq_slice + "\n")

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {fasta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Image-1 style contribution logos for one sequence across one or more models."
    )
    parser.add_argument("--base-dir", default=os.getcwd(), help="Repository/base directory containing predict_true/")
    parser.add_argument("--tf-name", default="CTCF", help="TF name, e.g. CTCF")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["NT", "GROVER", "DNABERT2", "GENA"],
        help="Models to plot as rows. Aliases: NT GROVER DNABERT2 GENA",
    )
    parser.add_argument("--sample-index", type=int, default=None, help="0-based sample index to plot")
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="If --sample-index is omitted, choose this rank after averaging normalized max abs contribution across models.",
    )
    parser.add_argument("--window", type=int, default=180, help="Window width around the strongest position; <=0 plots full length")
    parser.add_argument("--start", type=int, default=None, help="Optional inclusive window start")
    parser.add_argument("--end", type=int, default=None, help="Optional exclusive window end")
    parser.add_argument(
        "--normalize",
        choices=["maxabs", "none"],
        default="maxabs",
        help="Normalize each row by its max absolute contribution in the plotted window.",
    )
    parser.add_argument("--out-dir", default="scripts/single_sequence_logos", help="Output directory")
    parser.add_argument("--out-prefix", default=None, help="Output file prefix without extension")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [resolve_model(name) for name in args.models]
    data = [load_model_data(args.base_dir, model, args.tf_name) for model in models]

    min_n = min(item.ohe.shape[0] for item in data)
    min_l = min(item.ohe.shape[2] for item in data)
    if args.sample_index is None:
        sample_index = choose_sample_index(data, args.rank)
    else:
        sample_index = int(args.sample_index)
    if not 0 <= sample_index < min_n:
        raise ValueError(f"sample index {sample_index} out of range for selected models; max shared index={min_n - 1}")

    center = strongest_position_across_models(data, sample_index, min_l)
    if args.start is not None or args.end is not None:
        start = 0 if args.start is None else max(0, int(args.start))
        end = min_l if args.end is None else min(min_l, int(args.end))
        if start >= end:
            raise ValueError(f"Invalid window: start={start}, end={end}")
    else:
        start, end = window_from_center(min_l, center, args.window)

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = "-".join(MODEL_LABELS.get(model, model) for model in models)
    out_prefix = args.out_prefix or f"{args.tf_name}_sample{sample_index}_{model_tag}_{start}_{end}"
    out_prefix = os.path.join(args.out_dir, out_prefix)

    plot_single_sequence(
        model_data=data,
        tf_name=args.tf_name,
        sample_index=sample_index,
        start=start,
        end=end,
        out_prefix=out_prefix,
        normalize=args.normalize,
        dpi=args.dpi,
        title=args.title,
    )


if __name__ == "__main__":
    main()
