#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


DEFAULT_TFS = [
    "CTCF",
    "GATA4",
    "SPI1",
    "YY1",
    "GATA1",
    "NRF1",
    "MEF2A",
    "USF2",
    "FOXA1",
    "JUN",
    "MYC",
    "LDB1",
]

MA_ID_RE = re.compile(r"^MA\d+\.\d+$")


def clean_motif_label(label: object) -> str:
    """Remove duplicated JASPAR MA ids from labels.

    Example:
    MA0139.1 MA0139.1.CTCF -> MA0139.1.CTCF
    """
    text = str(label).strip()
    parts = text.split()
    if len(parts) >= 2 and MA_ID_RE.match(parts[0]) and parts[1].startswith(parts[0] + "."):
        return " ".join(parts[1:])
    return text


def clean_pattern_label(label: object) -> str:
    text = str(label).strip()
    parts = text.split("/")
    if len(parts) == 3:
        return f"{parts[0]}/{parts[2]}"
    return text


def read_similarity_matrix(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [clean_motif_label(col) for col in df.columns]
    df.index = [clean_pattern_label(idx) for idx in df.index]
    return df


def filter_columns_by_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    keep_cols = df.columns[df.max(axis=0) > threshold]
    return df.loc[:, keep_cols]


def load_matrices(
    heatmap_dir: str,
    tfs: Sequence[str],
    ldb1_threshold: float,
) -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        csv_path = os.path.join(heatmap_dir, f"{tf}_similarity_matrix.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing matrix for {tf}: {csv_path}")
        df = read_similarity_matrix(csv_path)
        if tf.upper() == "LDB1":
            df = filter_columns_by_threshold(df, ldb1_threshold)
        matrices[tf] = df
    return matrices


def figure_size(matrices: Dict[str, pd.DataFrame], ncols: int) -> tuple[float, float]:
    nrows = math.ceil(len(matrices) / ncols)
    max_cols = max((df.shape[1] for df in matrices.values()), default=1)
    max_rows = max((df.shape[0] for df in matrices.values()), default=1)

    width_per_panel = max(4.4, min(8.0, 2.8 + max_cols * 0.12))
    height_per_panel = max(3.2, min(5.0, 2.4 + max_rows * 0.12))
    return ncols * width_per_panel + 1.0, nrows * height_per_panel


def plot_all_heatmaps(
    matrices: Dict[str, pd.DataFrame],
    out_prefix: str,
    cmap: str,
    ncols: int,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    nrows = math.ceil(len(matrices) / ncols)
    fig_w, fig_h = figure_size(matrices, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    cbar_ax = fig.add_axes([0.93, 0.18, 0.012, 0.64])

    for idx, (tf, df) in enumerate(matrices.items()):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx][col_idx]
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=idx == 0,
            cbar_ax=cbar_ax if idx == 0 else None,
            linewidths=0.0,
        )
        if idx == 0:
            cbar_ax.set_ylabel("Similarity", fontsize=9)
        ax.set_title(tf, fontsize=13, pad=6)
        ax.set_xlabel("JASPAR motif" if row_idx == nrows - 1 else "", fontsize=9)
        ax.set_ylabel("Discovered motif" if col_idx == 0 else "", fontsize=9)
        ax.tick_params(axis="x", labelrotation=60, labelsize=6)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    for idx in range(len(matrices), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Discovered Motifs vs JASPAR Similarity", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.91, bottom=0.08, top=0.94, wspace=0.55, hspace=0.95)

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_prefix}.png", dpi=300)
    fig.savefig(f"{out_prefix}.pdf")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one large heatmap figure containing multiple TF similarity matrices."
    )
    parser.add_argument(
        "--heatmap_dir",
        default="scripts/heatmaps",
        help="Directory containing *_similarity_matrix.csv files.",
    )
    parser.add_argument(
        "--tfs",
        default=",".join(DEFAULT_TFS),
        help="Comma-separated TF names to include.",
    )
    parser.add_argument(
        "--out_prefix",
        default="scripts/heatmaps/all_tfs_discovered_vs_jaspar_heatmap",
        help="Output path prefix without file extension.",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    parser.add_argument("--ncols", type=int, default=4, help="Number of subplot columns.")
    parser.add_argument(
        "--ldb1_threshold",
        type=float,
        default=0.55,
        help="For LDB1, keep columns whose maximum value is greater than this threshold.",
    )
    parser.add_argument("--vmin", type=float, default=0.0, help="Heatmap lower color limit.")
    parser.add_argument("--vmax", type=float, default=1.0, help="Heatmap upper color limit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    tfs = [tf.strip() for tf in args.tfs.split(",") if tf.strip()]
    matrices = load_matrices(args.heatmap_dir, tfs, args.ldb1_threshold)
    plot_all_heatmaps(
        matrices=matrices,
        out_prefix=args.out_prefix,
        cmap=args.cmap,
        ncols=args.ncols,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    for tf, df in matrices.items():
        print(f"{tf}: rows={df.shape[0]} cols={df.shape[1]}")
    print(f"Saved: {args.out_prefix}.png")
    print(f"Saved: {args.out_prefix}.pdf")


if __name__ == "__main__":
    main()
