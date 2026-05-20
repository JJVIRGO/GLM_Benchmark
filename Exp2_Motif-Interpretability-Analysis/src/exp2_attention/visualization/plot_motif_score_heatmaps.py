import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exp2_attention.paths import OUTPUT_ROOT


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


DEFAULT_RESULTS_DIR = str(OUTPUT_ROOT / "motif_attention_stats")
DEFAULT_OUTPUT_DIR = str(OUTPUT_ROOT / "motif_attention_stats" / "visualize" / "heatmap")

MODEL_FILES: Dict[str, str] = {
    "DNABERT2_5.6": "DNABERT2_5.6",
    "GENA_LM_BERT": "GENA_LM_BERT",
    "NT": "NT",
    "GROVER": "GROVER",
}


def read_score_matrix(results_dir: str, tf_name: str, model_name: str) -> pd.DataFrame:
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unsupported model: {model_name}")

    file_model = MODEL_FILES[model_name]
    input_path = os.path.join(
        results_dir,
        tf_name,
        model_name,
        f"motif_scores_{file_model}_optimized.csv",
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    required = {"layer", "head", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")

    df = df[["layer", "head", "score"]].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    if df.empty:
        raise ValueError(f"{input_path} has no finite motif scores.")

    matrix = (
        df.pivot_table(index="layer", columns="head", values="score", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    matrix.index = [int(layer) + 1 for layer in matrix.index]
    matrix.columns = [int(head) + 1 for head in matrix.columns]
    return matrix


def plot_single_heatmap(
    model_name: str,
    tf_name: str,
    results_dir: str,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="white")
    matrix = read_score_matrix(results_dir, tf_name, model_name)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 8.5))
    sns.heatmap(
        matrix,
        cmap="viridis",
        annot=False,
        linewidths=0.0,
        cbar=True,
        cbar_kws={"label": "Motif score", "shrink": 0.9, "pad": 0.02},
        ax=ax,
    )
    ax.set_title(f"{model_name} on {tf_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"motif_score_heatmap_{model_name}_{tf_name}.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot motif score heatmaps by layer and head, one PDF per model and TF."
    )
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tf_name", default="CTCF")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["DNABERT2_5.6", "GENA_LM_BERT", "NT", "GROVER"],
        choices=list(MODEL_FILES),
    )
    args = parser.parse_args()

    output_paths: List[str] = []
    for model_name in args.models:
        output_path = plot_single_heatmap(
            model_name=model_name,
            tf_name=args.tf_name,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
        )
        output_paths.append(output_path)
        print(f"Saved PDF: {output_path}")


if __name__ == "__main__":
    main()
