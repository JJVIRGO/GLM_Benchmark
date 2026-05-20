import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


from exp2_attention.paths import PROCESSED_DATA_DIR as _PROCESSED_DATA_DIR
PROCESSED_DATA_DIR = str(_PROCESSED_DATA_DIR)


def display_model_name(model_type):
    return "DNABERT2" if model_type == "DNABERT2_5.6" else model_type


def main():
    """Parse arguments and generate visualizations."""

    parser = argparse.ArgumentParser(description="Generate attention visualizations from processed CSV files.")
    parser.add_argument("--model_type", type=str, required=True, help="Model directory/name, e.g. DNABERT-2 or DNABERT2_5.6")
    parser.add_argument("--all_tfs", action="store_true", help="Process all transcription factors")
    parser.add_argument("--tf_name", type=str, help="Single transcription factor name")
    parser.add_argument("--aggregation_mode", type=str, default="mean", choices=["mean", "max_head"],
                       help="Aggregation mode for attention data")
    parser.add_argument("--tf_list", type=str, nargs='*',
                       default=["CTCF", "FOXA1", "GATA1", "GATA4", "JUN", "MEF2A", "MYC", "NRF1", "SPI1", "USF2", "YY1"],
                       help="Transcription factor list for --all_tfs")
    parser.add_argument("--input_type", type=str, default="ratio", choices=["ratio", "threshold"], help="Input data type")
    parser.add_argument("--data_source", type=str, default="attention", choices=["attention", "motif"], help="Compatibility option; plots use attention_scores files")
    args = parser.parse_args()

    if args.all_tfs:
        print("Processing all TFs...")
        generate_all_tfs_barchart(args.tf_list, args.model_type, args.aggregation_mode, args.input_type)
    else:
        if not args.tf_name:
            print("Error: --tf_name is required unless --all_tfs is used.")
            return

        data_dir = os.path.join(PROCESSED_DATA_DIR, args.tf_name, args.model_type)
        output_dir = data_dir
        input_path = resolve_input_path(args.tf_name, args.model_type, args.input_type, args.data_source)

        if not os.path.exists(input_path):
            print(f"Error: input file does not exist - {input_path}")
            return

        print(f"Reading: {input_path}")
        results_df = pd.read_csv(input_path)

        print("Generating visualization...")
        generate_single_barchart(results_df, args.tf_name, args.model_type, args.aggregation_mode, output_dir)

        print(f"Finished {args.tf_name}.")


def resolve_input_path(tf_name, model_type, input_type, data_source):
    data_dir = os.path.join(PROCESSED_DATA_DIR, tf_name, model_type)
    return os.path.join(data_dir, f"attention_scores_{model_type}_{input_type}.csv")

def generate_single_barchart(results_df, tf_name, model_type, aggregation_mode, output_dir):
    """Generate one TF attention bar chart."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    if aggregation_mode == "mean":
        layer_scores = results_df.groupby('layer')['score'].mean()
        ylabel = 'Mean Attention Score'
    else:  # max_head
        layer_scores = results_df.groupby('layer')['score'].max()
        ylabel = 'Max Attention Score (per head)'

    sns.barplot(x=layer_scores.index, y=layer_scores.values, ax=ax, palette='viridis')
    ax.set_title(f'{tf_name} - Attention Scores ({display_model_name(model_type)} Model)', fontsize=16)
    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=0)

    for i, v in enumerate(layer_scores.values):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    barchart_path = os.path.join(output_dir, f"attention_barchart_{model_type}_{aggregation_mode}.png")
    plt.savefig(barchart_path)
    plt.close(fig)
    print(f"Saved bar chart: {barchart_path}")


def generate_all_tfs_barchart(tf_list, model_type, aggregation_mode, input_type):
    """Generate an all-TF attention summary PDF."""
    output_dir = os.path.join(PROCESSED_DATA_DIR, input_type)
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"attention_barchart_all_tfs_{model_type}_{aggregation_mode}_{input_type}.pdf")

    with PdfPages(pdf_path) as pdf:
        n_tfs = len(tf_list)
        n_cols = 3
        n_rows = int(np.ceil(n_tfs / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        plt.style.use('seaborn-v0_8-whitegrid')

        axes_flat = np.atleast_1d(axes).flatten()

        for i, tf_name in enumerate(tf_list):
            ax = axes_flat[i]

            input_path = resolve_input_path(tf_name, model_type, input_type, "attention")

            if not os.path.exists(input_path):
                print(f"Warning: missing input for {tf_name}, skipping - {input_path}")
                ax.text(0.5, 0.5, f'{tf_name}\n(No data)', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

            try:
                results_df = pd.read_csv(input_path)

                if aggregation_mode == "mean":
                    layer_scores = results_df.groupby('layer')['score'].mean()
                    ylabel = 'Mean Attention Score'
                else:  # max_head
                    layer_scores = results_df.groupby('layer')['score'].max()
                    ylabel = 'Max Attention Score'

                sns.barplot(x=layer_scores.index, y=layer_scores.values, ax=ax, palette='viridis')

                ax.set_title(f'{tf_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Layer', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.tick_params(axis='x', rotation=0, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)

                for j, v in enumerate(layer_scores.values):
                    ax.text(j, v, f"{v:.3f}", ha='center', va='bottom', fontsize=7)

            except Exception as e:
                print(f"Error while processing {tf_name}: {e}")
                ax.text(0.5, 0.5, f'{tf_name}\n(Error)', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        for i in range(n_tfs, n_rows * n_cols):
            axes_flat[i].set_visible(False)

        fig.suptitle(f'Attention Scores Across All Transcription Factors\n({display_model_name(model_type)} Model, {aggregation_mode})',
                    fontsize=20, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved all-TF attention PDF: {pdf_path}")

# 保留旧函数名以兼容性，但标记为废弃
def generate_barchart(results_df, tf_name, model_type, output_dir):
    """
    废弃: 请使用 generate_single_barchart
    """
    generate_single_barchart(results_df, tf_name, model_type, "mean", output_dir)

# 热图功能已暂时禁用，专注于直方图绘制
# def generate_heatmap(results_df, tf_name, model_type, output_dir):
#     """
#     生成并保存热图
#     """
#     try:
#         # 从DataFrame重建score_matrix
#         num_layers = int(results_df['layer'].max() + 1)
#         num_heads = int(results_df['head'].max() + 1)
#         score_matrix = np.zeros((num_layers, num_heads))
#         for _, row in results_df.iterrows():
#             score_matrix[int(row['layer']), int(row['head'])] = row['score']
#     except Exception as e:
#         print(f"创建热图矩阵时出错: {e}")
#         return
#
#     fig, ax = plt.subplots(figsize=(14, 10))
#     sns.heatmap(score_matrix, ax=ax, cmap="autumn", cbar=True)
#     ax.set_title(f'{tf_name} - attention scores ({model_type} Model)', fontsize=16)
#     ax.set_xlabel('Attention Head', fontsize=12)
#     ax.set_ylabel('Transformer Layer', fontsize=12)
#     ax.figure.tight_layout()
#
#     heatmap_path = os.path.join(output_dir, f"attention_heatmap_{model_type}.png")
#     plt.savefig(heatmap_path)
#     plt.close(fig)
#     print(f"热图已保存至: {heatmap_path}")

if __name__ == "__main__":
    main()
#bash run_visualization.sh --all_tfs --model_type GENA_LM_BERT --aggregation_mode max_head
# python visualize_attention.py --tf_name GATA4 --model_type GROVER --aggregation_mode mean
