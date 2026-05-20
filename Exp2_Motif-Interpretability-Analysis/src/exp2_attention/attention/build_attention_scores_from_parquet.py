import argparse
import ast
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd


from exp2_attention.paths import PROCESSED_DATA_DIR

DEFAULT_PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)


def parse_mapping(mapping_value):
    if isinstance(mapping_value, str):
        return np.asarray(ast.literal_eval(mapping_value), dtype=float)
    if isinstance(mapping_value, list):
        return np.asarray(mapping_value, dtype=float)
    return np.asarray(mapping_value, dtype=float)


def infer_block_size(part_df):
    if part_df.empty:
        raise ValueError("Cannot infer block size from an empty parquet part.")

    first_key = (part_df.iloc[0]["layer"], part_df.iloc[0]["head"])
    for i in range(1, len(part_df)):
        key = (part_df.iloc[i]["layer"], part_df.iloc[i]["head"])
        if key == first_key:
            return i

    return len(part_df)


def vector_to_array(vector_value):
    if isinstance(vector_value, str):
        return np.asarray(ast.literal_eval(vector_value), dtype=float)
    return np.asarray(vector_value, dtype=float)


def compute_scores(mapping_df, attention_dir):
    part_files = sorted(glob.glob(os.path.join(attention_dir, "*.parquet")))
    if not part_files:
        raise FileNotFoundError(f"No parquet part files found in {attention_dir}")

    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    mapping_idx = 0
    block_size = None

    for part_i, part_file in enumerate(part_files, start=1):
        part_df = pd.read_parquet(
            part_file,
            columns=["sequence_name", "layer", "head", "attention_vector"],
        )
        if part_df.empty:
            continue

        current_block_size = infer_block_size(part_df)
        if block_size is None:
            block_size = current_block_size
            print(f"Inferred {block_size} layer/head rows per sequence.")
        elif current_block_size != block_size:
            raise ValueError(
                f"Inconsistent block size in {part_file}: "
                f"expected {block_size}, got {current_block_size}"
            )

        if len(part_df) % block_size != 0:
            raise ValueError(f"Parquet part does not contain complete sequence blocks: {part_file}")

        num_blocks = len(part_df) // block_size
        for block_start in range(0, len(part_df), block_size):
            if mapping_idx >= len(mapping_df):
                raise ValueError("More attention blocks than mapping rows.")

            mapping_row = mapping_df.iloc[mapping_idx]
            block = part_df.iloc[block_start:block_start + block_size]
            if "sequence_name" in mapping_df.columns:
                mapping_sequence = str(mapping_row["sequence_name"])
                block_sequence = str(block.iloc[0]["sequence_name"])
                if mapping_sequence != block_sequence:
                    raise ValueError(
                        f"Sequence order mismatch at mapping row {mapping_idx}: "
                        f"mapping={mapping_sequence}, parquet={block_sequence}"
                    )

            mapping = parse_mapping(mapping_row["mapping"])
            motif_length = float(mapping_row["motif_length"])

            for _, attention_row in block.iterrows():
                attention_vector = vector_to_array(attention_row["attention_vector"])
                min_len = min(len(mapping), len(attention_vector))
                motif_score = float(np.dot(mapping[:min_len], attention_vector[:min_len]))
                normalized_score = motif_score / motif_length if motif_length > 0 else 0.0

                key = (int(attention_row["layer"]), int(attention_row["head"]))
                score_sums[key] += normalized_score
                score_counts[key] += 1

            mapping_idx += 1

        print(f"Processed parquet part {part_i}/{len(part_files)} ({num_blocks} sequence blocks).")

    if mapping_idx != len(mapping_df):
        raise ValueError(f"Processed {mapping_idx} attention blocks but mapping CSV has {len(mapping_df)} rows.")

    rows = [
        {"layer": layer, "head": head, "score": score_sums[(layer, head)] / score_counts[(layer, head)]}
        for layer, head in sorted(score_sums)
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build layer/head attention_scores CSV from streamed attention parquet parts and motif mapping CSV."
    )
    parser.add_argument("--tf_name", required=True)
    parser.add_argument("--model_type", default="DNABERT2_5.6")
    parser.add_argument("--input_type", default="threshold", choices=["threshold", "ratio"])
    parser.add_argument("--processed_data_dir", default=DEFAULT_PROCESSED_DATA_DIR)
    parser.add_argument("--mapping_csv", default=None)
    parser.add_argument("--attention_dir", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = os.path.join(args.processed_data_dir, args.tf_name, args.model_type)
    mapping_csv = args.mapping_csv or os.path.join(
        model_dir, f"motif_mapping_{args.model_type}_{args.input_type}.csv"
    )
    attention_dir = args.attention_dir or os.path.join(
        model_dir, f"attention_scores_{args.model_type}_original"
    )
    output_csv = args.output_csv or os.path.join(
        model_dir, f"attention_scores_{args.model_type}_{args.input_type}.csv"
    )

    if os.path.exists(output_csv) and not args.overwrite:
        print(f"Output already exists, skipping: {output_csv}")
        return

    print(f"Mapping CSV: {mapping_csv}")
    print(f"Attention parquet dir: {attention_dir}")
    print(f"Output CSV: {output_csv}")

    mapping_df = pd.read_csv(mapping_csv)
    scores_df = compute_scores(mapping_df, attention_dir)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    scores_df.to_csv(output_csv, index=False)
    print(f"Saved attention scores: {output_csv}")


if __name__ == "__main__":
    main()
