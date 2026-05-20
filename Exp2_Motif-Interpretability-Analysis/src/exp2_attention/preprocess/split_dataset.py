#!/usr/bin/env python3
"""
Script to split balanced dataset CSV files into train, validation, and test sets
with an 8:1:1 ratio.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import argparse

from exp2_attention.paths import PROCESSED_DATA_DIR, TF_LIST

def split_dataset(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split a single CSV dataset into train, validation, and test sets.

    Args:
        file_path (str): Path to the CSV file
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        random_state (int): Random state for reproducibility
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"Processing {file_path}")
    print(f"Total samples: {len(df)}")

    # First split: separate train from temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        random_state=random_state,
        stratify=df['label']  # Maintain class balance
    )

    # Second split: separate val and test from temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
        stratify=temp_df['label']  # Maintain class balance
    )

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df, val_df, test_df

def save_split_datasets(base_dir, target_tfs=None):
    """
    Process balanced dataset CSV files in the given directory.

    Args:
        base_dir (str): Base directory containing subdirectories with CSV files
        target_tfs (list): List of transcription factors to process. If None, process all.
    """
    # Find all balanced dataset CSV files
    pattern = os.path.join(base_dir, "**/*_balanced_dataset.csv")
    csv_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(csv_files)} balanced dataset files")

    for csv_file in csv_files:
        # Get the transcription factor name from the filename
        tf_name = os.path.basename(csv_file).replace('_balanced_dataset.csv', '')

        # Skip if not in target list
        if target_tfs is not None and tf_name not in target_tfs:
            continue
        try:
            # Get the directory containing the CSV file
            file_dir = os.path.dirname(csv_file)

            # Split the dataset
            train_df, val_df, test_df = split_dataset(csv_file)

            # Save the split datasets
            train_path = os.path.join(file_dir, f"train.csv")
            val_path = os.path.join(file_dir, f"val.csv")
            test_path = os.path.join(file_dir, f"test.csv")

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"Saved split datasets for {tf_name}:")
            print(f"  Train: {train_path}")
            print(f"  Val: {val_path}")
            print(f"  Test: {test_path}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue

def cli():
    parser = argparse.ArgumentParser(description="Split balanced TFBS datasets into train/val/test CSV files.")
    parser.add_argument("--base-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--tf-list", nargs="+", default=TF_LIST)
    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        raise FileNotFoundError(f"Directory does not exist: {args.base_dir}")
    save_split_datasets(args.base_dir, target_tfs=args.tf_list)
    print("Dataset splitting completed!")


if __name__ == "__main__":
    cli()
