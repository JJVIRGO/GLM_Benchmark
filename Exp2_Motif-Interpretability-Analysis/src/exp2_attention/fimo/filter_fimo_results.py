#!/usr/bin/env python3
"""
Filter FIMO results to keep only the highest scoring record for each sequence.

This script reads a FIMO TSV output file and filters it to retain only the
highest scoring (most confident) record for each unique sequence_name.

Usage:
    python filter_fimo_results.py <input_file> [-o <output_file>]

Arguments:
    input_file: Path to the input FIMO TSV file
    -o, --output: Optional output file path (default: fimo_filtered.tsv in input directory)

Examples:
    python filter_fimo_results.py fimo.tsv
    python filter_fimo_results.py fimo.tsv -o filtered_results.tsv
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path


def filter_fimo_results(input_file, output_file=None):
    """
    Filter FIMO results to keep only the highest scoring record per sequence.

    Args:
        input_file (str): Path to the input FIMO TSV file
        output_file (str, optional): Path to the output filtered TSV file.
                                   If None, will use 'fimo_filtered.tsv' in the same directory

    Returns:
        str: Path to the output file
    """
    # Read the TSV file
    print(f"Reading FIMO results from: {input_file}")
    df = pd.read_csv(input_file, sep='\t')

    print(f"Total records: {len(df)}")
    print(f"Unique sequences: {df['sequence_name'].nunique()}")

    # Group by sequence_name and find the index of the maximum score for each group
    # For ties, idxmax will return the first occurrence
    idx_max_score = df.groupby('sequence_name')['score'].idxmax()

    # Filter the dataframe to keep only the highest scoring records
    filtered_df = df.loc[idx_max_score]

    # Sort by sequence_name for consistent output
    filtered_df = filtered_df.sort_values('sequence_name').reset_index(drop=True)

    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / 'fimo_filtered.tsv'

    # Save the filtered results
    print(f"Saving filtered results to: {output_file}")
    print(f"Filtered records: {len(filtered_df)}")

    filtered_df.to_csv(output_file, sep='\t', index=False)

    return str(output_file)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter FIMO results to keep only the highest scoring record for each sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_fimo_results.py input.tsv
  python filter_fimo_results.py input.tsv -o output_filtered.tsv
  python filter_fimo_results.py /path/to/fimo.tsv --output /path/to/filtered.tsv
  python filter_fimo_results.py /path/to/fimo.tsv --output /path/to/filtered.tsv
        """
    )

    parser.add_argument(
        'input_file',
        help='Path to the input FIMO TSV file'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='Path to the output filtered TSV file (default: fimo_filtered.tsv in input directory)'
    )

    return parser.parse_args()


def main():
    """Main function to run the filtering process."""
    # Parse command line arguments
    args = parse_arguments()

    input_file = args.input_file
    output_file = args.output_file

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    try:
        # Filter the results
        output_file = filter_fimo_results(input_file, output_file)

        print("\nFiltering completed successfully!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")

        # Show some statistics
        df_original = pd.read_csv(input_file, sep='\t')
        df_filtered = pd.read_csv(output_file, sep='\t')

        print("\nStatistics:")
        print(f"Original records: {len(df_original)}")
        print(f"Filtered records: {len(df_filtered)}")
        print(f"Sequences processed: {len(df_filtered)}")
        print(f"Average score of filtered records: {df_filtered['score'].mean():.2f}")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
