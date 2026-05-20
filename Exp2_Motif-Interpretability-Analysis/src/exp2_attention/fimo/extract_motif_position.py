#!/usr/bin/env python3
"""
Extract motif positions using FIMO from CTCF sequences.

This script performs two main tasks:
1. Convert sequences from CSV file to FASTA format
2. Use FIMO to scan for motif positions in the sequences
  
  
Usage:
    python extract_motif_position.py --input_csv <csv_file> --input_meme <meme_file>
"""

import argparse
import pandas as pd
import subprocess
import os
import sys
from pathlib import Path


def csv_to_fasta(csv_file, output_fasta):
    """
    Convert CSV sequences to FASTA format.

    Args:
        csv_file (str): Path to input CSV file
        output_fasta (str): Path to output FASTA file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} sequences from {csv_file}")

        # Write FASTA file
        with open(output_fasta, 'w') as f:
            for idx, row in df.iterrows():
                sequence = row['sequence']
                # Use sequence index as identifier
                header = f">seq_{idx + 1}"
                f.write(f"{header}\n{sequence}\n")

        print(f"Converted sequences to FASTA format: {output_fasta}")
        return True

    except Exception as e:
        print(f"Error converting CSV to FASTA: {e}")
        return False


def merge_meme_files(meme_files, output_meme_file):
    """
    Merge multiple MEME files into a single file.

    Args:
        meme_files (list): List of MEME file paths to merge
        output_meme_file (str): Output merged MEME file path
    """
    with open(output_meme_file, 'w') as outfile:
        header_written = False

        for meme_file in meme_files:
            with open(meme_file, 'r') as infile:
                lines = infile.readlines()

                # Write header only once (from first file)
                if not header_written:
                    # Find where MOTIF section starts
                    motif_start_idx = -1
                    for i, line in enumerate(lines):
                        if line.startswith('MOTIF'):
                            motif_start_idx = i
                            break

                    if motif_start_idx != -1:
                        # Write header (everything before first MOTIF)
                        outfile.writelines(lines[:motif_start_idx])
                        header_written = True

                # Write MOTIF sections from all files
                in_motif_section = False
                for line in lines:
                    if line.startswith('MOTIF'):
                        in_motif_section = True
                        outfile.write(line)
                    elif in_motif_section:
                        outfile.write(line)

                        # Check if this is the end of motif (next MOTIF or end of file)
                        if line.strip() == '' and len(lines) > lines.index(line) + 1:
                            next_line = lines[lines.index(line) + 1]
                            if next_line.startswith('MOTIF'):
                                continue

    print(f"Merged {len(meme_files)} MEME files into {output_meme_file}")


def run_fimo(fasta_file, meme_files, output_dir):
    """
    Run FIMO to scan for motif positions.

    Args:
        fasta_file (str): Path to input FASTA file
        meme_files (list): List of paths to input MEME files
        output_dir (str): Output directory for FIMO results
    """
    try:
        # Remove output directory if it exists, then create it
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Try direct multi-MEME scanning first
        print("Attempting to scan with multiple MEME files directly...")
        success = run_fimo_multiple_memes(fasta_file, meme_files, output_dir)

        if success:
            print("Direct multi-MEME scanning succeeded!")
            return True
        else:
            print("Direct multi-MEME scanning failed, trying merged MEME approach...")

        # Fallback: merge MEME files and scan
        merged_meme_file = os.path.join(output_dir, "merged_motifs.meme")
        merge_meme_files(meme_files, merged_meme_file)

        # Run FIMO with merged MEME file
        conda_init_cmd = "source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source ~/.conda/etc/profile.d/conda.sh"
        fimo_cmd = f"fimo --oc {output_dir} {merged_meme_file} {fasta_file}"

        cmd = [
            "bash", "-c",
            f"{conda_init_cmd} && conda activate fimo_env && {fimo_cmd}"
        ]

        print(f"Running FIMO with merged MEME file")
        print(f"Command: {conda_init_cmd} && conda activate fimo_env && {fimo_cmd}")

        # Execute FIMO
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("FIMO scan with merged MEME completed successfully")
            print(f"Results saved to: {output_dir}")

            # Check if results exist
            fimo_tsv = os.path.join(output_dir, "fimo.tsv")
            if os.path.exists(fimo_tsv):
                print(f"FIMO results file: {fimo_tsv}")
                return True
            else:
                print("Warning: FIMO results file not found")
                return False
        else:
            print(f"FIMO failed with error: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error running FIMO: {e}")
        return False


def run_fimo_multiple_memes(fasta_file, meme_files, output_dir):
    """
    Try to run FIMO with multiple MEME files directly.

    Args:
        fasta_file (str): Path to input FASTA file
        meme_files (list): List of paths to input MEME files
        output_dir (str): Output directory for FIMO results

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Construct command with multiple MEME files
        meme_args = ' '.join(meme_files)

        conda_init_cmd = "source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source ~/.conda/etc/profile.d/conda.sh"
        fimo_cmd = f"fimo --oc {output_dir} {meme_args} {fasta_file}"

        cmd = [
            "bash", "-c",
            f"{conda_init_cmd} && conda activate fimo_env && {fimo_cmd}"
        ]

        print(f"Attempting FIMO with multiple MEME files: {meme_args}")

        # Execute FIMO
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"Direct multi-MEME FIMO failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error in direct multi-MEME FIMO: {e}")
        return False




def main():
    parser = argparse.ArgumentParser(description="Extract motif positions from sequences using FIMO")

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV file containing sequences"
    )

    parser.add_argument(
        "--input_meme",
        nargs='+',
        required=True,
        help="Path(s) to input MEME file(s) containing motifs (can specify multiple files)"
    )

    parser.add_argument(
        "--output_dir",
        default="fimo_results",
        help="Output directory for FIMO results (default: fimo_results)"
    )

    parser.add_argument(
        "--fasta_file",
        default="sequences.fasta",
        help="Output FASTA file name (default: sequences.fasta)"
    )

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file does not exist: {args.input_csv}")
        sys.exit(1)

    # Check if all MEME files exist
    for meme_file in args.input_meme:
        if not os.path.exists(meme_file):
            print(f"Error: Input MEME file does not exist: {meme_file}")
            sys.exit(1)

    # Step 1: Convert CSV to FASTA
    print("Step 1: Converting CSV to FASTA format...")
    if not csv_to_fasta(args.input_csv, args.fasta_file):
        sys.exit(1)

    # Step 2: Run FIMO
    print(f"\nStep 2: Running FIMO motif scanning with {len(args.input_meme)} MEME file(s)...")
    if not run_fimo(args.fasta_file, args.input_meme, args.output_dir):
        sys.exit(1)

    print("\nProcess completed successfully!")
    print(f"FASTA file: {args.fasta_file}")
    print(f"FIMO results: {args.output_dir}/fimo.tsv")


if __name__ == "__main__":
    main()
