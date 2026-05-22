#!/usr/bin/env python
# coding: utf-8

"""
Aggregate AUPRC values from per-model result JSON files into a summary CSV.

Expected directory layout:
  <outputs_root>/<TF>/<model_group>/<model_variant>/results/{val,test}_results.json

Run:
  python -m exp3_dnase.analysis.extract_auprc [--outputs_dir path/to/outputs]
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

TF_LIST = ["BRD4", "CTCF", "EZH2", "GABPA", "POLR2A", "USF2"]


def extract_auprc(outputs_dir: str):
    outputs_path = Path(outputs_dir)
    records = []

    for tf in TF_LIST:
        tf_path = outputs_path / tf
        if not tf_path.is_dir():
            print(f"Warning: no directory for TF {tf} at {tf_path}")
            continue

        for model_group in sorted(tf_path.iterdir()):
            if not model_group.is_dir() or model_group.name == "Enformer":
                continue

            for model_variant in sorted(model_group.iterdir()):
                if not model_variant.is_dir():
                    continue

                model_name = f"{model_group.name}_{model_variant.name}"

                for split in ("val", "test"):
                    result_file = model_variant / "results" / f"{split}_results.json"
                    auprc = None
                    if result_file.exists():
                        try:
                            with result_file.open() as f:
                                auprc = json.load(f).get("eval_auprc")
                        except json.JSONDecodeError:
                            print(f"Warning: could not parse {result_file}")

                    if auprc is not None:
                        records.append({"tf": tf, "model": model_name, "split": split, "auprc": auprc})

    if not records:
        print("No results found. Check --outputs_dir.")
        return

    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="model", columns=["tf", "split"], values="auprc")
    pivot.columns = [f"{tf}_{split}" for tf, split in pivot.columns]
    pivot.reset_index(inplace=True)

    output_csv = outputs_path / "auprc_summary_model_rows.csv"
    pivot.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"Summary CSV written to: {output_csv}")
    print(pivot.to_string())


def main():
    parser = argparse.ArgumentParser(description="Aggregate AUPRC from finetune result JSONs.")
    parser.add_argument("--outputs_dir", default="scripts/finetune/outputs",
                        help="Root directory containing per-TF/model results")
    args = parser.parse_args()
    extract_auprc(args.outputs_dir)


if __name__ == "__main__":
    main()
