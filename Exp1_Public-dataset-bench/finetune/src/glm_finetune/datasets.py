from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split


NT_TASKS = [
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "enhancers",
    "enhancers_types",
    "splice_sites_all",
    "splice_sites_acceptors",
    "splice_sites_donors",
    "H2AFZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "H4K20me1",
]

GUE_TASKS = [
    "EMP/H3",
    "EMP/H3K14ac",
    "EMP/H3K36me3",
    "EMP/H3K4me1",
    "EMP/H3K4me2",
    "EMP/H3K4me3",
    "EMP/H3K79me3",
    "EMP/H3K9ac",
    "EMP/H4",
    "EMP/H4ac",
    "prom/prom_300_all",
    "prom/prom_300_notata",
    "prom/prom_300_tata",
    "prom/prom_core_all",
    "prom/prom_core_notata",
    "prom/prom_core_tata",
    "splice/reconstructed",
    "tf/0",
    "tf/1",
    "tf/2",
    "tf/3",
    "tf/4",
]

TFBS_TASKS = [
    "CTCF",
    "HNF4G",
    "IRF1",
    "IRF1_new",
    "REST",
    "USF2",
    "YY1",
]


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label2id: dict[Any, int]
    id2label: dict[int, str]

    @property
    def num_labels(self) -> int:
        return len(self.label2id)


def load_task(data_root: Path, dataset_name: str, task_name: str, seed: int) -> SplitData:
    dataset_name = dataset_name.upper()
    if dataset_name == "NT":
        return _load_nt(data_root, task_name, seed)
    if dataset_name == "GUE":
        return _load_gue(data_root, task_name)
    if dataset_name == "TFBS":
        return _load_tfbs(data_root, task_name)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_nt(data_root: Path, task_name: str, seed: int) -> SplitData:
    task_dir = data_root / "NT" / task_name
    train_path = task_dir / "train.parquet"
    test_path = task_dir / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing NT parquet files under {task_dir}")

    train_full = _normalize_frame(pd.read_parquet(train_path))
    test = _normalize_frame(pd.read_parquet(test_path))
    val_size = min(len(test), max(1, len(train_full) - 1))

    stratify = train_full["label"] if train_full["label"].value_counts().min() >= 2 else None
    train, validation = train_test_split(
        train_full,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return _finalize_splits(train, validation, test)


def _load_gue(data_root: Path, task_name: str) -> SplitData:
    task_dir = data_root / "GUE" / task_name
    return _load_csv_splits(task_dir, "GUE")


def _load_tfbs(data_root: Path, task_name: str) -> SplitData:
    if task_name not in TFBS_TASKS:
        raise ValueError(f"Unsupported TFBS task: {task_name}. Expected one of {TFBS_TASKS}")
    task_dir = data_root / "TF_data_all" / "Exist_motif" / task_name
    return _load_csv_splits(task_dir, "TFBS")


def _load_csv_splits(task_dir: Path, dataset_name: str) -> SplitData:
    paths = {
        "train": task_dir / "train.csv",
        "validation": task_dir / "dev.csv",
        "test": task_dir / "test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {dataset_name} files: {missing}")

    train = _normalize_frame(pd.read_csv(paths["train"]))
    validation = _normalize_frame(pd.read_csv(paths["validation"]))
    test = _normalize_frame(pd.read_csv(paths["test"]))
    return _finalize_splits(train, validation, test)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"Expected sequence,label columns, got {list(frame.columns)}")
    out = frame.loc[:, ["sequence", "label"]].copy()
    out["sequence"] = out["sequence"].astype(str).str.upper()
    return out.dropna(subset=["sequence", "label"]).reset_index(drop=True)


def _finalize_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> SplitData:
    labels = pd.concat([train["label"], validation["label"], test["label"]], ignore_index=True)
    ordered_labels = sorted(labels.unique().tolist(), key=lambda value: (str(type(value)), str(value)))
    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: str(label) for label, idx in label2id.items()}

    def map_labels(frame: pd.DataFrame) -> pd.DataFrame:
        mapped = frame.copy()
        mapped["label"] = mapped["label"].map(label2id).astype(int)
        return mapped.reset_index(drop=True)

    return SplitData(map_labels(train), map_labels(validation), map_labels(test), label2id, id2label)


class SequenceDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.sequences = frame["sequence"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.sequences[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()} | {"labels": torch.tensor(self.labels[index], dtype=torch.long)}
