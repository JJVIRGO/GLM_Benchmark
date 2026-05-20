from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .datasets import SplitData


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def dataset_manifest(dataset_name: str, task_name: str, split_data: SplitData, token_stats: dict[str, float]) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "task": task_name,
        "num_labels": split_data.num_labels,
        "id2label": split_data.id2label,
        "split_sizes": {
            "train": len(split_data.train),
            "validation": len(split_data.validation),
            "test": len(split_data.test),
        },
        "label_counts": {
            "train": _counts(split_data.train),
            "validation": _counts(split_data.validation),
            "test": _counts(split_data.test),
        },
        "token_stats": token_stats,
    }


def append_token_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(label): int(count) for label, count in frame["label"].value_counts().sort_index().items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value
