from __future__ import annotations

from pathlib import Path


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return experiment_root().parents[1]


def default_data_root() -> Path:
    return repo_root() / "Data"


def default_model_root() -> Path:
    return repo_root() / "GLM_weights"
