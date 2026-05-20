"""Shared path defaults for the Experiment 2 repository."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_ROOT = Path(os.environ.get("EXP2_DATA_ROOT", REPO_ROOT / "Data"))
PROCESSED_DATA_DIR = Path(os.environ.get("EXP2_PROCESSED_DATA_DIR", DATA_ROOT / "processed_data"))
REFERENCE_DIR = Path(os.environ.get("EXP2_REFERENCE_DIR", DATA_ROOT / "reference"))
MOTIF_DIR = Path(os.environ.get("EXP2_MOTIF_DIR", DATA_ROOT / "meme"))
MODEL_ROOT = Path(os.environ.get("EXP2_MODEL_ROOT", REPO_ROOT / "GLM_weights"))
OUTPUT_ROOT = Path(os.environ.get("EXP2_OUTPUT_ROOT", REPO_ROOT / "outputs"))
_DISCOVERY_ROOT_ENV = os.environ.get("EXP2_DISCOVERY_ROOT") or os.environ.get("EXP2_RECOVERY_ROOT")
DISCOVERY_ROOT = Path(_DISCOVERY_ROOT_ENV) if _DISCOVERY_ROOT_ENV else OUTPUT_ROOT / "motif_discovery"
RECOVERY_ROOT = DISCOVERY_ROOT
STAT_ROOT = Path(os.environ.get("EXP2_STAT_ROOT", OUTPUT_ROOT / "motif_attention_stats"))

TF_LIST = [
    "CTCF",
    "FOXA1",
    "GATA1",
    "GATA4",
    "JUN",
    "LDB1",
    "MEF2A",
    "MYC",
    "NRF1",
    "SPI1",
    "USF2",
    "YY1",
]
TF_LIST_WITH_KNOWN_MOTIF = [tf for tf in TF_LIST if tf != "LDB1"]

MODEL_NAME_TO_DIR = {
    "DNABERT-2": "DNABERT-2-117M",
    "DNABERT2_5.6": "DNABERT-2-117M",
    "GENA_LM_BERT": "GENA_LM_BERT",
    "gena": "GENA_LM_BERT",
    "GROVER": "GROVER",
    "grover": "GROVER",
    "dnabert2": "DNABERT-2-117M",
    "NT": "NT/NT_500M_model",
}


def model_path(model_type: str, model_root: str | os.PathLike[str] | None = None) -> str:
    root = Path(model_root) if model_root is not None else MODEL_ROOT
    return str(root / MODEL_NAME_TO_DIR.get(model_type, model_type))
