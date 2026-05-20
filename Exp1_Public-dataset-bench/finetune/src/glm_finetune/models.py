from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BigBirdForSequenceClassification


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dnabert2_117m": {
        "path": "dnabert2_117m",
        "loader": "auto_sequence",
        "trust_remote_code": True,
        "strategy": "tokenizer_p999",
        "hard_max_length": 512,
        "pooling_or_head": "AutoModelForSequenceClassification CLS pooler",
    },
    "ntv2_500m_multi": {
        "path": "ntv2_500m_multi",
        "loader": "auto_sequence",
        "trust_remote_code": True,
        "strategy": "ntv2_6mer",
        "hard_max_length": 2050,
        "pooling_or_head": "EsmForSequenceClassification release head",
    },
    "hyenadna_large_1m": {
        "path": "hyenadna_large_1m",
        "loader": "auto_sequence",
        "trust_remote_code": True,
        "strategy": "nucleotide",
        "hard_max_length": 1000002,
        "pooling_or_head": "HyenaDNA last non-pad token head",
    },
    "gena_bigbird_t2t": {
        "path": "gena_bigbird_t2t",
        "loader": "bigbird_sequence",
        "trust_remote_code": False,
        "strategy": "tokenizer_p999",
        "hard_max_length": 4096,
        "pooling_or_head": "BigBirdForSequenceClassification head",
    },
    "grover": {
        "path": "grover",
        "loader": "auto_sequence",
        "trust_remote_code": False,
        "strategy": "tokenizer_p999",
        "hard_max_length": 512,
        "pooling_or_head": "BertForSequenceClassification head",
    },
}


def load_tokenizer(model_root: Path, model_alias: str) -> Any:
    cfg = MODEL_CONFIGS[model_alias]
    model_path = _resolve_model_path(model_root, cfg["path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token or tokenizer.unk_token
    return tokenizer


def load_model(model_root: Path, model_alias: str, num_labels: int, id2label: dict[int, str]) -> Any:
    cfg = MODEL_CONFIGS[model_alias]
    model_path = _resolve_model_path(model_root, cfg["path"])
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True,
        num_labels=num_labels,
        id2label={str(k): v for k, v in id2label.items()},
        label2id={v: k for k, v in id2label.items()},
    )
    if cfg["loader"] == "bigbird_sequence":
        model = BigBirdForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=cfg["trust_remote_code"],
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
    if getattr(model.config, "pad_token_id", None) is None:
        tokenizer_config = _read_tokenizer_config(model_path)
        model.config.pad_token_id = tokenizer_config.get("pad_token_id")
    if model_alias == "dnabert2_117m":
        _force_dnabert2_pytorch_attention(model)
    return model


def resolve_max_length(
    model_alias: str,
    tokenizer: Any,
    sequences: list[str],
    sample_size: int,
    allow_truncation: bool,
) -> tuple[int, dict[str, float]]:
    cfg = MODEL_CONFIGS[model_alias]
    sample = sequences[:sample_size] if len(sequences) > sample_size else sequences
    lengths = np.array([len(tokenizer(seq, add_special_tokens=True, truncation=False)["input_ids"]) for seq in sample])
    if lengths.size == 0:
        raise ValueError("Cannot resolve max_length from an empty sequence list")

    stats = {
        "token_len_max": float(np.max(lengths)),
        "token_len_p95": float(np.percentile(lengths, 95)),
        "token_len_p99": float(np.percentile(lengths, 99)),
        "token_len_p999": float(np.percentile(lengths, 99.9)),
    }
    bp_max = max(len(seq) for seq in sample)
    configured = _initial_max_length(cfg["strategy"], stats, bp_max)
    max_length = int(max(configured, math.ceil(stats["token_len_p999"])))
    hard_max = int(cfg["hard_max_length"])
    if max_length > hard_max:
        if not allow_truncation:
            raise ValueError(
                f"{model_alias} needs max_length {max_length}, above hard limit {hard_max}. "
                "Pass --allow-truncation to truncate explicitly."
            )
        max_length = hard_max

    stats["configured_max_length"] = float(max_length)
    stats["truncation_rate"] = float(np.mean(lengths > max_length))
    return max_length, stats


def model_metadata(model_alias: str) -> dict[str, Any]:
    return dict(MODEL_CONFIGS[model_alias])


def _resolve_model_path(model_root: Path, configured_path: str) -> Path:
    model_path = model_root / configured_path
    if model_path.exists():
        return model_path

    folded = configured_path.casefold()
    if model_root.exists():
        for child in model_root.iterdir():
            if child.is_dir() and child.name.casefold() == folded:
                return child
    return model_path


def _initial_max_length(strategy: str, stats: dict[str, float], bp_max: int) -> int:
    if strategy == "ntv2_6mer":
        return int(math.ceil(bp_max / 6) + 8)
    if strategy == "nucleotide":
        return int(bp_max + 8)
    return int(math.ceil(stats["token_len_p999"]))


def _read_tokenizer_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "tokenizer_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _force_dnabert2_pytorch_attention(model: Any) -> None:
    """Avoid the bundled Triton kernel, which is incompatible with local Triton."""
    for module in model.modules():
        if hasattr(module, "p_dropout") and hasattr(module, "Wqkv"):
            module.p_dropout = 1e-12
