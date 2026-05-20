from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import transformers as _transformers

# The shared glm_hf environment currently has peft newer than transformers 4.38.1.
# Trainer imports peft when present; this placeholder is enough for import-time
# compatibility because this experiment does not use PEFT adapters.
if not hasattr(_transformers, "EncoderDecoderCache"):
    class EncoderDecoderCache:  # pragma: no cover - import compatibility only
        pass

    _transformers.EncoderDecoderCache = EncoderDecoderCache

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .datasets import GUE_TASKS, NT_TASKS, SequenceDataset, load_task
from .manifest import append_summary, append_token_summary, dataset_manifest, write_json
from .metrics import compute_classification_metrics
from .models import MODEL_CONFIGS, load_model, load_tokenizer, model_metadata, resolve_max_length
from .paths import default_data_root, default_model_root, experiment_root


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        payload = {"step": state.global_step, "epoch": state.epoch, **_jsonable(logs)}
        with self.path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune local genome language models on Experiment 1 tasks.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--dataset", choices=["NT", "GUE"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Root containing NT/ and GUE/.")
    parser.add_argument("--model-root", type=Path, default=default_model_root(), help="Root containing local model directories.")
    parser.add_argument("--output-root", type=Path, default=experiment_root() / "outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Load data/model and run one tiny forward pass without training.")
    parser.add_argument("--allow-truncation", action="store_true", help="Explicitly allow truncation if token p999 exceeds model limit.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--token-length-sample-size", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--effective-batch-size", type=int, default=128)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task_parts = args.task.split("/")
    out_dir = args.output_root / args.model / args.dataset / Path(*task_parts)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_data = load_task(args.data_root, args.dataset, args.task, args.seed)
    split_data = _cap_splits(split_data, args.max_train_samples, args.max_eval_samples)

    tokenizer = load_tokenizer(args.model_root, args.model)
    all_sequences = (
        split_data.train["sequence"].tolist()
        + split_data.validation["sequence"].tolist()
        + split_data.test["sequence"].tolist()
    )
    max_length, token_stats = resolve_max_length(
        args.model,
        tokenizer,
        all_sequences,
        args.token_length_sample_size,
        args.allow_truncation,
    )
    manifest = dataset_manifest(args.dataset, args.task, split_data, token_stats)
    write_json(out_dir / "dataset_manifest.json", manifest)
    append_token_summary(
        args.output_root / "token_length_summary.csv",
        {"model": args.model, "dataset": args.dataset, "task": args.task, **token_stats},
    )

    model = load_model(args.model_root, args.model, split_data.num_labels, split_data.id2label)
    train_dataset = SequenceDataset(split_data.train, tokenizer, max_length)
    eval_dataset = SequenceDataset(split_data.validation, tokenizer, max_length)
    test_dataset = SequenceDataset(split_data.test, tokenizer, max_length)

    resolved_config = {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "seed": args.seed,
        "max_length": max_length,
        "num_labels": split_data.num_labels,
        "training_args": vars(args),
        "model_metadata": model_metadata(args.model),
    }
    write_json(out_dir / "config.resolved.json", resolved_config)

    if args.smoke_test:
        _run_smoke_forward(model, train_dataset)
        print(f"SMOKE_OK {args.model} {args.dataset}/{args.task} max_length={max_length}")
        return

    grad_accum = max(1, math.ceil(args.effective_batch_size / args.per_device_train_batch_size))
    warmup_steps = max(50, int(args.warmup_ratio * math.ceil(len(train_dataset) / args.per_device_train_batch_size / grad_accum)))
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=1,
        metric_for_best_model="mcc",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to=[],
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            JsonlMetricsCallback(out_dir / "metrics.jsonl"),
        ],
    )
    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    write_json(out_dir / "test_metrics.json", _jsonable(test_metrics))
    _save_best_checkpoint(trainer, out_dir)
    append_summary(args.output_root / "summary_metrics.csv", _summary_row(args, split_data.num_labels, max_length, trainer, test_metrics, out_dir))


def _run_smoke_forward(model: torch.nn.Module, dataset: SequenceDataset) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    item = dataset[0]
    batch = {
        key: (torch.tensor([value]) if not torch.is_tensor(value) else value.unsqueeze(0)).to(device)
        for key, value in item.items()
    }
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    if logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected smoke logits shape: {tuple(logits.shape)}")


def _cap_splits(split_data: Any, max_train: int | None, max_eval: int | None) -> Any:
    if max_train is None and max_eval is None:
        return split_data
    from .datasets import SplitData

    return SplitData(
        train=split_data.train.head(max_train).reset_index(drop=True) if max_train else split_data.train,
        validation=split_data.validation.head(max_eval).reset_index(drop=True) if max_eval else split_data.validation,
        test=split_data.test.head(max_eval).reset_index(drop=True) if max_eval else split_data.test,
        label2id=split_data.label2id,
        id2label=split_data.id2label,
    )


def _save_best_checkpoint(trainer: Trainer, out_dir: Path) -> None:
    best = trainer.state.best_model_checkpoint
    target = out_dir / "checkpoint-best"
    if target.exists():
        shutil.rmtree(target)
    if best:
        shutil.copytree(best, target)
    else:
        trainer.save_model(str(target))


def _summary_row(args: argparse.Namespace, num_labels: int, max_length: int, trainer: Trainer, test_metrics: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    return {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "num_labels": num_labels,
        "primary_metric": "eval_mcc",
        "best_val_mcc": trainer.state.best_metric,
        "test_mcc": test_metrics.get("test_mcc"),
        "test_accuracy": test_metrics.get("test_accuracy"),
        "test_f1_binary": test_metrics.get("test_f1_binary"),
        "test_f1_macro": test_metrics.get("test_f1_macro"),
        "test_auroc_binary": test_metrics.get("test_auroc_binary"),
        "test_auroc_macro": test_metrics.get("test_auroc_macro"),
        "test_auprc_binary": test_metrics.get("test_auprc_binary"),
        "test_auprc_macro": test_metrics.get("test_auprc_macro"),
        "seed": args.seed,
        "max_length": max_length,
        "pooling_or_head": model_metadata(args.model)["pooling_or_head"],
        "effective_batch_size": args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps,
        "checkpoint_path": str(out_dir / "checkpoint-best"),
    }


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif hasattr(value, "item"):
            out[key] = value.item()
        elif isinstance(value, (Path, os.PathLike)):
            out[key] = str(value)
        else:
            out[key] = str(value)
    return out


if __name__ == "__main__":
    main()
