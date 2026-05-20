from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from .datasets import SequenceDataset, load_task
from .manifest import append_summary, append_token_summary, dataset_manifest, write_json
from .metrics import compute_classification_metrics
from .models import MODEL_CONFIGS, load_model, load_tokenizer, model_metadata, resolve_max_length
from .paths import default_data_root, default_model_root, experiment_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune local genome language models on local classification tasks.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--dataset", choices=["NT", "GUE", "TFBS"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Root containing NT/, GUE/, and TF_data_all/.")
    parser.add_argument("--model-root", type=Path, default=default_model_root(), help="Root containing local model directories.")
    parser.add_argument("--output-root", type=Path, default=experiment_root() / "outputs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-truncation", action="store_true")
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
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader (default enabled when CUDA is available).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_root / args.model / args.dataset / Path(*args.task.split("/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    split_data = _cap_splits(load_task(args.data_root, args.dataset, args.task, args.seed), args.max_train_samples, args.max_eval_samples)
    tokenizer = load_tokenizer(args.model_root, args.model)
    all_sequences = split_data.train["sequence"].tolist() + split_data.validation["sequence"].tolist() + split_data.test["sequence"].tolist()
    max_length, token_stats = resolve_max_length(args.model, tokenizer, all_sequences, args.token_length_sample_size, args.allow_truncation)

    write_json(out_dir / "dataset_manifest.json", dataset_manifest(args.dataset, args.task, split_data, token_stats))
    append_token_summary(args.output_root / "token_length_summary.csv", {"model": args.model, "dataset": args.dataset, "task": args.task, **token_stats})

    model = load_model(args.model_root, args.model, split_data.num_labels, split_data.id2label)
    train_dataset = SequenceDataset(split_data.train, tokenizer, max_length)
    eval_dataset = SequenceDataset(split_data.validation, tokenizer, max_length)
    test_dataset = SequenceDataset(split_data.test, tokenizer, max_length)

    write_json(
        out_dir / "config.resolved.json",
        {
            "model": args.model,
            "dataset": args.dataset,
            "task": args.task,
            "seed": args.seed,
            "max_length": max_length,
            "num_labels": split_data.num_labels,
            "training_args": vars(args),
            "model_metadata": model_metadata(args.model),
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        _run_smoke_forward(model, train_dataset, device)
        print(f"SMOKE_OK {args.model} {args.dataset}/{args.task} max_length={max_length}")
        return

    result = train_and_evaluate(args, model, tokenizer, train_dataset, eval_dataset, test_dataset, out_dir, device)
    append_summary(args.output_root / "summary_metrics.csv", _summary_row(args, split_data.num_labels, max_length, result, out_dir))


def train_and_evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: SequenceDataset,
    eval_dataset: SequenceDataset,
    test_dataset: SequenceDataset,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    pin_memory = device.type == "cuda" and not args.no_pin_memory
    num_workers = max(0, int(args.dataloader_num_workers))
    persistent = num_workers > 0
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent}
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, **loader_kwargs)

    grad_accum = max(1, math.ceil(args.effective_batch_size / args.per_device_train_batch_size))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, int(math.ceil(args.num_train_epochs) * updates_per_epoch))
    warmup_steps = max(50, int(args.warmup_ratio * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_mcc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, int(math.ceil(args.num_train_epochs)) + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            with _autocast_context(args, device):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().cpu()) * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    _write_metrics(metrics_path, {"step": global_step, "epoch": epoch, "loss": running_loss / max(1, step)})

        eval_metrics = evaluate(model, eval_loader, device, args, "eval")
        eval_metrics.update({"step": global_step, "epoch": epoch, "train_loss": running_loss / max(1, len(train_loader))})
        _write_metrics(metrics_path, eval_metrics)
        print(json.dumps(eval_metrics, sort_keys=True))

        current_mcc = float(eval_metrics.get("eval_mcc", -float("inf")))
        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(model, tokenizer, out_dir / "checkpoint-best")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch} best_val_mcc={best_mcc}")
                break

    test_metrics = evaluate(model, test_loader, device, args, "test")
    write_json(out_dir / "test_metrics.json", test_metrics)
    _write_metrics(metrics_path, {"best_epoch": best_epoch, "best_val_mcc": best_mcc, **test_metrics})
    return {"best_val_mcc": best_mcc, "test_metrics": test_metrics, "effective_batch_size": args.per_device_train_batch_size * grad_accum}


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace, prefix: str) -> dict[str, float]:
    model.eval()
    logits_parts = []
    label_parts = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch["labels"].detach().cpu()
            with _autocast_context(args, device):
                outputs = model(**batch)
            total_loss += float(outputs.loss.detach().cpu())
            logits_parts.append(outputs.logits.detach().float().cpu())
            label_parts.append(labels)
    metrics = compute_classification_metrics((torch.cat(logits_parts).numpy(), torch.cat(label_parts).numpy()))
    metrics = {f"{prefix}_{key}": value for key, value in metrics.items()}
    metrics[f"{prefix}_loss"] = total_loss / max(1, len(loader))
    return metrics


def _run_smoke_forward(model: torch.nn.Module, dataset: SequenceDataset, device: torch.device) -> None:
    model.to(device)
    model.eval()
    item = dataset[0]
    batch = {key: (torch.tensor([value]) if not torch.is_tensor(value) else value.unsqueeze(0)).to(device) for key, value in item.items()}
    with torch.no_grad():
        output = model(**batch)
    if output.logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected smoke logits shape: {tuple(output.logits.shape)}")


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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _autocast_context(args: argparse.Namespace, device: torch.device) -> Any:
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    return torch.cuda.amp.autocast(enabled=args.bf16 or args.fp16, dtype=dtype)


def _save_checkpoint(model: torch.nn.Module, tokenizer: Any, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)


def _write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _summary_row(args: argparse.Namespace, num_labels: int, max_length: int, result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    test_metrics = result["test_metrics"]
    return {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "num_labels": num_labels,
        "primary_metric": "eval_mcc",
        "best_val_mcc": result["best_val_mcc"],
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
        "effective_batch_size": result["effective_batch_size"],
        "checkpoint_path": str(out_dir / "checkpoint-best"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
