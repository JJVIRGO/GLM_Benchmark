#!/usr/bin/env python
# coding: utf-8

"""
Fine-tune HyenaDNA with optional DNase-seq signal fusion.

HyenaDNA uses single-base tokenization (each token = 1 nucleotide),
so DNase signal alignment is direct positional indexing.

Default backbone: LongSafari/hyenadna-small-32k-seqlen-hf
Override with HYENA_MODEL_PATH env var or --model_path argument.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import sklearn.metrics
import torch
import transformers
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Default model identifiers; override via env vars or --model_path
_LOCAL_PATHS = {
    "hyena-small":  os.environ.get("HYENA_SMALL_PATH",  "LongSafari/hyenadna-small-32k-seqlen-hf"),
    "hyena-medium": os.environ.get("HYENA_MEDIUM_PATH", "LongSafari/hyenadna-medium-160k-seqlen-hf"),
}


def load_model_and_tokenizer(model_name: str, model_path: str, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, model_max_length=512)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels,
        trust_remote_code=True, output_hidden_states=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or "[PAD]"

    if hasattr(model, "config") and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name:      str           = field(default="hyena-small")
    model_path:      Optional[str] = field(default=None)
    hidden_dim:      int           = field(default=256)
    freeze_backbone: bool          = field(default=True)


@dataclass
class DataArguments:
    train_data_path: str = field(default=None)
    val_data_path:   str = field(default=None)
    test_data_path:  str = field(default=None)
    peak_type:       str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir:                   Optional[str] = field(default=None)
    run_name:                    str  = field(default="exp3_hyena_run")
    optim:                       str  = field(default="adamw_torch")
    model_max_length:            int  = field(default=512)
    gradient_accumulation_steps: int  = field(default=1)
    per_device_train_batch_size: int  = field(default=4)
    per_device_eval_batch_size:  int  = field(default=8)
    num_train_epochs:            int  = field(default=3)
    fp16:                        bool = field(default=False)
    logging_steps:               int  = field(default=100)
    save_steps:                  int  = field(default=500)
    eval_steps:                  int  = field(default=500)
    evaluation_strategy:         str  = field(default="steps")
    warmup_steps:                int  = field(default=50)
    weight_decay:                float= field(default=0.01)
    learning_rate:               float= field(default=1e-4)
    save_total_limit:            int  = field(default=3)
    load_best_model_at_end:      bool = field(default=True)
    output_dir:                  str  = field(default="./outputs/exp3_hyena")
    dataloader_pin_memory:       bool = field(default=False)
    dataloader_num_workers:      int  = field(default=0)
    eval_and_save_results:       bool = field(default=True)
    save_model:                  bool = field(default=True)
    seed:                        int  = field(default=42)
    metric_for_best_model:       str  = field(default="auprc")
    save_safetensors:            bool = field(default=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HyenaDNaDataset(Dataset):
    def __init__(self, h5_path: str, tokenizer, split: str = "train",
                 max_length: int = 512, max_samples: Optional[int] = None):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        print(f"Loading {split} data from {h5_path} ...")
        with h5py.File(h5_path, "r") as f:
            self.labels        = f["labels"][:]
            self.sequences     = [s.decode("ascii") for s in f["sequences"][:]]
            self.dnase_signals = f["dnase_signals"][:]

        n_total = len(self.labels)
        n_pos   = int(np.sum(self.labels))
        print(f"  {n_total:,} samples, {n_pos:,} positive ({n_pos/n_total*100:.1f}%)")

        if max_samples and max_samples < n_total:
            idx = np.random.choice(n_total, max_samples, replace=False)
            self.labels        = self.labels[idx]
            self.sequences     = [self.sequences[i] for i in idx]
            self.dnase_signals = self.dnase_signals[idx]

        self.length = len(self.labels)

    def __len__(self): return self.length

    def _process_sequence(self, dna_seq: str):
        enc       = self.tokenizer(dna_seq, truncation=True,
                                   max_length=self.max_length, return_tensors="pt",
                                   return_attention_mask=True)
        token_ids = enc["input_ids"].squeeze(0).tolist()
        tokens    = self.tokenizer.convert_ids_to_tokens(token_ids)

        special = {t for t in [self.tokenizer.cls_token, self.tokenizer.pad_token,
                                self.tokenizer.eos_token, self.tokenizer.sep_token,
                                self.tokenizer.unk_token] if t is not None}

        offsets, pos = [], 0
        for tok in tokens:
            if tok in special:
                offsets.append((0, 0))
            else:
                offsets.append((pos, pos + 1))
                pos += 1

        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).squeeze(0)
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": attn,
                "offset_mapping": offsets}

    def __getitem__(self, idx):
        enc    = self._process_sequence(self.sequences[idx])
        signal = self.dnase_signals[idx]

        dnase_features = []
        for start, end in enc["offset_mapping"]:
            if start == 0 and end == 0:
                dnase_features.append(0.0)
            elif start < len(signal):
                dnase_features.append(float(signal[start]))
            else:
                dnase_features.append(0.0)

        L = len(enc["input_ids"])
        if len(dnase_features) < L:
            dnase_features += [0.0] * (L - len(dnase_features))
        else:
            dnase_features = dnase_features[:L]

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "dnase_signals":  torch.FloatTensor(dnase_features),
            "label":          torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class DataCollatorForHyenaDNase:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch  = {}
        padded = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding="longest", return_tensors="pt",
        )
        batch.update(padded)
        max_len = batch["input_ids"].shape[1]
        batch["dnase_signals"] = torch.stack([
            nn.functional.pad(f["dnase_signals"], (0, max_len - len(f["dnase_signals"])))
            for f in features
        ])
        batch["labels"] = torch.stack([f["label"] for f in features])
        return batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HyenaDNaseFusionClassifier(nn.Module):
    def __init__(self, backbone_model, hidden_dim: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone_model
        if freeze_backbone:
            for p in backbone_model.parameters():
                p.requires_grad = False

        self.dnase_expander = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, hidden_dim, 3, padding=1),
        )
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, input_ids, attention_mask, dnase_signals, labels=None):
        # HyenaDNA does not accept attention_mask
        out = self.backbone(input_ids=input_ids)
        seq_emb = out.hidden_states[-1]

        dnase = dnase_signals.unsqueeze(1) if dnase_signals.dim() == 2 else dnase_signals.permute(0, 2, 1)
        dnase_exp = self.dnase_expander(dnase).permute(0, 2, 1)

        combined = self.feature_fusion(torch.cat([seq_emb, dnase_exp], dim=-1))
        pooled   = combined.mean(dim=1)
        logits   = self.classifier(pooled)

        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=logits.device))(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid = labels != -100
    preds, lbls = predictions[valid], labels[valid]

    if preds.ndim > 1 and preds.shape[1] >= 2:
        scores  = preds[:, 1]
        classes = np.argmax(preds, axis=1)
    else:
        scores  = preds
        classes = (preds > 0.5).astype(int)

    m = {
        "accuracy":             sklearn.metrics.accuracy_score(lbls, classes),
        "f1_weighted":          sklearn.metrics.f1_score(lbls, classes, average="weighted", zero_division=0),
        "f1_class1":            sklearn.metrics.f1_score(lbls, classes, pos_label=1, zero_division=0),
        "precision_class1":     sklearn.metrics.precision_score(lbls, classes, pos_label=1, zero_division=0),
        "recall_class1":        sklearn.metrics.recall_score(lbls,    classes, pos_label=1, zero_division=0),
        "matthews_correlation":  sklearn.metrics.matthews_corrcoef(lbls, classes),
    }
    try:
        prec, rec, _ = sklearn.metrics.precision_recall_curve(lbls, scores, pos_label=1)
        m["auprc"] = sklearn.metrics.auc(rec, prec)
        m["auroc"] = sklearn.metrics.roc_auc_score(lbls, scores)
        m["average_precision"] = average_precision_score(lbls, scores)
    except Exception as e:
        logging.warning("AUC computation failed: %s", e)
        m["auprc"] = m["auroc"] = m["average_precision"] = float("nan")
    return m


def preprocess_logits_for_metrics(logits, _):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.softmax(logits, dim=1)


def compute_metrics(eval_pred):
    return calculate_metric_with_sklearn(*eval_pred)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    if os.path.exists(training_args.output_dir):
        for item in os.listdir(training_args.output_dir):
            p = os.path.join(training_args.output_dir, item)
            if os.path.isdir(p) and item.startswith("checkpoint-"):
                shutil.rmtree(p)

    if not model_args.model_path:
        model_args.model_path = _LOCAL_PATHS.get(model_args.model_name,
                                                  f"LongSafari/hyenadna-{model_args.model_name}-32k-seqlen-hf")

    print(f"Loading backbone: {model_args.model_path}")
    backbone, tokenizer = load_model_and_tokenizer(model_args.model_name, model_args.model_path)

    model = HyenaDNaseFusionClassifier(
        backbone_model=backbone, hidden_dim=model_args.hidden_dim,
        freeze_backbone=model_args.freeze_backbone,
    )

    L = training_args.model_max_length
    train_ds = HyenaDNaDataset(data_args.train_data_path, tokenizer, "train", L)
    val_ds   = HyenaDNaDataset(data_args.val_data_path,   tokenizer, "val",   L)
    test_ds  = HyenaDNaDataset(data_args.test_data_path,  tokenizer, "test",  L)

    trainer = transformers.Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForHyenaDNase(tokenizer),
    )

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

    if training_args.eval_and_save_results:
        results_dir = os.path.join(training_args.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        for split, ds in [("val", val_ds), ("test", test_ds)]:
            res = trainer.evaluate(eval_dataset=ds)
            with open(os.path.join(results_dir, f"{split}_results.json"), "w") as f:
                json.dump(res, f, indent=2)
            print(f"{split} AUPRC: {res.get('eval_auprc', 'N/A'):.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
