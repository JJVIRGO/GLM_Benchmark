#!/usr/bin/env python
# coding: utf-8

"""
Fine-tune BPE-tokenised DNA language models (DNABERT-2, GROVER, GENA-LM)
with optional DNase-seq signal fusion for cross-cell-type TF binding prediction.

The DNase signal is projected to the hidden dimension via a 1D CNN and
concatenated with the last-layer token hidden states before classification.

Usage (via shell wrapper):
  bash scripts/04_finetune_BPE.sh CTCF --model_name GROVER
"""

import inspect
import json
import logging
import os
import shutil
import sys
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
from accelerate import Accelerator

# ── Compatibility shims ──────────────────────────────────────────────────────
if not hasattr(transformers, "EncoderDecoderCache"):
    class EncoderDecoderCache:  # pragma: no cover
        pass
    transformers.EncoderDecoderCache = EncoderDecoderCache

if "dispatch_batches" not in inspect.signature(Accelerator.__init__).parameters:
    _orig_init = Accelerator.__init__
    _orig_params = set(inspect.signature(_orig_init).parameters.keys())

    def _patched_init(self, *args, **kwargs):  # pragma: no cover
        kwargs = {k: v for k, v in kwargs.items() if k in _orig_params}
        return _orig_init(self, *args, **kwargs)

    Accelerator.__init__ = _patched_init
# ─────────────────────────────────────────────────────────────────────────────

# Optional GENA-LM import – loaded from GFM_MODEL_USE env var or sys.path
_GENA_LM_USE = os.environ.get("GFM_MODEL_USE", "")
if _GENA_LM_USE and _GENA_LM_USE not in sys.path:
    sys.path.append(_GENA_LM_USE)

try:
    from GENA_LM.src.gena_lm.modeling_bert import BertForSequenceClassification as GENABertForSequenceClassification
    GENA_LM_AVAILABLE = True
except ImportError:
    GENA_LM_AVAILABLE = False
    GENABertForSequenceClassification = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name: str = field(default="GROVER",
        metadata={"help": "Backbone model name: DNABERT2 | GROVER | GENA_LM_BERT | GENA_LM_BigBird"})
    model_path: str = field(default="",
        metadata={"help": "Local path or HuggingFace model ID for the backbone"})
    hidden_dim: int = field(default=768, metadata={"help": "Backbone hidden dimension"})
    freeze_backbone: bool = field(default=True, metadata={"help": "Freeze backbone weights"})


@dataclass
class DataArguments:
    train_data_path: str  = field(default=None)
    val_data_path:   str  = field(default=None)
    test_data_path:  str  = field(default=None)
    peak_type:       str  = field(default=None)
    max_train_samples: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir:                   Optional[str] = field(default=None)
    run_name:                    str  = field(default="exp3_bpe_run")
    optim:                       str  = field(default="adamw_torch")
    model_max_length:            int  = field(default=1002)
    gradient_accumulation_steps: int  = field(default=1)
    per_device_train_batch_size: int  = field(default=4)
    per_device_eval_batch_size:  int  = field(default=8)
    num_train_epochs:            int  = field(default=3)
    fp16:                        bool = field(default=False)
    logging_steps:               int  = field(default=100)
    save_steps:                  int  = field(default=300)
    eval_steps:                  int  = field(default=300)
    evaluation_strategy:         str  = field(default="steps")
    warmup_steps:                int  = field(default=100)
    weight_decay:                float= field(default=0.01)
    learning_rate:               float= field(default=2e-5)
    save_total_limit:            int  = field(default=3)
    load_best_model_at_end:      bool = field(default=True)
    output_dir:                  str  = field(default="./outputs/exp3_bpe")
    dataloader_pin_memory:       bool = field(default=False)
    eval_and_save_results:       bool = field(default=True)
    save_model:                  bool = field(default=True)
    seed:                        int  = field(default=42)
    metric_for_best_model:       str  = field(default="auprc")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrossCellDNaseDataset(Dataset):
    def __init__(self, h5_path: str, tokenizer, split: str = "train",
                 max_samples: Optional[int] = None):
        self.tokenizer = tokenizer

        print(f"Loading {split} data from {h5_path} ...")
        with h5py.File(h5_path, "r") as f:
            self.chroms       = [c.decode("utf-8") for c in f["chroms"][:]]
            self.starts       = f["starts"][:]
            self.ends         = f["ends"][:]
            self.labels       = f["labels"][:]
            self.sequences    = [s.decode("ascii") for s in f["sequences"][:]]
            self.dnase_signals= f["dnase_signals"][:]

        n_total = len(self.labels)
        n_pos   = int(np.sum(self.labels))
        print(f"  {n_total:,} samples, {n_pos:,} positive ({n_pos/n_total*100:.1f}%)")

        if max_samples and max_samples < n_total:
            idx = np.random.choice(n_total, max_samples, replace=False)
            self.chroms        = [self.chroms[i] for i in idx]
            self.starts        = self.starts[idx]
            self.ends          = self.ends[idx]
            self.labels        = self.labels[idx]
            self.sequences     = [self.sequences[i] for i in idx]
            self.dnase_signals = self.dnase_signals[idx]

        self.length = len(self.labels)

    def __len__(self): return self.length

    def _process_sequence(self, dna_seq: str):
        enc = self.tokenizer(
            dna_seq,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping").squeeze().tolist()
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "offset_mapping": offset_mapping,
        }

    def __getitem__(self, idx):
        enc    = self._process_sequence(self.sequences[idx])
        signal = self.dnase_signals[idx]

        dnase_features = []
        for start, end in enc["offset_mapping"]:
            if start == 0 and end == 0:
                dnase_features.append(0.0)
            else:
                s0 = max(0, min(start, len(signal) - 1))
                s1 = max(s0 + 1, min(end, len(signal)))
                w  = signal[s0:s1]
                dnase_features.append(float(w.max()) if len(w) > 0 else 0.0)

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "dnase_signals":  torch.FloatTensor(dnase_features),
            "label":          torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class DataCollatorForCrossCellDNase:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
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

class CrossCellDNaseClassifier(nn.Module):
    """BPE DNA-LM + DNase 1D-CNN fusion classifier."""

    def __init__(self, model_name: str, model_path: str, hidden_dim: int = 768,
                 freeze_backbone: bool = True):
        super().__init__()
        self.model_name = model_name

        if model_name in ("GENA_LM_BERT", "GENA_LM_BigBird"):
            if not GENA_LM_AVAILABLE:
                raise ImportError("GENA_LM not available. Set GFM_MODEL_USE env var to the repo root.")
            self.LLM_model = GENABertForSequenceClassification.from_pretrained(
                model_path, num_labels=2, output_hidden_states=True)
        else:
            self.LLM_model = AutoModelForSequenceClassification.from_pretrained(
                model_path, trust_remote_code=True, output_hidden_states=True)

        if model_name == "DNABERT2":
            self._disable_dnabert2_triton()

        if freeze_backbone:
            for p in self.LLM_model.parameters():
                p.requires_grad = False

        self.dnase_expander = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
        )
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),
        )

    def _disable_dnabert2_triton(self):
        for module in self.LLM_model.modules():
            if module.__class__.__name__ == "BertUnpadSelfAttention":
                mod_obj = sys.modules.get(module.__class__.__module__)
                if mod_obj and hasattr(mod_obj, "flash_attn_qkvpacked_func"):
                    mod_obj.flash_attn_qkvpacked_func = None

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, dnase_signals, labels=None):
        out = self.LLM_model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True)
        seq_emb = out.hidden_states[-1]  # (B, L, D)

        dnase = dnase_signals.unsqueeze(1) if dnase_signals.dim() == 2 else dnase_signals.permute(0, 2, 1)
        dnase_exp = self.dnase_expander(dnase).permute(0, 2, 1)

        combined = self.feature_fusion(torch.cat([seq_emb, dnase_exp], dim=-1))
        pooled   = self._masked_mean(combined, attention_mask)
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
        "accuracy":           sklearn.metrics.accuracy_score(lbls, classes),
        "f1_weighted":        sklearn.metrics.f1_score(lbls, classes, average="weighted", zero_division=0),
        "f1_macro":           sklearn.metrics.f1_score(lbls, classes, average="macro",    zero_division=0),
        "precision_class1":   sklearn.metrics.precision_score(lbls, classes, pos_label=1, zero_division=0),
        "recall_class1":      sklearn.metrics.recall_score(lbls,    classes, pos_label=1, zero_division=0),
        "f1_class1":          sklearn.metrics.f1_score(lbls,        classes, pos_label=1, zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(lbls, classes),
    }
    try:
        prec, rec, _ = sklearn.metrics.precision_recall_curve(lbls, scores, pos_label=1)
        m["auprc"]             = sklearn.metrics.auc(rec, prec)
        m["auroc"]             = sklearn.metrics.roc_auc_score(lbls, scores)
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

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # Clean stale checkpoints
    if os.path.exists(training_args.output_dir):
        for item in os.listdir(training_args.output_dir):
            p = os.path.join(training_args.output_dir, item)
            if os.path.isdir(p) and item.startswith("checkpoint-"):
                shutil.rmtree(p)

    print(f"Loading tokenizer from {model_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_path, trust_remote_code=True,
        model_max_length=training_args.model_max_length,
    )

    train_ds = CrossCellDNaseDataset(data_args.train_data_path, tokenizer, "train", data_args.max_train_samples)
    val_ds   = CrossCellDNaseDataset(data_args.val_data_path,   tokenizer, "val")
    test_ds  = CrossCellDNaseDataset(data_args.test_data_path,  tokenizer, "test")

    model = CrossCellDNaseClassifier(
        model_name=model_args.model_name, model_path=model_args.model_path,
        hidden_dim=model_args.hidden_dim, freeze_backbone=model_args.freeze_backbone,
    )

    trainer = transformers.Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForCrossCellDNase(tokenizer),
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
    train()
