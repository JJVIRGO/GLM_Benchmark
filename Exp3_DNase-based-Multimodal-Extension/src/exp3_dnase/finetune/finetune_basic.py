#!/usr/bin/env python
# coding: utf-8

"""
From-scratch CNN-LSTM baseline for DNase-seq + DNA sequence TF binding prediction.

Architecture:
  DNA sequence  → one-hot → Conv1D(240) → MaxPool → LSTM(32) → Linear → hidden
  DNase signal  → Conv1D expander (1 → 64 → 128 → hidden)   → GlobalAvgPool → hidden
  concat → LayerNorm + Linear + GELU + Dropout → Dense(1024) → Dense(512) → sigmoid
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name:   str   = field(default="basic-cnn-lstm")
    hidden_dim:   int   = field(default=256)
    freeze_backbone: bool = field(default=False)
    conv_filters: int   = field(default=240)
    filter_size:  int   = field(default=20)
    strides:      int   = field(default=15)
    pool_size:    int   = field(default=15)
    lstm_nodes:   int   = field(default=32)
    dense1_nodes: int   = field(default=1024)
    dense2_nodes: int   = field(default=512)
    dropout:      float = field(default=0.5)


@dataclass
class DataArguments:
    train_data_path: str  = field(default=None)
    val_data_path:   str  = field(default=None)
    test_data_path:  str  = field(default=None)
    peak_type:       str  = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir:                   Optional[str] = field(default=None)
    run_name:                    str  = field(default="exp3_basic_run")
    optim:                       str  = field(default="adamw_torch")
    model_max_length:            int  = field(default=500)
    gradient_accumulation_steps: int  = field(default=1)
    per_device_train_batch_size: int  = field(default=16)
    per_device_eval_batch_size:  int  = field(default=32)
    num_train_epochs:            int  = field(default=15)
    fp16:                        bool = field(default=False)
    logging_steps:               int  = field(default=100)
    save_steps:                  int  = field(default=500)
    eval_steps:                  int  = field(default=500)
    evaluation_strategy:         str  = field(default="steps")
    warmup_steps:                int  = field(default=50)
    weight_decay:                float= field(default=0.01)
    learning_rate:               float= field(default=1e-3)
    save_total_limit:            int  = field(default=3)
    load_best_model_at_end:      bool = field(default=True)
    output_dir:                  str  = field(default="./outputs/exp3_basic")
    dataloader_pin_memory:       bool = field(default=False)
    dataloader_num_workers:      int  = field(default=1)
    eval_and_save_results:       bool = field(default=True)
    save_model:                  bool = field(default=True)
    seed:                        int  = field(default=42)
    metric_for_best_model:       str  = field(default="auprc")
    save_safetensors:            bool = field(default=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BasicDNaDataset(Dataset):
    _MAPPING = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}

    def __init__(self, h5_path: str, split: str = "train", max_length: int = 500,
                 max_samples: Optional[int] = None):
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

    def _to_onehot(self, seq: str) -> torch.Tensor:
        oh = [self._MAPPING.get(b, [0,0,0,0]) for b in seq.upper()]
        return torch.FloatTensor(oh)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        signal = self.dnase_signals[idx]
        oh     = self._to_onehot(seq)
        L      = self.max_length

        if len(oh) > L:
            oh, signal = oh[:L], signal[:L]
        elif len(oh) < L:
            pad = L - len(oh)
            oh     = torch.cat([oh, torch.zeros(pad, 4)])
            signal = np.concatenate([signal, np.zeros(pad)])

        return {
            "seq_onehot":    oh,
            "dnase_signals": torch.FloatTensor(signal),
            "label":         torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class DataCollatorForBasicDNase:
    def __call__(self, features):
        return {
            "seq_onehot":    torch.stack([f["seq_onehot"]    for f in features]),
            "dnase_signals": torch.stack([f["dnase_signals"] for f in features]),
            "labels":        torch.stack([f["label"]         for f in features]),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicDNaseFusionClassifier(nn.Module):
    def __init__(self, hidden_dim=256, conv_filters=240, filter_size=20,
                 strides=15, pool_size=15, lstm_nodes=32,
                 dense1_nodes=1024, dense2_nodes=512, dropout=0.5, **_):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.seq_conv = nn.Sequential(
            nn.Conv1d(4, conv_filters, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=strides),
        )
        self.seq_lstm       = nn.LSTM(conv_filters, lstm_nodes, batch_first=True)
        self.seq_projection = nn.Linear(lstm_nodes, hidden_dim)

        self.dnase_expander = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, hidden_dim, 3, padding=1),
        )
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dense1_nodes), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense1_nodes, dense2_nodes), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense2_nodes, 1), nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq_onehot, dnase_signals, labels=None):
        # seq branch
        x = seq_onehot.permute(0, 2, 1)          # (B, 4, L)
        x = self.seq_conv(x)                      # (B, F, L')
        x, _ = self.seq_lstm(x.permute(0, 2, 1)) # (B, L', lstm)
        seq_emb = self.seq_projection(x[:, -1, :])# (B, H)

        # dnase branch
        d = dnase_signals.unsqueeze(1)            # (B, 1, L)
        d = self.dnase_expander(d)                # (B, H, L)
        d_pooled = d.mean(dim=-1)                 # (B, H)

        fused  = self.feature_fusion(torch.cat([seq_emb, d_pooled], dim=-1))
        logits = self.classifier(fused).squeeze(-1)

        if labels is not None:
            loss = nn.BCELoss()(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid = labels != -100
    preds, lbls = predictions[valid], labels[valid]

    scores  = preds[:, 1] if preds.ndim > 1 and preds.shape[1] >= 2 else preds
    classes = np.argmax(preds, axis=1) if preds.ndim > 1 else (preds > 0.5).astype(int)

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
    return logits


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

    model = BasicDNaseFusionClassifier(
        hidden_dim=model_args.hidden_dim, conv_filters=model_args.conv_filters,
        filter_size=model_args.filter_size, strides=model_args.strides,
        pool_size=model_args.pool_size, lstm_nodes=model_args.lstm_nodes,
        dense1_nodes=model_args.dense1_nodes, dense2_nodes=model_args.dense2_nodes,
        dropout=model_args.dropout,
    )

    L = training_args.model_max_length
    train_ds = BasicDNaDataset(data_args.train_data_path, "train", L)
    val_ds   = BasicDNaDataset(data_args.val_data_path,   "val",   L)
    test_ds  = BasicDNaDataset(data_args.test_data_path,  "test",  L)

    trainer = transformers.Trainer(
        model=model, tokenizer=None, args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForBasicDNase(),
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
