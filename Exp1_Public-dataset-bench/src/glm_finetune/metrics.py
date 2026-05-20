from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=-1)
    num_labels = probs.shape[1]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    if num_labels == 2:
        metrics["f1_binary"] = float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0))
        metrics["auroc_binary"] = _safe_metric(roc_auc_score, labels, probs[:, 1])
        metrics["auprc_binary"] = _safe_metric(average_precision_score, labels, probs[:, 1])
    else:
        y_true = label_binarize(labels, classes=list(range(num_labels)))
        metrics["auroc_macro"] = _safe_metric(
            roc_auc_score,
            y_true,
            probs,
            average="macro",
            multi_class="ovr",
        )
        metrics["auprc_macro"] = _safe_metric(average_precision_score, y_true, probs, average="macro")
    return metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _safe_metric(fn: Any, *args: Any, **kwargs: Any) -> float:
    try:
        value = float(fn(*args, **kwargs))
    except ValueError:
        return math.nan
    return value
