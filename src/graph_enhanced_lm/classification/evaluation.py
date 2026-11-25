"""Classifier evaluation: accuracy, precision, recall, F1, per-class, confusion matrix.

Only call this with REAL predictions produced by a model that was actually run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


def evaluate_single_label(y_true: list[str], y_pred: list[str],
                          labels: list[str] | None = None) -> dict[str, Any]:
    """Evaluate primary-label predictions (single-label view)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    if not y_true:
        raise ValueError("Nothing to evaluate")
    labels = labels or sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    support = dict(zip(labels, map(int, s)))
    total = sum(support.values()) or 1
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {lab: {"precision": float(pi), "recall": float(ri), "f1": float(fi),
                            "support": int(si)}
                      for lab, pi, ri, fi, si in zip(labels, p, r, f, s)},
        "class_distribution": {k: round(v / total, 4) for k, v in support.items()},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report_text": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def evaluate_multi_label(true_sets: list[list[str]], pred_sets: list[list[str]],
                         labels: list[str]) -> dict[str, Any]:
    """Per-label binary metrics for the multi-label view (e.g. COMMUNITY_TARGETED)."""
    out: dict[str, Any] = {}
    for lab in labels:
        yt = [lab in s for s in true_sets]
        yp = [lab in s for s in pred_sets]
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        out[lab] = {"precision": float(p), "recall": float(r), "f1": float(f),
                    "positives": int(sum(yt))}
    out["macro_f1"] = float(np.mean([v["f1"] for v in out.values() if isinstance(v, dict)]))
    return out
