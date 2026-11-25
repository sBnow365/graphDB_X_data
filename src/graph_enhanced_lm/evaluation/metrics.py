"""Shared metric helpers (re-exports classifier metrics for convenience)."""
from __future__ import annotations

from typing import Any

from graph_enhanced_lm.classification.evaluation import evaluate_multi_label, evaluate_single_label

__all__ = ["evaluate_single_label", "evaluate_multi_label", "rate"]


def rate(numerator: int, denominator: int) -> float | None:
    """Safe ratio; None when the denominator is zero (never invent a value)."""
    return round(numerator / denominator, 4) if denominator else None


def summarize_latencies(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {"n": len(s), "mean": round(sum(s) / len(s), 3), "median": s[len(s) // 2],
            "max": s[-1]}
