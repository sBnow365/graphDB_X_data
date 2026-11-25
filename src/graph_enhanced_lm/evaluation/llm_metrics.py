"""Text-to-Cypher and Graph-RAG metrics."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _normalise(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return tuple(_normalise(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _normalise(x)) for k, x in v.items()))
    return v


def result_set_match(gold: list[dict[str, Any]], pred: list[dict[str, Any]],
                     key_columns: list[str] | None = None) -> dict[str, Any]:
    """Order-insensitive comparison of two result sets.

    If ``key_columns`` is given, only those columns are compared (column names in the
    generated query may legitimately differ, so compare by VALUES of key columns).
    """
    def project(rows: list[dict[str, Any]]) -> Counter:
        out = Counter()
        for r in rows:
            if key_columns:
                vals = [r.get(k) for k in key_columns]
                if any(v is None for v in vals):
                    vals = sorted(map(str, r.values()))
            else:
                vals = sorted(map(str, r.values()))
            out[tuple(_normalise(v) for v in vals)] += 1
        return out

    g, p = project(gold), project(pred)
    inter = sum((g & p).values())
    prec = inter / sum(p.values()) if p else (1.0 if not g else 0.0)
    rec = inter / sum(g.values()) if g else (1.0 if not p else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"exact": g == p, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def value_set_match(gold: list[dict[str, Any]], pred: list[dict[str, Any]], gold_col: str) -> dict[str, Any]:
    """Compare the SET of values of one gold column with all values in predicted rows."""
    gv = {str(r.get(gold_col)) for r in gold}
    pv = {str(v) for r in pred for v in r.values()}
    inter = len(gv & pv)
    prec = inter / len(pv) if pv else 0.0
    rec = inter / len(gv) if gv else 1.0
    return {"recall": round(rec, 4), "contained_all": gv <= pv}


def grounding_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Graph-RAG grounding metrics from RAGResult.to_dict() outputs."""
    answered = [r for r in results if r.get("answer")]
    with_unknown = [r for r in answered if r.get("unknown_citations")]
    insufficient = [r for r in answered if r.get("insufficient")]
    return {"questions": len(results), "answered": len(answered),
            "insufficient_evidence": len(insufficient),
            "answers_with_unknown_citations": len(with_unknown),
            "unsupported_citation_rate": round(len(with_unknown) / len(answered), 4) if answered else None}
