"""Text-to-Cypher benchmark runner.

Each benchmark item has a natural-language question, parameters, and a GOLD Cypher
query written and checked by hand. The gold query's results are the reference; the
generated query is scored on validity, safety, execution and result agreement.
"""
from __future__ import annotations

import time
from typing import Any

import yaml

from graph_enhanced_lm.config import PROJECT_ROOT
from graph_enhanced_lm.evaluation.llm_metrics import result_set_match, value_set_match
from graph_enhanced_lm.llm.query_validation import CypherValidator
from graph_enhanced_lm.llm.text_to_cypher import TextToCypher


def load_benchmark(path: str) -> list[dict[str, Any]]:
    with (PROJECT_ROOT / path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["items"]


def run_safety_suite(items: list[dict[str, Any]], validator: CypherValidator) -> dict[str, Any]:
    """Items with ``unsafe: true`` must be REJECTED; others must pass."""
    rows = []
    for it in items:
        res = validator.validate(it["cypher"], it.get("params", {}))
        rows.append({"id": it["id"], "expected_unsafe": bool(it.get("unsafe")),
                     "rejected": not res.ok, "correct": (not res.ok) == bool(it.get("unsafe")),
                     "errors": res.errors})
    return {"items": rows, "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None}


def run_text_to_cypher_benchmark(items: list[dict[str, Any]], t2c: TextToCypher,
                                 client: Any) -> dict[str, Any]:
    per_item = []
    for it in items:
        params = dict(it.get("params", {}))
        params.setdefault("limit", t2c.validator.max_rows)
        gold_rows = client.run_read(it["gold_cypher"], params, max_rows=t2c.validator.max_rows)
        t0 = time.perf_counter()
        res = t2c.run(it["question"], params)
        latency = time.perf_counter() - t0
        rec: dict[str, Any] = {"id": it["id"], "question": it["question"], "category": it.get("category"),
                               "generated_cypher": res.cypher, "valid": res.valid, "executed": res.executed,
                               "attempts": res.attempts, "latency_seconds": round(latency, 3),
                               "errors": res.errors, "gold_rows": len(gold_rows), "pred_rows": len(res.rows)}
        if res.executed:
            rec["result_match"] = result_set_match(gold_rows, res.rows, it.get("key_columns"))
            if it.get("key_columns"):
                rec["key_value_recall"] = value_set_match(gold_rows, res.rows, it["key_columns"][0])
        per_item.append(rec)
    n = len(per_item) or 1
    return {
        "n": len(per_item),
        "syntax_valid_rate": round(sum(r["valid"] for r in per_item) / n, 4),
        "execution_success_rate": round(sum(r["executed"] for r in per_item) / n, 4),
        "exact_result_match_rate": round(sum(r.get("result_match", {}).get("exact", False) for r in per_item) / n, 4),
        "mean_result_f1": round(sum(r.get("result_match", {}).get("f1", 0.0) for r in per_item) / n, 4),
        "mean_latency_seconds": round(sum(r["latency_seconds"] for r in per_item) / n, 3),
        "items": per_item,
    }
