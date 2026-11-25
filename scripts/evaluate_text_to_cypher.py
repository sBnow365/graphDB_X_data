"""Text-to-Cypher evaluation.

    python scripts/evaluate_text_to_cypher.py --safety-only      # offline, no Neo4j/LLM
    python scripts/evaluate_text_to_cypher.py                     # full benchmark (Neo4j + LLM)
"""
import argparse

import yaml

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import PROJECT_ROOT, get_config
from graph_enhanced_lm.evaluation.benchmark import load_benchmark, run_safety_suite, run_text_to_cypher_benchmark
from graph_enhanced_lm.llm.qwen_client import build_llm_client
from graph_enhanced_lm.llm.query_validation import CypherValidator
from graph_enhanced_lm.llm.text_to_cypher import TextToCypher
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import capture_environment, save_json, utc_now_iso


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety-only", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "eval_t2c")
    t2c_cfg = cfg.llm["text_to_cypher"]
    validator = CypherValidator(int(t2c_cfg["max_rows"]), t2c_cfg.get("allowed_procedures"))
    with (PROJECT_ROOT / "benchmarks/cypher_safety_suite.yaml").open(encoding="utf-8") as fh:
        safety = run_safety_suite(yaml.safe_load(fh)["items"], validator)
    print(f"Safety suite accuracy: {safety['accuracy']}  ({len(safety['items'])} cases)")
    for r in safety["items"]:
        if not r["correct"]:
            print("  MISCLASSIFIED:", r)
    result = {"evaluated_at": utc_now_iso(), "safety_suite": safety, "environment": capture_environment()}
    if not args.safety_only:
        s = cfg.llm_settings()
        llm = build_llm_client(s)
        items = load_benchmark(cfg.evaluation["text_to_cypher_benchmark"])
        drafted = sum(1 for i in items if i.get("status") != "verified")
        if drafted:
            print(f"NOTE: {drafted} benchmark item(s) are not yet human-verified (status != verified).")
        with connect(cfg) as client:
            t2c = TextToCypher(llm, validator, client, int(t2c_cfg["max_retries"]),
                               float(t2c_cfg["query_timeout_seconds"]))
            bench = run_text_to_cypher_benchmark(items, t2c, client)
        bench.update({"llm_backend": s["backend"], "llm_model": s["model"], "unverified_items": drafted})
        result["benchmark"] = bench
        for k in ("n", "syntax_valid_rate", "execution_success_rate", "exact_result_match_rate",
                  "mean_result_f1", "mean_latency_seconds"):
            print(f"{k}: {bench[k]}")
    save_json(result, cfg.path("metrics") / "text_to_cypher_eval.json")
    print(f"Saved {cfg.path('metrics') / 'text_to_cypher_eval.json'}")


if __name__ == "__main__":
    main()
