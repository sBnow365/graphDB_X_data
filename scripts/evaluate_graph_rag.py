"""Graph-RAG evaluation.

    python scripts/evaluate_graph_rag.py --routing-only   # offline: intent routing accuracy
    python scripts/evaluate_graph_rag.py                  # full: retrieval + answers (Neo4j + LLM)
    python scripts/evaluate_graph_rag.py --no-llm         # retrieval only (Neo4j)
The full run saves every answer so you can MANUALLY score correctness/grounding.
"""
import argparse

import yaml

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import PROJECT_ROOT, get_config
from graph_enhanced_lm.evaluation.llm_metrics import grounding_stats
from graph_enhanced_lm.llm.qwen_client import build_llm_client
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.retrieval.graph_rag import build_graph_rag
from graph_enhanced_lm.retrieval.intent_parser import parse_question
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import save_json, utc_now_iso


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing-only", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "eval_rag")
    with (PROJECT_ROOT / cfg.evaluation["graph_rag_benchmark"]).open(encoding="utf-8") as fh:
        items = yaml.safe_load(fh)["items"]
    routing = []
    for it in items:
        intent = parse_question(it["question"], cfg.temporal["election_date"], cfg.temporal["timezone"])
        labels_ok = set(it.get("expected_labels", [])) <= set(intent.labels)
        routing.append({"id": it["id"], "expected": it.get("expected_intent"), "got": intent.intent,
                        "intent_correct": intent.intent == it.get("expected_intent"), "labels_ok": labels_ok})
    acc = sum(r["intent_correct"] for r in routing) / len(routing)
    print(f"Intent routing accuracy: {acc:.2f}")
    for r in routing:
        if not r["intent_correct"]:
            print("  mismatch:", r)
    result = {"evaluated_at": utc_now_iso(), "routing_accuracy": round(acc, 4), "routing": routing}
    if not args.routing_only:
        llm = None if args.no_llm else build_llm_client(cfg.llm_settings())
        outputs = []
        with connect(cfg) as client:
            rag = build_graph_rag(client, llm, cfg)
            for it in items:
                d = rag.ask(it["question"], generate_answer=llm is not None).to_dict()
                d["id"] = it["id"]
                d["manual_scores"] = {"answer_correct": None, "fully_grounded": None, "evidence_relevant": None}
                outputs.append(d)
        result["outputs"] = outputs
        result["automatic_grounding"] = grounding_stats(outputs)
        print(result["automatic_grounding"])
        print("Fill in manual_scores in the saved JSON to compute correctness and grounding rates.")
    save_json(result, cfg.path("metrics") / "graph_rag_eval.json")
    print(f"Saved {cfg.path('metrics') / 'graph_rag_eval.json'}")


if __name__ == "__main__":
    main()
