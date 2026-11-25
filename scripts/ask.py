"""Ask a natural-language question (Graph RAG).

    python scripts/ask.py "Which users posted hate-labelled content during the ten days before the election?"
    python scripts/ask.py --no-llm "Which communities contain a high proportion of hate-labelled posts?"
    python scripts/ask.py --t2c-only "How many spam posts use the hashtag deal?"
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.llm.qwen_client import build_llm_client
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.retrieval.graph_rag import build_graph_rag
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import save_json, new_run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--no-llm", action="store_true", help="retrieve evidence only (no Qwen)")
    ap.add_argument("--t2c-only", action="store_true", help="skip templates; force Text-to-Cypher")
    ap.add_argument("--json", action="store_true", help="print the full JSON result")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "ask")
    llm = None if args.no_llm else build_llm_client(cfg.llm_settings())
    with connect(cfg) as client:
        rag = build_graph_rag(client, llm, cfg)
        if args.t2c_only:
            rag.retriever.retrieve = lambda intent: None  # type: ignore[assignment]
        res = rag.ask(args.question)
    d = res.to_dict()
    save_json(d, cfg.path("logs") / f"{new_run_id('ask')}.json")
    if args.json:
        print(json.dumps(d, indent=2, default=str, ensure_ascii=False))
        return
    print(f"Intent: {d['intent']}  labels={d['labels']}  range={d['date_range']}")
    print(f"Method: {d['method']}")
    if d["cypher"]:
        print(f"Cypher:\n{d['cypher']}\nParams: {d['params']}")
    print(f"\nEvidence ({len(d['evidence'])} items):")
    for e in d["evidence"][:15]:
        print(f"  [{e['id']}] ({e['kind']}) {e['content'][:160]}")
    for n in d["evidence_notes"]:
        print(f"  NOTE: {n}")
    if d["errors"]:
        print(f"\nErrors: {d['errors']}")
    print(f"\nAnswer:\n{d['answer']}")
    if d["unknown_citations"]:
        print(f"WARNING: answer cites evidence that does not exist: {d['unknown_citations']}")


if __name__ == "__main__":
    main()
