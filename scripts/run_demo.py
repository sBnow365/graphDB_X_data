"""END-TO-END demo on SYNTHETIC data.

    python scripts/run_demo.py --offline     # steps 1-4 only: no Neo4j, no LLM
    python scripts/run_demo.py --no-llm      # + Neo4j, GDS, community analysis, evidence
    python scripts/run_demo.py               # + Qwen answers (LLM endpoint configured)
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.data.loaders import read_table
from graph_enhanced_lm.evaluation.graph_metrics import (
    graph_summary, ground_truth_recovery, interaction_graph_from_tables, modularity_of, offline_louvain,
)
from graph_enhanced_lm.pipeline import (
    classify_dataset, connect, generate_synthetic, ingest, load_processed, merge_predictions,
    preprocess_graph_dataset, processed_dir,
)
from graph_enhanced_lm.temporal.election_windows import build_windows
from graph_enhanced_lm.temporal.trend_analysis import CAUSALITY_NOTE, window_comparison_df
from graph_enhanced_lm.utils.logging_config import setup_logging

QUESTIONS = [
    "Which users posted hate-labelled content during the ten days before the election?",
    "Which communities contain a high proportion of hate-labelled posts?",
    "Which users repeatedly amplified the same posts?",
    "How did the network change before, during, and after the election?",
]


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\nSTEP {n}: {title}\n{'=' * 70}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    cfg.dataset["active_graph_dataset"] = "synthetic"
    setup_logging("WARNING", cfg.path("logs"), "demo")

    step(1, "Generate SYNTHETIC dataset (fixed seed)")
    ds = generate_synthetic(cfg)
    print(ds.summary())
    step(2, "Preprocess + validate")
    clean, report = preprocess_graph_dataset(cfg, "synthetic")
    print(report.pretty())
    step(3, "Classify posts (source labels = synthetic ground truth)")
    preds = classify_dataset(cfg, "synthetic", "source_labels")
    print(preds["primary_label"].value_counts().to_string())
    step(4, "Offline graph + temporal analysis (NetworkX Louvain, pandas windows)")
    g = interaction_graph_from_tables(clean.posts, clean.interactions, cfg.gds["projection"]["weights"])
    assign = offline_louvain(g, int(cfg.project["random_seed"]))
    print("graph:", graph_summary(g), " modularity:", modularity_of(g, assign))
    print("recovery of planted communities:", ground_truth_recovery(assign, clean.users))
    posts = merge_predictions(load_processed(cfg, "synthetic"), read_table(processed_dir(cfg, "synthetic") / "predictions.csv"))
    cmp_ = window_comparison_df(posts, build_windows(cfg.temporal))
    print(cmp_[["window", "posts", "posts_per_day", "hate_share", "spam_share"]].to_string(index=False))
    print(CAUSALITY_NOTE)
    if args.offline:
        print("\nOffline demo complete. Start Neo4j and re-run without --offline for the graph steps.")
        return

    from graph_enhanced_lm.gds.community_detection import louvain_write, materialize_communities
    from graph_enhanced_lm.gds.centrality import pagerank
    from graph_enhanced_lm.gds.projections import project_interaction_graph
    from graph_enhanced_lm.graph.constraints import create_constraints_and_indexes
    from graph_enhanced_lm.llm.qwen_client import build_llm_client
    from graph_enhanced_lm.retrieval.graph_rag import build_graph_rag
    from graph_enhanced_lm.utils.reproducibility import new_run_id

    with connect(cfg) as client:
        step(5, "Neo4j: constraints + idempotent ingestion")
        print(client.verify())
        create_constraints_and_indexes(client, cfg.neo4j["constraints_file"], cfg.neo4j["indexes_file"])
        print(json.dumps(ingest(cfg, "synthetic", client), default=str)[:600], "...")
        step(6, "GDS: projection -> Louvain (write) -> PageRank")
        p = cfg.gds["projection"]
        print(project_interaction_graph(client, p["name"], p["weights"]))
        print(louvain_write(client, p["name"], cfg.gds["louvain"]))
        print(pagerank(client, p["name"], cfg.gds["pagerank"]))
        print(materialize_communities(client, cfg.gds["louvain"]["write_property"], new_run_id("louvain"), p["name"]))
        step(7, "Graph RAG questions" + (" (evidence only)" if args.no_llm else " (with Qwen)"))
        llm = None if args.no_llm else build_llm_client(cfg.llm_settings())
        rag = build_graph_rag(client, llm, cfg)
        for q in QUESTIONS:
            d = rag.ask(q).to_dict()
            print(f"\nQ: {q}\n  method={d['method']}  evidence={len(d['evidence'])}")
            for e in d["evidence"][:3]:
                print(f"   [{e['id']}] {e['content'][:140]}")
            print(f"  A: {(d['answer'] or '')[:600]}")
    print("\nDemo complete. All results above are computed on SYNTHETIC data.")


if __name__ == "__main__":
    main()
