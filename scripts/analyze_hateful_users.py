"""Hateful Users dataset: homophily (offline) and community concentration (Neo4j).

    python scripts/analyze_hateful_users.py --offline      # from processed CSVs, no database
    python scripts/analyze_hateful_users.py                # + Louvain concentration (run GDS first)
Uses HUMAN user-level annotations. Results describe structure, not intent.
"""
import argparse
import json

import pandas as pd

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.evaluation.user_label_analysis import concentration, homophily_table
from graph_enhanced_lm.graph.cypher_queries import get_template
from graph_enhanced_lm.pipeline import connect, load_processed
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import save_json, utc_now_iso


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="CSV-only homophily analysis")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "hateful_users")
    ds = load_processed(cfg, "hateful_users")
    result = {"generated_at": utc_now_iso(), "dataset": "hateful_users",
              "label_source": "human crowdsourced user-level annotation (Ribeiro et al. 2018)",
              "label_counts": ds.users["user_hate_label"].value_counts(dropna=False).to_dict(),
              "homophily": homophily_table(ds.users, ds.user_edges)}
    if not args.offline:
        prop = cfg.gds["louvain"]["write_property"]
        with connect(cfg) as client:
            stats = pd.DataFrame(client.run_read(get_template("community_user_label_stats").cypher,
                                                 {"community_property": prop, "min_users": 1}))
        result["concentration"] = concentration(stats, args.top_k)
        if stats.empty:
            result["concentration"]["note"] = "No communities: run scripts/run_graph_algorithms.py first."
    out = cfg.path("metrics") / "hateful_users_analysis.json"
    save_json(result, out)
    print(json.dumps(result, indent=2, default=str)[:4000])
    print(f"\nSaved {out}. Structure only: not evidence of coordination or intent.")


if __name__ == "__main__":
    main()
