"""Load processed users/posts/interactions/predictions into Neo4j (idempotent MERGE).

    python scripts/ingest_graph.py --dataset synthetic
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.data.dataset_adapters import GRAPH_ADAPTERS
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.graph.validation import validate_graph
from graph_enhanced_lm.pipeline import connect, ingest
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthetic", choices=sorted(GRAPH_ADAPTERS))
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "ingest")
    with connect(cfg) as client:
        report = ingest(cfg, args.dataset, client)
        print(json.dumps(report, indent=2, default=str))
        v = validate_graph(client)
        print("\nGraph validation:", "OK" if v["ok"] else v["problems"])
        for row in v["node_counts"]:
            print(f"  {row['label']:<15} {row['n']}")


if __name__ == "__main__":
    main()
