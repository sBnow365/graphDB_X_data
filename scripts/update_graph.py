"""Incremental update: ingest only NEW or CHANGED posts and NEW interactions.

    python scripts/update_graph.py --dataset synthetic
Run preprocess_dataset.py (and classify_posts.py) on the new batch first.
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.data.dataset_adapters import GRAPH_ADAPTERS
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.pipeline import connect, ingest
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthetic", choices=sorted(GRAPH_ADAPTERS))
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "update_graph")
    with connect(cfg) as client:
        print(json.dumps(ingest(cfg, args.dataset, client, incremental=True), indent=2, default=str))


if __name__ == "__main__":
    main()
