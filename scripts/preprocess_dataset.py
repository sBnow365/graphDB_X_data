"""Convert a dataset into the canonical schema, clean it and validate it.

    python scripts/preprocess_dataset.py --dataset synthetic
    python scripts/preprocess_dataset.py --dataset hatexplain   (classifier table + splits)
    python scripts/preprocess_dataset.py --dataset twibot22
Raw files are never modified; outputs go to data/processed/<dataset>/.
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.data.dataset_adapters import GRAPH_ADAPTERS
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.pipeline import (
    CLASSIFIER_DATASETS, preprocess_classification_dataset, preprocess_graph_dataset, processed_dir,
)
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(GRAPH_ADAPTERS) + ["hatexplain", "hasoc"])
    args = ap.parse_args()  #inorder to read args from terminal
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "preprocess")
    if args.dataset in CLASSIFIER_DATASETS:
        _, report = preprocess_classification_dataset(cfg, args.dataset)
    else:
        _, report = preprocess_graph_dataset(cfg, args.dataset)
    print(report.pretty())
    print(f"\nOutputs in {processed_dir(cfg, args.dataset)}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
