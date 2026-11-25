"""Label processed posts with the configured classifier -> predictions.csv.

    python scripts/classify_posts.py --dataset synthetic                   (source labels)
    python scripts/classify_posts.py --dataset twibot22 --backend transformer
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.data.dataset_adapters import GRAPH_ADAPTERS
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.pipeline import classify_dataset
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic", choices=sorted(GRAPH_ADAPTERS))
    ap.add_argument("--backend", choices=["source_labels", "transformer"])
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "classify")
    df = classify_dataset(cfg, args.dataset, args.backend) #resulting dataset after classification
    print(df["primary_label"].value_counts().to_string())
    print(f"\nlabel sources: {df['label_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
