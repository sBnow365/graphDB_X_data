"""Re-validate an already processed dataset (canonical graph tables).

    python scripts/validate_dataset.py --dataset synthetic
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.data.dataset_adapters import GRAPH_ADAPTERS
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.data.validators import validate_canonical
from graph_enhanced_lm.pipeline import load_processed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthetic", choices=sorted(GRAPH_ADAPTERS))
    args = ap.parse_args()
    cfg = get_config()
    ds = load_processed(cfg, args.dataset)
    req = {k: v["required"] for k, v in cfg.dataset["canonical_tables"].items() if k != "classification_posts"}
    rep = validate_canonical(ds, req, cfg.dataset["interaction_types"])
    print(rep.pretty())
    raise SystemExit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
