"""Generate the reproducible SYNTHETIC dataset (no internet, no Neo4j needed).

    python scripts/generate_synthetic_data.py            # uses configs/synthetic.yaml
    python scripts/generate_synthetic_data.py --seed 7   # different but reproducible
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.pipeline import generate_synthetic
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="override the random seed")
    ap.add_argument("--n-users", type=int, help="override number of users")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "synthetic")
    if args.seed is not None:
        cfg.synthetic["seed"] = args.seed
    if args.n_users is not None:
        cfg.synthetic["n_users"] = args.n_users
    ds = generate_synthetic(cfg)
    s = ds.summary()
    print(f"\nSYNTHETIC dataset written to {cfg.path('synthetic')}")
    print(f"  users={s['users']}  posts={s['posts']}  interactions={s['interactions']}")
    print("  Every record has synthetic=True. Do not present results as real-world evidence.")


if __name__ == "__main__":
    main()
