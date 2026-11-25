"""Inspect any CSV/TSV/JSON/JSONL/XLSX/Parquet file: columns, types, missing values, sample.

    python scripts/inspect_dataset.py --path data/raw/hatexplain/dataset.json
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.data.loaders import inspect_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()
    info = inspect_file(args.path, args.rows)
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str)[:20000])


if __name__ == "__main__":
    main()
