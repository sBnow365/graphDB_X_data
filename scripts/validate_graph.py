"""Post-ingestion graph checks (counts, duplicates, orphans, placeholders).

    python scripts/validate_graph.py
"""
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.graph.validation import validate_graph
from graph_enhanced_lm.pipeline import connect


def main() -> None:
    cfg = get_config()
    with connect(cfg) as client:
        print(json.dumps(validate_graph(client), indent=2, default=str))


if __name__ == "__main__":
    main()
