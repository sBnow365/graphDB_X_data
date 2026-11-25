"""Extract a label- and date-filtered subgraph and export it (GraphML for Gephi, or JSON).

    python scripts/extract_subgraph.py --start 2025-03-05 --end 2025-03-14 --labels HATE SPAM
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.graph.subgraph_extraction import export_subgraph, extract_subgraph
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.temporal.date_filters import day_range


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="first local day (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last local day, inclusive (YYYY-MM-DD)")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/graphs/subgraph.graphml")
    args = ap.parse_args()
    cfg = get_config()
    dr = day_range(args.start, args.end, cfg.temporal.get("timezone", "UTC"))
    with connect(cfg) as client:
        g = extract_subgraph(client, dr.as_params()["start"], dr.as_params()["end"], args.labels)
    path = export_subgraph(g, args.out)
    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {path}")


if __name__ == "__main__":
    main()
