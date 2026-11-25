"""Create / list / drop GDS in-memory projections.

    python scripts/create_gds_projection.py                       # full interaction graph
    python scripts/create_gds_projection.py --name pre10 --start 2025-03-05 --end 2025-03-14
    python scripts/create_gds_projection.py --name hate_only --labels HATE
    python scripts/create_gds_projection.py --list
    python scripts/create_gds_projection.py --drop pre10
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.gds.projections import drop_graph, list_graphs, project_interaction_graph
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.temporal.date_filters import day_range
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--start", help="first local day YYYY-MM-DD")
    ap.add_argument("--end", help="last local day YYYY-MM-DD (inclusive)")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--drop")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "gds_projection")
    pcfg = cfg.gds["projection"]
    with connect(cfg) as client:
        if args.list:
            print(json.dumps(list_graphs(client), indent=2, default=str))
            return
        if args.drop:
            print("dropped" if drop_graph(client, args.drop) else "not found")
            return
        start = end = None
        if args.start and args.end:
            dr = day_range(args.start, args.end, cfg.temporal.get("timezone", "UTC"))
            start, end = dr.as_params()["start"], dr.as_params()["end"]
        res = project_interaction_graph(client, args.name or pcfg["name"], pcfg["weights"], start, end,
                                        args.labels, pcfg.get("include_isolated_users", False),
                                        max_relationships=pcfg.get("max_relationships_guard"))
        print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
