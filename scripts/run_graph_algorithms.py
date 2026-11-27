"""Project the interaction graph, run Louvain (+ degree, PageRank, optional betweenness),
write results to User nodes, and materialise Community nodes.

    python scripts/run_graph_algorithms.py
    python scripts/run_graph_algorithms.py --betweenness --mode stream
"""
import argparse
import json

import pandas as pd

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.evaluation.graph_metrics import community_size_distribution, ground_truth_recovery
from graph_enhanced_lm.gds import centrality
from graph_enhanced_lm.gds.availability import gds_info
from graph_enhanced_lm.gds.community_detection import louvain_stream, louvain_write, materialize_communities
from graph_enhanced_lm.gds.memory_estimation import estimate
from graph_enhanced_lm.gds.projections import project_interaction_graph
from graph_enhanced_lm.pipeline import connect, processed_dir
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import new_run_id, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["write", "stream"], default="write",
                    help="stream = inspect only (nothing stored); write = store on User nodes")
    ap.add_argument("--betweenness", action="store_true", help="also compute (sampled) betweenness")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "gds")
    g, pcfg = cfg.gds, cfg.gds["projection"] #g to get GDS settings,pcfg sub dictionary of g
    run_id = new_run_id("louvain")
    report = {"run_id": run_id}
    with connect(cfg) as client:
        report["gds"] = gds_info(client)
        proj = project_interaction_graph(client, pcfg["name"], pcfg["weights"],
                                         include_isolated=pcfg.get("include_isolated_users", False), #not including nodes with no interactions
                                         max_relationships=pcfg.get("max_relationships_guard"))
        report["projection"] = proj
        report["memory_estimate_louvain"] = estimate(client, pcfg["name"], "louvain")
        if args.mode == "stream":
            df = louvain_stream(client, pcfg["name"], g["louvain"])
            print(df.groupby("community").size().sort_values(ascending=False).head(20).to_string())
            return
        report["louvain"] = louvain_write(client, pcfg["name"], g["louvain"])
        report["degree"] = centrality.degree(client, pcfg["name"], g["degree"])
        report["pagerank"] = centrality.pagerank(client, pcfg["name"], g["pagerank"])
        if args.betweenness and report["gds"].get("has_betweenness"):
            report["betweenness"] = centrality.betweenness(client, pcfg["name"], g["betweenness"],
                                                           int(proj.get("node_count", 0)))
        prop = g["louvain"]["write_property"]
        report["materialized"] = materialize_communities(client, prop, run_id, pcfg["name"])
        assign = pd.DataFrame(client.run_read(
            "MATCH (u:User) WHERE u[$p] IS NOT NULL RETURN u.user_id AS user_id, u[$p] AS community", {"p": prop}))
        assign.to_csv(cfg.path("metrics") / "community_assignments.csv", index=False)
        report["size_distribution"] = community_size_distribution(assign)
        users_csv = processed_dir(cfg, cfg.dataset["active_graph_dataset"]) / "users.csv"
        if cfg.dataset["active_graph_dataset"] == "synthetic" and users_csv.exists(): # not relevant to russian trolls
            truth = pd.read_csv(users_csv, dtype=str)
            report["synthetic_ground_truth_recovery"] = ground_truth_recovery(assign, truth)
    save_json(report, cfg.path("metrics") / f"graph_algorithms_{run_id}.json")
    save_json(report, cfg.path("metrics") / "graph_algorithms_latest.json")
    print(json.dumps({k: report[k] for k in ("louvain", "size_distribution") if k in report}, indent=2, default=str))
    if "synthetic_ground_truth_recovery" in report:
        print("Synthetic ground-truth recovery:", report["synthetic_ground_truth_recovery"])


if __name__ == "__main__":
    main()
