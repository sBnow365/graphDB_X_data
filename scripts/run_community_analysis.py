"""Community statistics + transparent investigation-priority ranking.

    python scripts/run_community_analysis.py
Outputs outputs/metrics/community_ranking.csv and .md
"""
import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.gds.statistics import (
    INTERPRETATION, add_pre_election_growth, fetch_community_stats, score_communities,
)
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.temporal.election_windows import build_windows, nearest_pre_windows
from graph_enhanced_lm.temporal.trend_analysis import community_window_activity_neo4j
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import df_to_markdown


def main() -> None:
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "community_analysis")
    ca, prop = cfg.gds["community_analysis"], cfg.gds["louvain"]["write_property"]
    with connect(cfg) as client:
        stats = fetch_community_stats(client, prop, int(ca["min_users"]))
        if stats.empty:
            raise SystemExit("No communities found. Run scripts/run_graph_algorithms.py first.")
        windows = build_windows(cfg.temporal)
        pair = nearest_pre_windows(windows)
        if pair:
            far, near = pair
            activity = community_window_activity_neo4j(client, [far, near], prop)
            stats = add_pre_election_growth(stats, activity, near.name, far.name, near.days, far.days)
    ranked = score_communities(stats, ca["priority_weights"])
    out = cfg.path("metrics") / "community_ranking.csv"
    ranked.to_csv(out, index=False)
    cols = [c for c in ["priority_rank", "community", "users", "posts", "hate_share", "spam_share",
                        "offensive_share", "pre_election_growth", "internal_ratio", "priority_score",
                        "designation"] if c in ranked.columns]
    md = cfg.path("metrics") / "community_ranking.md"
    md.write_text("# Community ranking (for human review)\n\n" + INTERPRETATION + "\n\nWeights: "
                  + str(ca["priority_weights"]) + "\n\n" + df_to_markdown(ranked[cols]), encoding="utf-8")
    print(ranked[cols].head(15).to_string(index=False))
    print(f"\n{INTERPRETATION}\nSaved {out}")


if __name__ == "__main__":
    main()
