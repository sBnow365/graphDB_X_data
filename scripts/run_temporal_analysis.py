"""Election-window comparison and daily trends.

    python scripts/run_temporal_analysis.py --source csv      # no Neo4j needed
    python scripts/run_temporal_analysis.py --source neo4j
    python scripts/run_temporal_analysis.py --election-date 2024-04-19   # YOUR verified date
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.data.loaders import read_table
from graph_enhanced_lm.pipeline import connect, load_processed, merge_predictions, processed_dir
from graph_enhanced_lm.temporal.election_windows import build_windows
from graph_enhanced_lm.temporal.trend_analysis import (
    CAUSALITY_NOTE, community_window_activity_neo4j, daily_label_counts_df, plot_daily_counts,
    window_comparison_df, window_comparison_neo4j,
)
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import df_to_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["csv", "neo4j"], default="csv")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--election-date", help="override configs/temporal.yaml (YYYY-MM-DD)")
    ap.add_argument("--timezone", help="override configs/temporal.yaml timezone, e.g. America/New_York")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "temporal")
    if args.election_date:
        cfg.temporal["election_date"] = args.election_date
    if args.timezone:
        cfg.temporal["timezone"] = args.timezone
    name = args.dataset or cfg.dataset["active_graph_dataset"]
    windows = build_windows(cfg.temporal)
    tz = cfg.temporal.get("timezone", "UTC")
    out = cfg.path("metrics")
    if args.source == "csv":
        ds = load_processed(cfg, name)
        posts = merge_predictions(ds, read_table(processed_dir(cfg, name) / "predictions.csv"))
        overall = window_comparison_df(posts, windows)
        daily = daily_label_counts_df(posts, tz=tz)
        fig = plot_daily_counts(daily, cfg.temporal["election_date"], cfg.path("figures") / "daily_label_counts.png",
                                title="Daily posts per label (SYNTHETIC DATA)" if name == "synthetic" else f"Daily posts per label ({name})")
        daily.to_csv(out / "daily_label_counts.csv", index=False)
        print(f"Figure: {fig}")
    else:
        with connect(cfg) as client:
            overall = window_comparison_neo4j(client, windows)
            comm = community_window_activity_neo4j(client, windows, cfg.gds["louvain"]["write_property"])
        comm.to_csv(out / "community_window_activity.csv", index=False)
        print(comm.pivot_table(index="community", columns="window", values="hate_posts", fill_value=0).to_string())
    overall.to_csv(out / "window_comparison.csv", index=False)
    cols = ["window", "days", "posts", "posts_per_day", "hate_share", "offensive_share", "spam_share"]
    (out / "window_comparison.md").write_text(
        f"# Election-window comparison ({name})\n\nElection date (config): {cfg.temporal['election_date']} ({tz})\n\n"
        + df_to_markdown(overall[cols]) + f"\n\n{CAUSALITY_NOTE}\n", encoding="utf-8")
    print(overall[cols].to_string(index=False))
    print(f"\n{CAUSALITY_NOTE}")


if __name__ == "__main__":
    main()
