"""Temporal trends: daily counts, label proportions, window comparisons.

Two back-ends are provided:
* ``*_df`` functions work on the processed CSV tables (no database needed);
* ``*_neo4j`` functions query the graph.
An increase in OBSERVED activity in a window is not evidence that the election
CAUSED it; reports must say so.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from graph_enhanced_lm.data.schemas import split_labels
from graph_enhanced_lm.graph.cypher_queries import get_template
from graph_enhanced_lm.temporal.date_filters import DateRange

logger = logging.getLogger(__name__)
CAUSALITY_NOTE = ("Observed differences between windows describe activity in this dataset; "
                  "they do not show that the election caused the change.")


def _explode_labels(posts: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = posts.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    missing = int(df["ts"].isna().sum())
    if missing:
        logger.warning("%d post(s) without a usable timestamp are excluded from temporal analysis",
                       missing)
    df = df.dropna(subset=["ts"])
    df["label"] = df[label_col].map(lambda v: split_labels(v) or ["UNKNOWN"])
    return df.explode("label")


def daily_label_counts_df(posts: pd.DataFrame, label_col: str = "labels",
                          tz: str = "UTC") -> pd.DataFrame:
    """Posts per local day per label."""
    df = _explode_labels(posts, label_col)
    df["day"] = df["ts"].dt.tz_convert(tz).dt.date
    return (df.groupby(["day", "label"])["post_id"].nunique().rename("posts")
              .reset_index().sort_values(["day", "label"]))


def window_comparison_df(posts: pd.DataFrame, windows: list[DateRange],
                         label_col: str = "labels",
                         community_col: str | None = None) -> pd.DataFrame:
    """Counts and shares per window (and per community if ``community_col`` is given)."""
    df = _explode_labels(posts, label_col)
    rows: list[dict[str, Any]] = []
    for w in windows:
        sub = df[(df["ts"] >= w.start) & (df["ts"] < w.end)]
        groups = sub.groupby(community_col) if community_col else [(None, sub)]
        for comm, g in groups:
            total = g["post_id"].nunique()
            row: dict[str, Any] = {"window": w.name, "start_utc": w.start.isoformat(),
                                   "end_utc": w.end.isoformat(), "days": round(w.days, 2),
                                   "posts": total, "posts_per_day": round(total / w.days, 3) if w.days else 0}
            if community_col:
                row["community"] = comm
            for lab in ("HATE", "OFFENSIVE", "SPAM", "COMMUNITY_TARGETED", "NORMAL"):
                n = g.loc[g["label"] == lab, "post_id"].nunique()
                row[f"{lab.lower()}_posts"] = n
                row[f"{lab.lower()}_share"] = round(n / total, 4) if total else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def window_comparison_neo4j(client: Any, windows: list[DateRange]) -> pd.DataFrame:
    """Same as :func:`window_comparison_df` (overall level) but from the graph."""
    rows = []
    for w in windows:
        counts = client.run_read(get_template("label_counts_in_range").cypher, w.as_params())
        total = client.run_read(
            "MATCH (p:Post) WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end) "
            "RETURN count(p) AS n", w.as_params())[0]["n"]
        row = {"window": w.name, "start_utc": w.start.isoformat(), "end_utc": w.end.isoformat(),
               "days": round(w.days, 2), "posts": total,
               "posts_per_day": round(total / w.days, 3) if w.days else 0}
        by_label = {r["label"]: r["posts"] for r in counts}
        for lab in ("HATE", "OFFENSIVE", "SPAM", "COMMUNITY_TARGETED", "NORMAL"):
            n = by_label.get(lab, 0)
            row[f"{lab.lower()}_posts"] = n
            row[f"{lab.lower()}_share"] = round(n / total, 4) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def community_window_activity_neo4j(client: Any, windows: list[DateRange],
                                    community_property: str) -> pd.DataFrame:
    """Per-community post / hate / spam counts in each window."""
    frames = []
    for w in windows:
        rows = client.run_read(get_template("community_activity_in_range").cypher,
                               {**w.as_params(), "community_property": community_property})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["window"] = w.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["community", "posts", "hate_posts", "spam_posts", "window"])


def plot_daily_counts(daily: pd.DataFrame, election_day: str | None, path: str | Path,
                      title: str = "Daily posts per label") -> Path:
    """Line chart of daily counts per label with the election day marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = daily.pivot(index="day", columns="label", values="posts").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pivot.plot(ax=ax, linewidth=1.6)
    if election_day:
        ax.axvline(pd.to_datetime(election_day).date(), color="black", linestyle="--",
                   linewidth=1, label="election day")
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("posts")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
