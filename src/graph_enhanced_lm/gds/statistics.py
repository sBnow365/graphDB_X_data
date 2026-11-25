"""Community statistics and a TRANSPARENT investigation-priority score.

The score ranks communities for HUMAN review. Its weights are analytical choices from
configs/gds.yaml, not validated thresholds. Output wording always says
"candidate community exhibiting hate/spam propagation patterns".
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from graph_enhanced_lm.graph.cypher_queries import get_template
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

INTERPRETATION = ("Ranking for human review only. A high score reflects label shares and "
                  "interaction structure in this dataset; it is not evidence of coordination "
                  "or malicious intent.")


def fetch_community_stats(client: Neo4jClient, prop: str, min_users: int) -> pd.DataFrame:
    base = pd.DataFrame(client.run_read(get_template("community_label_stats").cypher,
                                        {"community_property": prop, "min_users": min_users}))
    edges = pd.DataFrame(client.run_read(get_template("community_internal_edges").cypher,
                                         {"community_property": prop}))
    if edges.empty:
        edges = pd.DataFrame(client.run_read(get_template("community_internal_user_edges").cypher,
                                             {"community_property": prop}))
    users = pd.DataFrame(client.run_read(get_template("community_user_label_stats").cypher,
                                         {"community_property": prop, "min_users": min_users}))
    if base.empty and users.empty:
        return base
    if base.empty:
        base = users.drop(columns=["normal_users"])
    elif not users.empty and users["annotated_users"].sum() > 0:
        base = base.merge(users[["community", "hateful_users", "annotated_users", "hateful_user_share"]],
                          on="community", how="left")
    if not edges.empty:
        base = base.merge(edges, on="community", how="left")
    for col in ("internal", "external", "internal_ratio"):
        if col not in base.columns:
            base[col] = 0
    return base.fillna({"internal": 0, "external": 0, "internal_ratio": 0.0})


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)


def score_communities(stats: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Add ``priority_score`` = weighted sum of min-max normalised components.

    Components (missing ones count as 0): hate_share, spam_share, offensive_share,
    pre_election_growth, internal_density (= internal_ratio).
    """
    df = stats.copy()
    if df.empty:
        return df
    comp = {
        "hate_share": df.get("hate_share", 0.0),
        "spam_share": df.get("spam_share", 0.0),
        "offensive_share": df.get("offensive_share", 0.0),
        "pre_election_growth": df.get("pre_election_growth", 0.0),
        "internal_density": df.get("internal_ratio", 0.0),
        "hateful_user_share": df.get("hateful_user_share", 0.0),
    }
    score = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        values = comp.get(name)
        if values is None:
            continue
        values = pd.Series(values, index=df.index) if not isinstance(values, pd.Series) else values
        norm = _minmax(values.fillna(0.0))
        df[f"norm_{name}"] = norm.round(4)
        score += float(w) * norm
    df["priority_score"] = score.round(4)
    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="min").astype(int)
    df["designation"] = "candidate community exhibiting hate/spam propagation patterns (for review)"
    no_posts_signal = (df.get("hate_share", 0) == 0) & (df.get("spam_share", 0) == 0)
    no_user_signal = df.get("hateful_user_share", pd.Series(0.0, index=df.index)).fillna(0) == 0
    df.loc[no_posts_signal & no_user_signal, "designation"] = "no hate/spam signal observed"
    df["interpretation"] = INTERPRETATION
    return df.sort_values(["priority_score", "community"], ascending=[False, True]).reset_index(drop=True)


def add_pre_election_growth(stats: pd.DataFrame, window_activity: pd.DataFrame,
                            near: str, far: str, near_days: float, far_days: float) -> pd.DataFrame:
    """Growth of hate+spam posts per day from the ``far`` window to the ``near`` window.

    ``window_activity`` has columns community, window, hate_posts, spam_posts.
    Growth = rate_near - rate_far (posts/day). Observed change only, NOT causation.
    """
    df = stats.copy()
    if window_activity.empty:
        df["pre_election_growth"] = 0.0
        return df
    wa = window_activity.assign(problem=lambda d: d["hate_posts"] + d["spam_posts"])
    near_rate = wa[wa["window"] == near].set_index("community")["problem"] / max(near_days, 1)
    far_rate = wa[wa["window"] == far].set_index("community")["problem"] / max(far_days, 1)
    growth = near_rate.sub(far_rate, fill_value=0.0)
    df["pre_election_growth"] = df["community"].map(growth).fillna(0.0).round(4)
    return df
