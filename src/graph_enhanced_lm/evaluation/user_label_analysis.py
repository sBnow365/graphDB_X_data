"""Hateful-user analysis for user-level labelled graphs (Hateful Users dataset).

Two questions, both answered from HUMAN annotations (not model predictions):
1. Homophily: do annotated hateful users retweet other hateful users more than chance?
2. Concentration: are hateful users concentrated in a few Louvain communities?
Neither result says anything about coordination or intent.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def homophily_table(users: pd.DataFrame, edges: pd.DataFrame,
                    label_col: str = "user_hate_label") -> dict[str, Any]:
    """Edge counts between annotated label pairs and the 'lift' over a random baseline.

    lift(H->H) = observed share of H->H among edges between annotated users, divided by
    the share expected if retweet targets were independent of labels.
    """
    lab = users.dropna(subset=[label_col]).set_index("user_id")[label_col]
    e = edges.assign(src=edges["source_user_id"].map(lab), dst=edges["target_user_id"].map(lab)).dropna(
        subset=["src", "dst"])
    if e.empty:
        return {"annotated_edges": 0}
    counts = e.groupby(["src", "dst"]).size()
    total = int(counts.sum())
    out: dict[str, Any] = {"annotated_edges": total, "pairs": {}}
    src_share = e["src"].value_counts(normalize=True)
    dst_share = e["dst"].value_counts(normalize=True)
    for (s, d), n in counts.items():
        expected = src_share.get(s, 0) * dst_share.get(d, 0)
        out["pairs"][f"{s}->{d}"] = {"edges": int(n), "share": round(n / total, 4),
                                     "lift_vs_random": round((n / total) / expected, 3) if expected else None}
    rate = lambda s: (counts.get((s, "HATEFUL"), 0) / counts.xs(s, level=0).sum()
                      if s in counts.index.get_level_values(0) else None)
    out["p_retweet_hateful_given_hateful"] = None if rate("HATEFUL") is None else round(float(rate("HATEFUL")), 4)
    out["p_retweet_hateful_given_normal"] = None if rate("NORMAL") is None else round(float(rate("NORMAL")), 4)
    return out


def concentration(community_stats: pd.DataFrame, top_k: int = 5) -> dict[str, Any]:
    """Share of all annotated hateful users that sit in the top-k communities by hateful count."""
    if community_stats.empty or "hateful_users" not in community_stats:
        return {"available": False}
    df = community_stats.sort_values("hateful_users", ascending=False)
    total_h = int(df["hateful_users"].sum())
    total_a = int(df["annotated_users"].sum())
    base_rate = total_h / total_a if total_a else None
    top = df.head(top_k)
    return {
        "available": True, "communities": int(len(df)), "hateful_users": total_h,
        "annotated_users": total_a, "overall_hateful_rate": round(base_rate, 4) if base_rate else None,
        f"share_of_hateful_in_top_{top_k}": round(int(top["hateful_users"].sum()) / total_h, 4) if total_h else None,
        f"share_of_users_in_top_{top_k}": round(int(top["users"].sum()) / int(df["users"].sum()), 4),
        "top_communities": top[["community", "users", "hateful_users", "annotated_users",
                                "hateful_user_share"]].to_dict("records"),
    }
