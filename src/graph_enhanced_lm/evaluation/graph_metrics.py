"""Graph-analytics evaluation: community sizes, density, ground-truth recovery.

High modularity or density shows strong STRUCTURE, never harmful intent.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def community_size_distribution(assignments: pd.DataFrame, col: str = "community") -> dict[str, Any]:
    sizes = assignments[col].value_counts()
    return {"communities": int(sizes.size), "largest": int(sizes.max()), "smallest": int(sizes.min()),
            "median": float(sizes.median()), "singletons": int((sizes == 1).sum()),
            "sizes": sizes.sort_values(ascending=False).astype(int).tolist()}


def ground_truth_recovery(assignments: pd.DataFrame, truth: pd.DataFrame, pred_col: str = "community",
                          true_col: str = "synthetic_gt_community", key: str = "user_id") -> dict[str, Any]:
    """NMI / ARI between detected communities and SYNTHETIC ground truth."""
    merged = assignments.merge(truth[[key, true_col]], on=key, how="inner").dropna(subset=[true_col])
    if merged.empty:
        return {"n": 0, "nmi": None, "ari": None}
    return {"n": int(len(merged)),
            "nmi": round(float(normalized_mutual_info_score(merged[true_col], merged[pred_col])), 4),
            "ari": round(float(adjusted_rand_score(merged[true_col], merged[pred_col])), 4)}


def interaction_graph_from_tables(posts: pd.DataFrame, interactions: pd.DataFrame,
                                  weights: dict[str, float]) -> nx.Graph:
    """Build the same undirected weighted User-User graph as the GDS projection, offline."""
    author = dict(zip(posts["post_id"], posts["user_id"]))
    g = nx.Graph()
    for r in interactions.itertuples(index=False):
        t = r.interaction_type
        if t == "RETWEETED":
            a, b = r.source_user_id, author.get(r.target_post_id)
        elif t in ("REPLIED_TO", "QUOTED"):
            a, b = author.get(r.source_post_id), author.get(r.target_post_id)
        elif t == "MENTIONED":
            a, b = author.get(r.source_post_id), r.target_user_id
        else:
            continue
        if not a or not b or a == b or pd.isna(a) or pd.isna(b):
            continue
        w = float(weights.get(t, 1.0))
        g.add_edge(a, b, weight=g[a][b]["weight"] + w if g.has_edge(a, b) else w)
    return g


def offline_louvain(g: nx.Graph, seed: int = 42) -> pd.DataFrame:
    """NetworkX Louvain (for tests / sensitivity analysis without Neo4j)."""
    comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
    return pd.DataFrame([{"user_id": u, "community": i} for i, c in enumerate(comms) for u in c])


def graph_summary(g: nx.Graph) -> dict[str, Any]:
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "density": round(nx.density(g), 6) if g.number_of_nodes() > 1 else 0.0,
            "components": nx.number_connected_components(g) if g.number_of_nodes() else 0}


def modularity_of(g: nx.Graph, assignments: pd.DataFrame) -> float:
    groups = assignments.groupby("community")["user_id"].apply(set).tolist()
    covered = set().union(*groups) if groups else set()
    missing = set(g.nodes) - covered
    if missing:
        groups.append(missing)
    return round(float(nx.community.modularity(g, groups, weight="weight")), 4)
