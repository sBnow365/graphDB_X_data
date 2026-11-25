"""Canonical tabular schema shared by every dataset adapter.

Every source dataset is converted into these tables. Graph ingestion and analysis
only ever see the canonical form, so swapping datasets never touches graph code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

USER_COLUMNS = ["user_id", "username", "dataset_source", "synthetic", "source_bot_label",
                "synthetic_gt_community", "user_hate_label"]
POST_COLUMNS = ["post_id", "user_id", "text", "timestamp", "language", "source_labels",
                "conversation_id", "dataset_source", "synthetic", "is_stub"]
INTERACTION_COLUMNS = ["interaction_id", "interaction_type", "source_user_id", "source_post_id",
                       "target_post_id", "target_user_id", "timestamp", "dataset_source",
                       "synthetic"]
CLASSIFICATION_COLUMNS = ["post_id", "text", "source_labels", "original_label", "group_id",
                          "split", "language", "dataset_source"]
PREDICTION_COLUMNS = ["post_id", "labels", "primary_label", "confidence", "label_source",
                      "model_name", "model_version", "predicted_at"]
HASHTAG_COLUMNS = ["post_id", "tag"]
FOLLOW_COLUMNS = ["source_user_id", "target_user_id", "dataset_source", "synthetic"]
# User->User retweet edges for datasets that only provide a user graph (e.g. Hateful Users)
USER_EDGE_COLUMNS = ["source_user_id", "target_user_id", "edge_type", "weight", "dataset_source",
                     "synthetic"]

# How each interaction type maps onto graph endpoints (documented in GRAPH_SCHEMA.md)
INTERACTION_ENDPOINTS = {
    "RETWEETED": ("source_user_id", "target_post_id"),  # (User)-[:RETWEETED]->(Post)
    "REPLIED_TO": ("source_post_id", "target_post_id"),  # (Post)-[:REPLIED_TO]->(Post)
    "QUOTED": ("source_post_id", "target_post_id"),      # (Post)-[:QUOTED]->(Post)
    "MENTIONED": ("source_post_id", "target_user_id"),   # (Post)-[:MENTIONED]->(User)
}

LABEL_SEPARATOR = ";"


@dataclass
class CanonicalDataset:
    """Users, posts and interactions in canonical form, plus what the source supports."""

    name: str
    users: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=USER_COLUMNS))
    posts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=POST_COLUMNS))
    interactions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=INTERACTION_COLUMNS))
    follows: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FOLLOW_COLUMNS))
    user_edges: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=USER_EDGE_COLUMNS))
    capabilities: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "users": len(self.users),
            "posts": len(self.posts),
            "interactions": len(self.interactions),
            "follows": len(self.follows),
            "user_edges": len(self.user_edges),
            "capabilities": self.capabilities,
            "notes": self.notes,
        }


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add any missing canonical columns (as empty) and order them canonically."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in df.columns if c not in columns]
    return df[columns + extra]


def split_labels(value: Any) -> list[str]:
    """Turn ``'HATE;COMMUNITY_TARGETED'`` into ``['HATE', 'COMMUNITY_TARGETED']``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    return [p.strip() for p in str(value).split(LABEL_SEPARATOR) if p.strip()]


def join_labels(labels: list[str]) -> str:
    """Inverse of :func:`split_labels`, de-duplicated and order-preserving."""
    seen: list[str] = []
    for lab in labels:
        if lab and lab not in seen:
            seen.append(lab)
    return LABEL_SEPARATOR.join(seen)
