"""Change detection for incremental graph updates.

A small manifest (CSV) remembers the content hash of every ingested post and every
ingested interaction id. A new batch is compared with it so only NEW or CHANGED
records are sent to Neo4j. Because ingestion uses MERGE, re-sending unchanged rows
would also be safe; the manifest only saves time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from graph_enhanced_lm.data.preprocessing import content_hash
from graph_enhanced_lm.data.schemas import CanonicalDataset

logger = logging.getLogger(__name__)


@dataclass
class ChangeSet:
    new_posts: int
    changed_posts: int
    unchanged_posts: int
    new_interactions: int
    dataset: CanonicalDataset


def _post_hashes(posts: pd.DataFrame) -> pd.Series:
    return pd.Series([content_hash(t, ts, u) for t, ts, u in
                      zip(posts["text"], posts["timestamp"], posts["user_id"])],
                     index=posts.index, dtype=object)


def compute_changes(ds: CanonicalDataset, manifest_dir: str | Path) -> ChangeSet:
    """Return a dataset containing only new/changed posts and new interactions."""
    manifest_dir = Path(manifest_dir)
    pm, im = manifest_dir / "posts_manifest.csv", manifest_dir / "interactions_manifest.csv"
    old_posts = pd.read_csv(pm, dtype=str) if pm.exists() else pd.DataFrame(columns=["post_id", "content_hash"])
    old_inter = set(pd.read_csv(im, dtype=str)["interaction_id"]) if im.exists() else set()

    posts = ds.posts.copy()
    posts["_hash"] = _post_hashes(posts)
    old = dict(zip(old_posts["post_id"], old_posts["content_hash"]))
    is_new = ~posts["post_id"].isin(old)
    is_changed = posts["post_id"].isin(old) & (posts["post_id"].map(old) != posts["_hash"])
    keep = posts[is_new | is_changed].drop(columns="_hash")
    inter = ds.interactions[~ds.interactions["interaction_id"].isin(old_inter)]
    # Users are cheap to MERGE, so all users are re-sent (properties may have changed).
    users = ds.users
    subset = CanonicalDataset(name=ds.name, users=users, posts=keep, interactions=inter,
                              follows=ds.follows, user_edges=ds.user_edges,
                              capabilities=ds.capabilities, notes=ds.notes)
    cs = ChangeSet(int(is_new.sum()), int(is_changed.sum()),
                   int(len(posts) - is_new.sum() - is_changed.sum()), len(inter), subset)
    logger.info("Changes: %d new posts, %d changed, %d unchanged, %d new interactions",
                cs.new_posts, cs.changed_posts, cs.unchanged_posts, cs.new_interactions)
    return cs


def update_manifest(ds: CanonicalDataset, manifest_dir: str | Path) -> None:
    """Record the posts/interactions of ``ds`` as ingested (call AFTER a successful write)."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pm, im = manifest_dir / "posts_manifest.csv", manifest_dir / "interactions_manifest.csv"
    new = pd.DataFrame({"post_id": ds.posts["post_id"], "content_hash": _post_hashes(ds.posts).values})
    if pm.exists():
        new = pd.concat([pd.read_csv(pm, dtype=str), new]).drop_duplicates("post_id", keep="last")
    new.to_csv(pm, index=False)
    ids = pd.DataFrame({"interaction_id": ds.interactions["interaction_id"]})
    if im.exists():
        ids = pd.concat([pd.read_csv(im, dtype=str), ids]).drop_duplicates()
    ids.to_csv(im, index=False)
