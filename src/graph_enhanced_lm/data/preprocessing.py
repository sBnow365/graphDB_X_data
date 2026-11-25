"""Deterministic cleaning: ID/timestamp normalisation, de-duplication, hashtags.

Raw files are never modified; these functions return new DataFrames.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import pandas as pd

from graph_enhanced_lm.data.schemas import (
    FOLLOW_COLUMNS, INTERACTION_COLUMNS, POST_COLUMNS, USER_EDGE_COLUMNS, USER_COLUMNS, CanonicalDataset, ensure_columns,
)

logger = logging.getLogger(__name__)
HASHTAG_RE = re.compile(r"(?<!\w)#(\w{1,100})", flags=re.UNICODE)


def normalize_id(value: Any) -> str | None:
    """Return a stripped string id, or None for missing values.

    Floats such as ``123.0`` (a common CSV artefact) become ``'123'``.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    return text or None


def normalize_timestamp(value: Any, assume_tz: str = "UTC") -> str | None:
    """Parse many timestamp formats and return ISO-8601 UTC (``...+00:00``) or None.

    Naive timestamps (no timezone) are interpreted in ``assume_tz``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "nat"}:
        return None
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError):
        try:  # Twitter v1 format: "Wed Oct 10 20:19:24 +0000 2018"
            ts = pd.to_datetime(text, format="%a %b %d %H:%M:%S %z %Y")
        except (ValueError, TypeError):
            return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(assume_tz)
    return ts.tz_convert("UTC").floor("s").isoformat()


def extract_hashtags(text: Any) -> list[str]:
    """Lower-cased unique hashtags in order of appearance."""
    if not isinstance(text, str):
        return []
    tags: list[str] = []
    for tag in HASHTAG_RE.findall(text):
        t = tag.lower()
        if t not in tags:
            tags.append(t)
    return tags


def content_hash(*parts: Any) -> str:
    """Stable SHA-1 of the given parts (used for change detection)."""
    joined = "\u241f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _to_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df, USER_COLUMNS)
    df["user_id"] = df["user_id"].map(normalize_id)
    df = df[df["user_id"].notna()]
    df["synthetic"] = _to_bool(df["synthetic"])
    before = len(df)
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    if before != len(df):
        logger.info("Dropped %d duplicate users", before - len(df))
    return df.sort_values("user_id").reset_index(drop=True)


def clean_posts(df: pd.DataFrame, assume_tz: str = "UTC") -> pd.DataFrame:
    df = ensure_columns(df, POST_COLUMNS)
    df["post_id"] = df["post_id"].map(normalize_id)
    df["user_id"] = df["user_id"].map(normalize_id)
    df = df[df["post_id"].notna()]
    df["timestamp"] = df["timestamp"].map(lambda v: normalize_timestamp(v, assume_tz))
    df["synthetic"] = _to_bool(df["synthetic"])
    df["is_stub"] = _to_bool(df["is_stub"])
    # a real post wins over a stub with the same id
    df = df.sort_values(["post_id", "is_stub"], kind="stable")
    before = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    if before != len(df):
        logger.info("Dropped %d duplicate posts", before - len(df))
    return df.sort_values("post_id").reset_index(drop=True)


def clean_interactions(df: pd.DataFrame, assume_tz: str = "UTC") -> pd.DataFrame:
    df = ensure_columns(df, INTERACTION_COLUMNS)
    for col in ["interaction_id", "source_user_id", "source_post_id",
                "target_post_id", "target_user_id"]:
        df[col] = df[col].map(normalize_id)
    df["interaction_type"] = df["interaction_type"].astype(str).str.upper().str.strip()
    df["timestamp"] = df["timestamp"].map(lambda v: normalize_timestamp(v, assume_tz))
    df["synthetic"] = _to_bool(df["synthetic"])
    missing_id = df["interaction_id"].isna()
    if missing_id.any():  # deterministic id from content
        df.loc[missing_id, "interaction_id"] = df[missing_id].apply(
            lambda r: "i_" + content_hash(r["interaction_type"], r["source_user_id"],
                                          r["source_post_id"], r["target_post_id"],
                                          r["target_user_id"])[:16], axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["interaction_id"], keep="first")
    if before != len(df):
        logger.info("Dropped %d duplicate interactions", before - len(df))
    return df.sort_values("interaction_id").reset_index(drop=True)


def clean_follows(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df, FOLLOW_COLUMNS)
    for col in ["source_user_id", "target_user_id"]:
        df[col] = df[col].map(normalize_id)
    df = df.dropna(subset=["source_user_id", "target_user_id"])
    df["synthetic"] = _to_bool(df["synthetic"])
    return df.drop_duplicates(subset=["source_user_id", "target_user_id"]).reset_index(drop=True)


def clean_user_edges(df: pd.DataFrame) -> pd.DataFrame:
    """User->User edges: normalise ids, drop self-loops, sum duplicate edges into weight."""
    df = ensure_columns(df, USER_EDGE_COLUMNS)
    if df.empty:
        return df
    for col in ["source_user_id", "target_user_id"]:
        df[col] = df[col].map(normalize_id)
    df = df.dropna(subset=["source_user_id", "target_user_id"])
    df = df[df["source_user_id"] != df["target_user_id"]].copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    df["edge_type"] = df["edge_type"].fillna("RETWEETED_USER")
    df["synthetic"] = _to_bool(df["synthetic"])
    keys = ["source_user_id", "target_user_id", "edge_type"]
    agg = df.groupby(keys, as_index=False).agg(weight=("weight", "sum"),
                                                dataset_source=("dataset_source", "first"),
                                                synthetic=("synthetic", "first"))
    return ensure_columns(agg, USER_EDGE_COLUMNS)


def build_hashtag_table(posts: pd.DataFrame) -> pd.DataFrame:
    rows = [{"post_id": pid, "tag": tag}
            for pid, text in zip(posts["post_id"], posts["text"])
            for tag in extract_hashtags(text)]
    return pd.DataFrame(rows, columns=["post_id", "tag"])


def preprocess_canonical(ds: CanonicalDataset, assume_tz: str = "UTC") -> CanonicalDataset:
    """Clean every table of a canonical dataset and return a new dataset."""
    return CanonicalDataset(
        name=ds.name,
        users=clean_users(ds.users),
        posts=clean_posts(ds.posts, assume_tz),
        interactions=clean_interactions(ds.interactions, assume_tz),
        follows=clean_follows(ds.follows),
        user_edges=clean_user_edges(ds.user_edges),
        capabilities=dict(ds.capabilities),
        notes=list(ds.notes),
    )
