"""Train/validation/test splitting that avoids leakage.

If a group column (user or conversation) is present, all rows of a group go to the
same split, so the classifier is never tested on users it saw during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(df: pd.DataFrame, train: float = 0.8, val: float = 0.1,
                  seed: int = 42, group_col: str | None = None,
                  existing_split_col: str = "split") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column (train/val/test).

    Rows that already have an official split (e.g. HateXplain divisions) keep it.
    """
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("train/val fractions must be in (0,1) and sum to < 1")
    out = df.copy()
    if existing_split_col in out.columns and out[existing_split_col].notna().all():
        return out
    rng = np.random.default_rng(seed)
    if group_col and group_col in out.columns and out[group_col].notna().any():
        keys = out[group_col].fillna(out["post_id"]).astype(str)
        unique = np.array(sorted(keys.unique()))
        rng.shuffle(unique)
        n = len(unique)
        n_train, n_val = int(round(n * train)), int(round(n * val))
        mapping = {k: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
                   for i, k in enumerate(unique)}
        out["split"] = keys.map(mapping)
    else:
        idx = np.arange(len(out))
        rng.shuffle(idx)
        n_train, n_val = int(round(len(out) * train)), int(round(len(out) * val))
        split = np.empty(len(out), dtype=object)
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"
        out["split"] = split
    return out


def check_group_leakage(df: pd.DataFrame, group_col: str) -> int:
    """Number of groups that appear in more than one split (should be 0)."""
    if group_col not in df.columns:
        return 0
    counts = df.dropna(subset=[group_col]).groupby(group_col)["split"].nunique()
    return int((counts > 1).sum())
