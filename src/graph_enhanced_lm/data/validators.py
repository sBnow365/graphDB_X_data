"""Validation rules producing a human-readable :class:`ValidationReport`."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from graph_enhanced_lm.data.schemas import CanonicalDataset


@dataclass
class ValidationReport:
    dataset: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d

    def pretty(self) -> str:
        lines = [f"Validation report for '{self.dataset}': {'PASS' if self.ok else 'FAIL'}"]
        for k, v in self.stats.items():
            lines.append(f"  - {k}: {v}")
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def check_required_columns(df: pd.DataFrame, required: list[str], table: str,
                           report: ValidationReport) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"{table}: missing required columns {missing}")


def check_nulls(df: pd.DataFrame, required: list[str], table: str,
                report: ValidationReport, error_cols: tuple[str, ...] = ()) -> None:
    for col in required:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n:
            msg = f"{table}.{col}: {n} missing value(s)"
            (report.errors if col in error_cols else report.warnings).append(msg)
        report.stats[f"{table}.{col}.missing"] = n


def check_duplicates(df: pd.DataFrame, key: str, table: str, report: ValidationReport) -> None:
    if key in df.columns:
        n = int(df[key].duplicated().sum())
        report.stats[f"{table}.duplicate_{key}"] = n
        if n:
            report.errors.append(f"{table}: {n} duplicate {key} value(s)")


def validate_canonical(ds: CanonicalDataset, required: dict[str, list[str]],
                       interaction_types: list[str]) -> ValidationReport:
    """Validate schema, keys, timestamps and referential integrity."""
    rep = ValidationReport(dataset=ds.name)
    rep.stats.update({"users": len(ds.users), "posts": len(ds.posts),
                      "interactions": len(ds.interactions)})
    check_required_columns(ds.users, required["users"], "users", rep)
    check_required_columns(ds.posts, required["posts"], "posts", rep)
    check_required_columns(ds.interactions, required["interactions"], "interactions", rep)
    if rep.errors:
        return rep

    check_nulls(ds.users, required["users"], "users", rep, error_cols=("user_id",))
    check_nulls(ds.posts, required["posts"], "posts", rep, error_cols=("post_id",))
    check_nulls(ds.interactions, required["interactions"], "interactions", rep,
                error_cols=("interaction_id", "interaction_type"))
    check_duplicates(ds.users, "user_id", "users", rep)
    check_duplicates(ds.posts, "post_id", "posts", rep)
    check_duplicates(ds.interactions, "interaction_id", "interactions", rep)

    # Timestamps must parse
    if "timestamp" in ds.posts.columns:
        ts = pd.to_datetime(ds.posts["timestamp"], errors="coerce", utc=True)
        bad = int((ts.isna() & ds.posts["timestamp"].notna()).sum())
        if bad:
            rep.errors.append(f"posts.timestamp: {bad} unparseable value(s)")
        if ts.notna().any():
            rep.stats["posts.time_range"] = f"{ts.min().isoformat()} -> {ts.max().isoformat()}"

    # Referential integrity
    user_ids = set(ds.users["user_id"].dropna())
    post_ids = set(ds.posts["post_id"].dropna())
    orphan_posts = int((~ds.posts["user_id"].isin(user_ids) & ds.posts["user_id"].notna()).sum())
    if orphan_posts:
        rep.warnings.append(f"posts: {orphan_posts} post(s) reference unknown users "
                            "(they will be created as bare User nodes)")
    if len(ds.interactions):
        bad_types = sorted(set(ds.interactions["interaction_type"]) - set(interaction_types))
        if bad_types:
            rep.errors.append(f"interactions: unknown interaction types {bad_types}")
        for col, known, name in [("target_post_id", post_ids, "posts"),
                                 ("source_post_id", post_ids, "posts"),
                                 ("source_user_id", user_ids, "users"),
                                 ("target_user_id", user_ids, "users")]:
            if col in ds.interactions.columns:
                vals = ds.interactions[col].dropna()
                n = int((~vals.isin(known)).sum())
                if n:
                    rep.warnings.append(f"interactions.{col}: {n} reference(s) to unknown {name}")
    ue = getattr(ds, "user_edges", None)
    if ue is not None and len(ue):
        rep.stats["user_edges"] = len(ue)
        for col in ("source_user_id", "target_user_id"):
            n = int((~ue[col].isin(user_ids)).sum())
            if n:
                rep.warnings.append(f"user_edges.{col}: {n} reference(s) to unknown users")
    return rep


def validate_classification_table(df: pd.DataFrame, required: list[str],
                                  name: str) -> ValidationReport:
    """Validation for classifier-only datasets (HateXplain / HASOC)."""
    rep = ValidationReport(dataset=name)
    rep.stats["rows"] = len(df)
    check_required_columns(df, required, "posts", rep)
    if rep.errors:
        return rep
    check_nulls(df, required, "posts", rep, error_cols=("post_id", "text"))
    check_duplicates(df, "post_id", "posts", rep)
    rep.stats["label_distribution"] = df["source_labels"].value_counts(dropna=False).to_dict()
    empty_text = int((df["text"].fillna("").str.strip() == "").sum())
    if empty_text:
        rep.warnings.append(f"{empty_text} post(s) have empty text")
    return rep
