"""Read processed data and saved metrics for display (no database needed)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from graph_enhanced_lm.config import PROJECT_ROOT, get_config

DATASETS: dict[str, dict[str, Any]] = {
    "synthetic": {"title": "Synthetic (built-in)", "role": "graph + labels (test data)", "access": "generated locally",
                  "link": None, "classify": "source_labels", "temporal": True,
                  "about": "300 fictional users in 5 planted communities around a fictional election (15 Mar 2025)."},
    "hateful_users": {"title": "Hateful Users on Twitter", "role": "user graph + human user labels",
                      "access": "Kaggle download (free login)",
                      "link": "https://www.kaggle.com/datasets/manoelribeiro/hateful-users-on-twitter",
                      "classify": None, "temporal": False,
                      "about": "~100k users, ~2.2M retweet edges, ~5k users annotated hateful/normal. No text/timestamps."},
    "russian_trolls": {"title": "Russian troll tweets (NBC News)", "role": "graph + text + time (2016 US election)",
                       "access": "public GitHub", "link": "https://github.com/loretoparisi/russian_trolls_tweets_dataset",
                       "classify": "transformer", "temporal": True,
                       "about": "Tweets from IRA-linked accounts; retweets/replies/mentions; election 8 Nov 2016."},
    "twibot22": {"title": "TwiBot-22", "role": "graph + text + time (bot labels only)",
                 "access": "application to authors", "link": "https://github.com/LuoUndergradXJTU/TwiBot-22",
                 "classify": "transformer", "temporal": True,
                 "about": "Large Twitter graph; labels are bot/human, never hate."},
    "hatexplain": {"title": "HateXplain", "role": "classifier only", "access": "public GitHub (script downloads)",
                   "link": "https://github.com/hate-alert/HateXplain", "classify": None, "temporal": False,
                   "about": "20,148 English posts labelled hate/offensive/normal + target communities."},
    "hasoc": {"title": "HASOC", "role": "classifier only", "access": "manual download (password via form)",
              "link": "https://hasocfire.github.io/hasoc/", "classify": None, "temporal": False,
              "about": "English/Hindi tweets, HOF/NOT and HATE/OFFN/PRFN."},
}
GRAPH_DATASETS = ["synthetic", "hateful_users", "russian_trolls", "twibot22"]
CLASSIFIER_DATASETS = ["hatexplain", "hasoc"]


def processed_path(name: str) -> Path:
    return get_config().path("processed") / name


def dataset_status(name: str) -> dict[str, Any]:
    d = processed_path(name)
    raw = get_config().path("raw") / name
    return {
        "raw_files": sorted(p.name for p in raw.iterdir() if p.is_file() and not p.name.startswith("_"))
        if raw.exists() else [],
        "processed": (d / "posts.csv").exists() or (d / "classification.csv").exists() or (d / "users.csv").exists(),
        "predictions": (d / "predictions.csv").exists(),
        "validation": load_json(d / "validation_report.json"),
        "summary": load_json(d / "dataset_summary.json"),
    }


def read_csv_safe(path: Path, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, nrows=nrows,
                           usecols=(lambda c: c in usecols) if usecols else None)
    except (ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def label_distribution(name: str) -> pd.DataFrame:
    d = processed_path(name)
    if (d / "predictions.csv").exists():
        df = read_csv_safe(d / "predictions.csv", usecols=["primary_label", "label_source"])
        return df.groupby(["primary_label", "label_source"]).size().rename("posts").reset_index()
    if (d / "classification.csv").exists():
        df = read_csv_safe(d / "classification.csv", usecols=["source_labels", "split"])
        df["primary_label"] = df["source_labels"].str.split(";").str[0]
        return df.groupby(["primary_label", "split"]).size().rename("posts").reset_index()
    if (d / "users.csv").exists():
        df = read_csv_safe(d / "users.csv", usecols=["user_hate_label"])
        if "user_hate_label" in df and df["user_hate_label"].notna().any():
            return df["user_hate_label"].fillna("unannotated").value_counts().rename_axis(
                "user_label").rename("users").reset_index()
    return pd.DataFrame()


def metric_files() -> list[Path]:
    m = get_config().path("metrics")
    return sorted([p for p in m.glob("*.json")] + [p for p in m.glob("*.md")]) if m.exists() else []


def results_markdown() -> str | None:
    p = PROJECT_ROOT / "reports" / "experiment_results.md"
    return p.read_text(encoding="utf-8") if p.exists() else None
