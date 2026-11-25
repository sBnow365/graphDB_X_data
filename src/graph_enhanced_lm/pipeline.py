"""Reusable pipeline steps shared by scripts/ and the end-to-end demo.

Layout of processed data:
  data/processed/<dataset>/users.csv, posts.csv, interactions.csv, follows.csv
  data/processed/<dataset>/predictions.csv        (classifier output)
  data/processed/<dataset>/classification.csv     (HateXplain / HASOC only)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from graph_enhanced_lm.classification.factory import build_classifier
from graph_enhanced_lm.config import AppConfig, PROJECT_ROOT, load_dataset_mapping
from graph_enhanced_lm.data.dataset_adapters import get_classification_adapter, get_graph_adapter
from graph_enhanced_lm.data.loaders import read_table, write_table
from graph_enhanced_lm.data.preprocessing import preprocess_canonical
from graph_enhanced_lm.data.schemas import (
    FOLLOW_COLUMNS, INTERACTION_COLUMNS, POST_COLUMNS, PREDICTION_COLUMNS, USER_COLUMNS,
    USER_EDGE_COLUMNS,
    CanonicalDataset, ensure_columns, join_labels,
)
from graph_enhanced_lm.data.splitting import assign_splits, check_group_leakage
from graph_enhanced_lm.data.synthetic import generate_synthetic_dataset
from graph_enhanced_lm.data.validators import validate_canonical, validate_classification_table
from graph_enhanced_lm.utils.errors import DatasetError
from graph_enhanced_lm.utils.reproducibility import save_json, utc_now_iso

logger = logging.getLogger(__name__)
CLASSIFIER_DATASETS = {"hatexplain", "hasoc"}


def processed_dir(cfg: AppConfig, name: str) -> Path:
    return cfg.path("processed") / name


# ------------------------------------------------------------------ synthetic
def generate_synthetic(cfg: AppConfig, out_dir: Path | None = None) -> CanonicalDataset:
    ds = generate_synthetic_dataset(cfg.synthetic)
    out = out_dir or cfg.path("synthetic")
    write_table(ds.users, out / "users.csv")
    write_table(ds.posts, out / "posts.csv")
    write_table(ds.interactions, out / "interactions.csv")
    save_json({**ds.summary(), "generated_at": utc_now_iso(), "config": cfg.synthetic},
              out / "README_SYNTHETIC.json")
    return ds


# ------------------------------------------------------------------ preprocessing
def preprocess_graph_dataset(cfg: AppConfig, name: str) -> tuple[CanonicalDataset, Any]:
    """Adapter -> clean -> validate -> write processed CSVs + validation report."""
    raw = get_graph_adapter(name).load()
    clean = preprocess_canonical(raw, assume_tz="UTC")
    report = validate_canonical(clean, {k: v["required"] for k, v in cfg.dataset["canonical_tables"].items()
                                        if k != "classification_posts"},
                                cfg.dataset["interaction_types"])
    out = processed_dir(cfg, name)
    write_table(clean.users, out / "users.csv")
    write_table(clean.posts, out / "posts.csv")
    write_table(clean.interactions, out / "interactions.csv")
    write_table(clean.follows, out / "follows.csv")
    write_table(clean.user_edges, out / "user_edges.csv")
    save_json(report.to_dict(), out / "validation_report.json")
    save_json(clean.summary(), out / "dataset_summary.json")
    return clean, report


def preprocess_classification_dataset(cfg: AppConfig, name: str) -> tuple[pd.DataFrame, Any]:
    df = get_classification_adapter(name, cfg.label_taxonomy).load()
    sp = cfg.dataset["splits"]
    df = assign_splits(df, sp["train"], sp["val"], int(cfg.project["random_seed"]), sp.get("group_column"))
    leakage = check_group_leakage(df, sp.get("group_column", "group_id"))
    report = validate_classification_table(df, cfg.dataset["canonical_tables"]["classification_posts"]["required"], name)
    report.stats["split_counts"] = df["split"].value_counts().to_dict()
    report.stats["groups_in_multiple_splits"] = leakage
    if leakage:
        report.errors.append(f"{leakage} group(s) appear in more than one split (leakage)")
    out = processed_dir(cfg, name)
    write_table(df, out / "classification.csv")
    save_json(report.to_dict(), out / "validation_report.json")
    return df, report


def load_processed(cfg: AppConfig, name: str) -> CanonicalDataset:
    d = processed_dir(cfg, name)
    if not (d / "posts.csv").exists():
        raise DatasetError(f"No processed data in {d}. Run: python scripts/preprocess_dataset.py --dataset {name}")
    follows = d / "follows.csv"
    user_edges = d / "user_edges.csv"
    caps = load_dataset_mapping(name).get("capabilities", {})
    return CanonicalDataset(
        name=name,
        users=ensure_columns(read_table(d / "users.csv"), USER_COLUMNS),
        posts=ensure_columns(read_table(d / "posts.csv"), POST_COLUMNS),
        interactions=ensure_columns(read_table(d / "interactions.csv"), INTERACTION_COLUMNS),
        follows=ensure_columns(read_table(follows), FOLLOW_COLUMNS) if follows.exists() else
        pd.DataFrame(columns=FOLLOW_COLUMNS),
        user_edges=ensure_columns(read_table(user_edges), USER_EDGE_COLUMNS) if user_edges.exists() else
        pd.DataFrame(columns=USER_EDGE_COLUMNS),
        capabilities=caps)


# ------------------------------------------------------------------ classification
def classify_dataset(cfg: AppConfig, name: str, backend: str | None = None) -> pd.DataFrame:  #this fn is all data plumbing
    """Run the configured classifier on processed posts and save predictions.csv."""
    ds = load_processed(cfg, name)
    if ds.posts.empty:
        raise DatasetError(f"'{name}' has no posts to classify (e.g. Hateful Users has user-level "
                           "labels only). Skip this step and ingest directly.")
    ccfg = dict(cfg.classifier)
    if backend:
        ccfg["backend"] = backend
    if ccfg["backend"] == "source_labels" and ds.posts["source_labels"].isna().all():
        raise DatasetError(f"'{name}' has no source labels; use --backend transformer")
    clf = build_classifier(ccfg, cfg.label_taxonomy, dataset_source=name)
    # Posts without text (e.g. stub originals of retweets) are NOT sent to the model.
    has_text = ds.posts["text"].fillna("").astype(str).str.strip() != ""
    todo = ds.posts[has_text]
    preds = clf.predict(todo["post_id"].tolist(), todo["text"].tolist(),
                        todo["source_labels"].tolist()) if len(todo) else []
    from graph_enhanced_lm.classification.base import Prediction
    preds += [Prediction(pid, ["UNKNOWN"], "UNKNOWN", None, "unlabeled", "no_text", "0")
              for pid in ds.posts.loc[~has_text, "post_id"]]
    if (~has_text).any():
        logger.info("%d post(s) without text labelled UNKNOWN (not sent to the model)", int((~has_text).sum()))
    now = utc_now_iso()
    df = pd.DataFrame([{"post_id": p.post_id, "labels": join_labels(p.labels),
                        "primary_label": p.primary_label, "confidence": p.confidence,
                        "label_source": p.label_source, "model_name": p.model_name,
                        "model_version": p.model_version, "predicted_at": now} for p in preds],
                      columns=PREDICTION_COLUMNS)
    write_table(df, processed_dir(cfg, name) / "predictions.csv")
    return df


def merge_predictions(ds: CanonicalDataset, preds: pd.DataFrame) -> pd.DataFrame:
    """Posts joined with their predicted labels (used by offline temporal analysis)."""
    return ds.posts.merge(preds[["post_id", "labels", "primary_label"]], on="post_id", how="left")


# ------------------------------------------------------------------ graph
def connect(cfg: AppConfig) -> Any:
    from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

    return Neo4jClient.from_config(cfg)


def ingest(cfg: AppConfig, name: str, client: Any, incremental: bool = False) -> dict[str, Any]:
    from graph_enhanced_lm.graph.incremental_updates import compute_changes, update_manifest
    from graph_enhanced_lm.graph.ingestion import GraphIngestor

    ds = load_processed(cfg, name)
    pred_path = processed_dir(cfg, name) / "predictions.csv"
    preds = read_table(pred_path) if pred_path.exists() else None
    if preds is None:
        logger.warning("No predictions.csv: posts will be ingested WITHOUT labels")
    manifest_dir = processed_dir(cfg, name) / "_manifest"
    target = ds
    report: dict[str, Any] = {}
    if incremental:
        cs = compute_changes(ds, manifest_dir)
        target = cs.dataset
        report["changes"] = {"new_posts": cs.new_posts, "changed_posts": cs.changed_posts,
                             "unchanged_posts": cs.unchanged_posts, "new_interactions": cs.new_interactions}
        if preds is not None:
            preds = preds[preds["post_id"].isin(set(target.posts["post_id"]))]
    ing = GraphIngestor(client, int(cfg.neo4j.get("batch_size", 1000)))
    report.update(ing.ingest_dataset(target, preds))
    update_manifest(target, manifest_dir)
    save_json(report, cfg.path("logs") / f"ingestion_{ing.batch_id}.json")
    return report
