"""Idempotent, batched ingestion of canonical tables into Neo4j.

Every statement uses ``UNWIND $rows AS row`` + ``MERGE`` on a unique key, so running
ingestion twice creates no duplicates. Each write records ``last_batch`` for provenance.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from graph_enhanced_lm.classification.base import Prediction
from graph_enhanced_lm.data.preprocessing import build_hashtag_table, content_hash
from graph_enhanced_lm.data.schemas import CanonicalDataset
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.utils.reproducibility import new_run_id, utc_now_iso

logger = logging.getLogger(__name__)

Q_USERS = """
UNWIND $rows AS row
MERGE (u:User {user_id: row.user_id})
  ON CREATE SET u.created_batch = $batch_id
SET u.username = row.username,
    u.dataset_source = row.dataset_source,
    u.synthetic = row.synthetic,
    u.source_bot_label = row.source_bot_label,
    u.synthetic_gt_community = row.synthetic_gt_community,
    u.user_hate_label = row.user_hate_label,
    u.user_label_source = CASE WHEN row.user_hate_label IS NULL THEN null ELSE 'human_crowdsourced' END,
    u.placeholder = false,
    u.last_batch = $batch_id
"""

Q_POSTS = """
UNWIND $rows AS row
MERGE (p:Post {post_id: row.post_id})
  ON CREATE SET p.created_batch = $batch_id
SET p.text = row.text,
    p.timestamp = CASE WHEN row.timestamp IS NULL THEN null ELSE datetime(row.timestamp) END,
    p.language = row.language,
    p.conversation_id = row.conversation_id,
    p.dataset_source = row.dataset_source,
    p.synthetic = row.synthetic,
    p.content_hash = row.content_hash,
    p.placeholder = coalesce(row.is_stub, false),
    p.last_batch = $batch_id
WITH p, row WHERE row.user_id IS NOT NULL
MERGE (u:User {user_id: row.user_id})
  ON CREATE SET u.placeholder = true, u.created_batch = $batch_id
MERGE (u)-[r:POSTED]->(p)
SET r.timestamp = p.timestamp
"""

# Primary label: a model never overwrites a human annotation.
Q_PRIMARY_LABEL = """
UNWIND $rows AS row
MATCH (p:Post {post_id: row.post_id})
WITH p, row
WHERE p.primary_label_source IS NULL OR p.primary_label_source <> 'human'
      OR row.label_source = 'human'
SET p.primary_label = row.primary_label,
    p.labels = row.labels,
    p.primary_label_source = row.label_source,
    p.classification_confidence = row.confidence,
    p.classifier_model = row.model_name,
    p.classifier_version = row.model_version
"""

# Every label source keeps its own HAS_LABEL edges (provenance is never lost).
Q_HAS_LABEL = """
UNWIND $rows AS row
MATCH (p:Post {post_id: row.post_id})
OPTIONAL MATCH (p)-[old:HAS_LABEL {label_source: row.label_source, model_name: row.model_name}]->(ol:ContentLabel)
WHERE NOT ol.name IN row.labels
DELETE old
WITH DISTINCT p, row
UNWIND row.labels AS lab
MERGE (l:ContentLabel {name: lab})
MERGE (p)-[h:HAS_LABEL {label_source: row.label_source, model_name: row.model_name}]->(l)
SET h.confidence = row.confidence, h.model_version = row.model_version, h.batch_id = $batch_id
"""

_INTERACTION_SET = """
SET r.timestamp = CASE WHEN row.timestamp IS NULL THEN null ELSE datetime(row.timestamp) END,
    r.dataset_source = row.dataset_source, r.synthetic = row.synthetic, r.batch_id = $batch_id
"""
Q_INTERACTIONS = {
    "RETWEETED": """
UNWIND $rows AS row
MERGE (u:User {user_id: row.source_user_id}) ON CREATE SET u.placeholder = true, u.created_batch = $batch_id
MERGE (p:Post {post_id: row.target_post_id}) ON CREATE SET p.placeholder = true, p.created_batch = $batch_id
MERGE (u)-[r:RETWEETED {interaction_id: row.interaction_id}]->(p)""" + _INTERACTION_SET,
    "REPLIED_TO": """
UNWIND $rows AS row
MERGE (s:Post {post_id: row.source_post_id}) ON CREATE SET s.placeholder = true, s.created_batch = $batch_id
MERGE (t:Post {post_id: row.target_post_id}) ON CREATE SET t.placeholder = true, t.created_batch = $batch_id
MERGE (s)-[r:REPLIED_TO {interaction_id: row.interaction_id}]->(t)""" + _INTERACTION_SET,
    "QUOTED": """
UNWIND $rows AS row
MERGE (s:Post {post_id: row.source_post_id}) ON CREATE SET s.placeholder = true, s.created_batch = $batch_id
MERGE (t:Post {post_id: row.target_post_id}) ON CREATE SET t.placeholder = true, t.created_batch = $batch_id
MERGE (s)-[r:QUOTED {interaction_id: row.interaction_id}]->(t)""" + _INTERACTION_SET,
    "MENTIONED": """
UNWIND $rows AS row
MERGE (s:Post {post_id: row.source_post_id}) ON CREATE SET s.placeholder = true, s.created_batch = $batch_id
MERGE (u:User {user_id: row.target_user_id}) ON CREATE SET u.placeholder = true, u.created_batch = $batch_id
MERGE (s)-[r:MENTIONED {interaction_id: row.interaction_id}]->(u)""" + _INTERACTION_SET,
}
REQUIRED_ENDPOINTS = {"RETWEETED": ["source_user_id", "target_post_id"],
                      "REPLIED_TO": ["source_post_id", "target_post_id"],
                      "QUOTED": ["source_post_id", "target_post_id"],
                      "MENTIONED": ["source_post_id", "target_user_id"]}

Q_HASHTAGS = """
UNWIND $rows AS row
MATCH (p:Post {post_id: row.post_id})
MERGE (h:Hashtag {tag: row.tag})
MERGE (p)-[:USES_HASHTAG]->(h)
"""

Q_FOLLOWS = """
UNWIND $rows AS row
MERGE (a:User {user_id: row.source_user_id}) ON CREATE SET a.placeholder = true
MERGE (b:User {user_id: row.target_user_id}) ON CREATE SET b.placeholder = true
MERGE (a)-[r:FOLLOWS]->(b)
SET r.dataset_source = row.dataset_source
"""

Q_USER_EDGES = """
UNWIND $rows AS row
MERGE (a:User {user_id: row.source_user_id}) ON CREATE SET a.placeholder = true, a.created_batch = $batch_id
MERGE (b:User {user_id: row.target_user_id}) ON CREATE SET b.placeholder = true, b.created_batch = $batch_id
MERGE (a)-[r:RETWEETED_USER]->(b)
SET r.weight = toFloat(coalesce(row.weight, 1.0)), r.dataset_source = row.dataset_source,
    r.batch_id = $batch_id
"""

Q_BATCH = """
MERGE (b:IngestionBatch {batch_id: $batch_id})
SET b.dataset_source = $dataset, b.ingested_at = datetime($at), b.counts = $counts_json
"""


def df_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list of dicts with NaN converted to None (Neo4j-safe)."""
    if df.empty:
        return []
    obj = df.astype(object)
    return obj.where(obj.notna(), None).to_dict(orient="records")


def predictions_to_rows(preds: list[Prediction]) -> list[dict[str, Any]]:
    return [{"post_id": p.post_id, "labels": p.labels, "primary_label": p.primary_label,
             "confidence": p.confidence, "label_source": p.label_source,
             "model_name": p.model_name, "model_version": p.model_version} for p in preds]


class GraphIngestor:
    def __init__(self, client: Neo4jClient, batch_size: int = 1000, batch_id: str | None = None):
        self.client = client
        self.batch_size = batch_size
        self.batch_id = batch_id or new_run_id("batch")

    def _write(self, query: str, rows: list[dict[str, Any]], what: str) -> dict[str, int]:
        if not rows:
            return {}
        counters = self.client.write_batches(query, rows, self.batch_size, {"batch_id": self.batch_id})
        logger.info("Ingested %d %s rows: %s", len(rows), what, counters)
        return counters

    def ingest_users(self, users: pd.DataFrame) -> dict[str, int]:
        return self._write(Q_USERS, df_to_rows(users), "user")

    def ingest_posts(self, posts: pd.DataFrame) -> dict[str, int]:
        posts = posts.copy()
        posts["content_hash"] = [content_hash(t, ts, u) for t, ts, u in
                                 zip(posts["text"], posts["timestamp"], posts["user_id"])]
        if "is_stub" not in posts.columns:
            posts["is_stub"] = False
        posts["is_stub"] = posts["is_stub"].map(lambda v: str(v).strip().lower() in {"true", "1"})
        cols = ["post_id", "user_id", "text", "timestamp", "language", "conversation_id",
                "dataset_source", "synthetic", "content_hash", "is_stub"]
        return self._write(Q_POSTS, df_to_rows(posts[cols]), "post")

    def ingest_predictions(self, preds: list[Prediction] | pd.DataFrame) -> dict[str, int]:
        if isinstance(preds, pd.DataFrame):
            rows = df_to_rows(preds)
            for r in rows:
                r["labels"] = [x for x in str(r["labels"] or "UNKNOWN").split(";") if x]
        else:
            rows = predictions_to_rows(preds)
        c1 = self._write(Q_PRIMARY_LABEL, rows, "primary-label")
        c2 = self._write(Q_HAS_LABEL, rows, "HAS_LABEL")
        return {k: c1.get(k, 0) + c2.get(k, 0) for k in set(c1) | set(c2)}

    def ingest_interactions(self, interactions: pd.DataFrame) -> dict[str, int]:
        totals: dict[str, int] = {}
        for itype, query in Q_INTERACTIONS.items():
            sub = interactions[interactions["interaction_type"] == itype]
            sub = sub.dropna(subset=REQUIRED_ENDPOINTS[itype])
            for k, v in self._write(query, df_to_rows(sub), itype).items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def ingest_hashtags(self, posts: pd.DataFrame) -> dict[str, int]:
        return self._write(Q_HASHTAGS, df_to_rows(build_hashtag_table(posts)), "hashtag")

    def ingest_follows(self, follows: pd.DataFrame) -> dict[str, int]:
        return self._write(Q_FOLLOWS, df_to_rows(follows), "follow")

    def ingest_user_edges(self, user_edges: pd.DataFrame, batch_size: int = 10000) -> dict[str, int]:
        """User->User edges (millions possible): larger batches, users must exist or become placeholders."""
        rows = df_to_rows(user_edges)
        if not rows:
            return {}
        counters = self.client.write_batches(Q_USER_EDGES, rows, batch_size, {"batch_id": self.batch_id})
        logger.info("Ingested %d user edges: %s", len(rows), counters)
        return counters

    def ingest_dataset(self, ds: CanonicalDataset,
                       predictions: list[Prediction] | pd.DataFrame | None = None) -> dict[str, Any]:
        """Ingest every table in dependency order and record an IngestionBatch node."""
        import json

        report = {"batch_id": self.batch_id, "dataset": ds.name}
        report["users"] = self.ingest_users(ds.users)
        report["posts"] = self.ingest_posts(ds.posts)
        report["hashtags"] = self.ingest_hashtags(ds.posts)
        report["interactions"] = self.ingest_interactions(ds.interactions)
        if len(ds.follows):
            report["follows"] = self.ingest_follows(ds.follows)
        if len(ds.user_edges):
            report["user_edges"] = self.ingest_user_edges(ds.user_edges)
        if predictions is not None:
            report["labels"] = self.ingest_predictions(predictions)
        counts = {"users": len(ds.users), "posts": len(ds.posts),
                  "interactions": len(ds.interactions), "follows": len(ds.follows),
                  "user_edges": len(ds.user_edges)}
        self.client.run_write(Q_BATCH, {"batch_id": self.batch_id, "dataset": ds.name,
                                        "at": utc_now_iso(), "counts_json": json.dumps(counts)})
        return report
