"""Single source of truth for the graph schema (labels, relationships, provenance).

``schema_as_text()`` is injected into LLM prompts so Text-to-Cypher always sees the
real schema.
"""
from __future__ import annotations

NODE_LABELS = {
    "User": {"key": "user_id", "properties": {
        "user_id": "STRING", "username": "STRING", "dataset_source": "STRING",
        "synthetic": "BOOLEAN", "source_bot_label": "STRING (TwiBot-22 bot/human, NOT hate)",
        "louvain_community": "INTEGER (written by GDS Louvain)", "pagerank": "FLOAT (GDS)",
        "degree": "FLOAT (GDS)", "betweenness": "FLOAT (GDS)", "placeholder": "BOOLEAN",
        "user_hate_label": "STRING HATEFUL|NORMAL (human user-level annotation; Hateful Users dataset only)"}},
    "Post": {"key": "post_id", "properties": {
        "post_id": "STRING", "text": "STRING", "timestamp": "DATETIME (UTC)",
        "language": "STRING", "primary_label": "STRING (one of the taxonomy labels)",
        "labels": "LIST<STRING>", "primary_label_source": "STRING (human|model|synthetic_ground_truth)",
        "classification_confidence": "FLOAT", "classifier_model": "STRING",
        "classifier_version": "STRING", "dataset_source": "STRING", "synthetic": "BOOLEAN"}},
    "Hashtag": {"key": "tag", "properties": {"tag": "STRING (lower-case, no #)"}},
    "ContentLabel": {"key": "name", "properties": {"name": "STRING"}},
    "Community": {"key": "community_key", "properties": {
        "community_key": "STRING (run_id:community_id)", "community_id": "INTEGER", #here key is a unique combo of community_id and run_number
        "run_id": "STRING", "algorithm": "STRING"}},
    "IngestionBatch": {"key": "batch_id", "properties": {"batch_id": "STRING"}},
}

RELATIONSHIPS = [
    # (start, type, end, provenance, properties)
    ("User", "POSTED", "Post", "source-provided", ["timestamp"]),
    ("User", "RETWEETED", "Post", "source-provided", ["interaction_id", "timestamp"]),
    ("Post", "REPLIED_TO", "Post", "source-provided", ["interaction_id", "timestamp"]),
    ("Post", "QUOTED", "Post", "source-provided", ["interaction_id", "timestamp"]),
    ("Post", "MENTIONED", "User", "source-provided", ["interaction_id", "timestamp"]),
    ("User", "FOLLOWS", "User", "source-provided (only if dataset has it)", []),
    ("User", "RETWEETED_USER", "User", "source-provided user-level retweet graph (Hateful Users)", ["weight"]),
    ("Post", "USES_HASHTAG", "Hashtag", "inferred (parsed from text)", []),
    ("Post", "HAS_LABEL", "ContentLabel", "model-inferred or human-annotated",
     ["label_source", "model_name", "model_version", "confidence"]),
    ("User", "MEMBER_OF", "Community", "derived analytical (GDS Louvain)", ["run_id"]),
]

LABEL_VALUES = ["NORMAL", "HATE", "OFFENSIVE", "COMMUNITY_TARGETED", "SPAM", "UNKNOWN"]


def schema_as_text() -> str:
    """Compact, LLM-friendly schema description."""   #fed to Qwen so that it knows how to write cypher
    lines = ["Node labels and properties:"]
    for label, spec in NODE_LABELS.items():
        if label == "IngestionBatch":
            continue
        props = ", ".join(f"{k}: {v}" for k, v in spec["properties"].items())
        lines.append(f"  (:{label} {{{props}}})")
    lines.append("Relationships:")
    for start, rel, end, _prov, props in RELATIONSHIPS:
        p = f" {{{', '.join(props)}}}" if props else ""
        lines.append(f"  (:{start})-[:{rel}{p}]->(:{end})")
    lines.append(f"ContentLabel names / Post.primary_label values: {', '.join(LABEL_VALUES)}")
    lines.append("Timestamps are DATETIME; compare with datetime($start) and datetime($end).")
    lines.append("Community membership: User.louvain_community (integer) after GDS has run.")
    return "\n".join(lines)
