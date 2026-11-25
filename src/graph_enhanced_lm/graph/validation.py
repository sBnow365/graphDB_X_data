"""Post-ingestion graph checks: counts, orphans, placeholders, missing labels."""
from __future__ import annotations

from typing import Any

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

CHECKS = {
    "node_counts": "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY l",
    "relationship_counts": "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type",
    "posts_without_author": "MATCH (p:Post) WHERE NOT ()-[:POSTED]->(p) AND NOT coalesce(p.placeholder,false) "
                            "RETURN count(p) AS n",
    "posts_without_timestamp": "MATCH (p:Post) WHERE p.timestamp IS NULL AND NOT coalesce(p.placeholder,false) "
                               "RETURN count(p) AS n",
    "posts_without_label": "MATCH (p:Post) WHERE NOT (p)-[:HAS_LABEL]->() AND NOT coalesce(p.placeholder,false) "
                           "RETURN count(p) AS n",
    "placeholder_users": "MATCH (u:User {placeholder: true}) RETURN count(u) AS n",
    "placeholder_posts": "MATCH (p:Post {placeholder: true}) RETURN count(p) AS n",
    "duplicate_user_ids": "MATCH (u:User) WITH u.user_id AS id, count(*) AS c WHERE c > 1 RETURN count(id) AS n",
    "duplicate_post_ids": "MATCH (p:Post) WITH p.post_id AS id, count(*) AS c WHERE c > 1 RETURN count(id) AS n",
    "label_source_counts": "MATCH (p:Post) WHERE p.primary_label_source IS NOT NULL "
                           "RETURN p.primary_label_source AS source, count(*) AS n ORDER BY source",
}


def validate_graph(client: Neo4jClient) -> dict[str, Any]:
    """Run every check and flag problems (duplicates must be zero)."""
    results = {name: client.run_read(q) for name, q in CHECKS.items()}
    problems = []
    for key in ("duplicate_user_ids", "duplicate_post_ids"):
        if results[key] and results[key][0]["n"]:
            problems.append(f"{key}: {results[key][0]['n']}")
    results["problems"] = problems
    results["ok"] = not problems
    return results
