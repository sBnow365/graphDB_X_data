"""Extract a filtered subgraph (by label and time) for inspection or export."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

Q_SUBGRAPH = """
MATCH (author:User)-[:POSTED]->(p:Post)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
  AND (size($labels) = 0 OR any(l IN coalesce(p.labels, []) WHERE l IN $labels))
WITH author, p LIMIT $max_posts
OPTIONAL MATCH (rt:User)-[r:RETWEETED]->(p)
OPTIONAL MATCH (p)-[:MENTIONED]->(m:User)
RETURN author.user_id AS author, p.post_id AS post_id, p.primary_label AS label,
       p.timestamp AS timestamp,
       collect(DISTINCT rt.user_id) AS retweeters, collect(DISTINCT m.user_id) AS mentioned
"""


def extract_subgraph(client: Neo4jClient, start: str, end: str, labels: list[str] | None = None,
                     max_posts: int = 2000) -> nx.MultiDiGraph:
    """Return a NetworkX graph of users/posts matching the filters."""
    rows = client.run_read(Q_SUBGRAPH, {"start": start, "end": end, "labels": labels or [],
                                        "max_posts": max_posts})
    g = nx.MultiDiGraph()
    for r in rows:
        g.add_node(r["author"], kind="User")
        g.add_node(r["post_id"], kind="Post", label=r["label"] or "", timestamp=str(r["timestamp"]))
        g.add_edge(r["author"], r["post_id"], type="POSTED")
        for u in r["retweeters"]:
            g.add_node(u, kind="User")
            g.add_edge(u, r["post_id"], type="RETWEETED")
        for u in r["mentioned"]:
            g.add_node(u, kind="User")
            g.add_edge(r["post_id"], u, type="MENTIONED")
    logger.info("Subgraph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
    return g


def export_subgraph(g: nx.MultiDiGraph, path: str | Path) -> Path:
    """Write GraphML (opens in Gephi/Cytoscape) or JSON node-link depending on suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".graphml":
        nx.write_graphml(g, path)
    else:
        data: dict[str, Any] = nx.node_link_data(g, edges="edges")
        path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return path
