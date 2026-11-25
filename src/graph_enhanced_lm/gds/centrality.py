"""Degree, PageRank and (sampled) betweenness centrality.

Centrality measures STRUCTURAL position (how connected / how much of a bridge a user
is). High centrality is not evidence of wrongdoing.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def degree(client: Neo4jClient, graph_name: str, cfg: dict[str, Any], mode: str = "write") -> Any:
    config = {"relationshipWeightProperty": "weight"}
    if mode == "stream":
        q = ("CALL gds.degree.stream($g, $config) YIELD nodeId, score "
             "RETURN gds.util.asNode(nodeId).user_id AS user_id, score ORDER BY score DESC")
        return pd.DataFrame(client.run_read(q, {"g": graph_name, "config": config}))
    config["writeProperty"] = cfg.get("write_property", "degree")
    q = ("CALL gds.degree.write($g, $config) YIELD nodePropertiesWritten, centralityDistribution "
         "RETURN nodePropertiesWritten, centralityDistribution")
    return client.run_write_query(q, {"g": graph_name, "config": config})[0]


def pagerank(client: Neo4jClient, graph_name: str, cfg: dict[str, Any], mode: str = "write") -> Any:
    config = {"relationshipWeightProperty": "weight",
              "dampingFactor": float(cfg.get("damping_factor", 0.85)),
              "maxIterations": int(cfg.get("max_iterations", 20))}
    if mode == "stream":
        q = ("CALL gds.pageRank.stream($g, $config) YIELD nodeId, score "
             "RETURN gds.util.asNode(nodeId).user_id AS user_id, score ORDER BY score DESC")
        return pd.DataFrame(client.run_read(q, {"g": graph_name, "config": config}))
    config["writeProperty"] = cfg.get("write_property", "pagerank")
    q = ("CALL gds.pageRank.write($g, $config) YIELD nodePropertiesWritten, ranIterations, didConverge "
         "RETURN nodePropertiesWritten, ranIterations, didConverge")
    return client.run_write_query(q, {"g": graph_name, "config": config})[0]


def betweenness(client: Neo4jClient, graph_name: str, cfg: dict[str, Any],
                node_count: int, mode: str = "write") -> Any:
    """Exact betweenness is O(n*m); above ``max_nodes_exact`` nodes a sample is used."""
    config: dict[str, Any] = {}
    if node_count > int(cfg.get("max_nodes_exact", 5000)):
        config["samplingSize"] = int(cfg.get("sampling_size", 500))
        config["samplingSeed"] = 42
        logger.info("Using sampled betweenness (samplingSize=%s)", config["samplingSize"])
    if mode == "stream":
        q = ("CALL gds.betweenness.stream($g, $config) YIELD nodeId, score "
             "RETURN gds.util.asNode(nodeId).user_id AS user_id, score ORDER BY score DESC")
        return pd.DataFrame(client.run_read(q, {"g": graph_name, "config": config}))
    config["writeProperty"] = cfg.get("write_property", "betweenness")
    q = ("CALL gds.betweenness.write($g, $config) YIELD nodePropertiesWritten "
         "RETURN nodePropertiesWritten")
    return client.run_write_query(q, {"g": graph_name, "config": config})[0]
