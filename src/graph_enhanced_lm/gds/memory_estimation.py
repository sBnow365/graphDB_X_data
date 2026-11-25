"""Memory estimation before running algorithms (avoid exhausting the heap)."""
from __future__ import annotations

from typing import Any

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

ESTIMATES = {
    "louvain": "CALL gds.louvain.write.estimate($g, {writeProperty: 'tmp', relationshipWeightProperty: 'weight'}) "
               "YIELD requiredMemory, bytesMin, bytesMax, nodeCount, relationshipCount "
               "RETURN requiredMemory, bytesMin, bytesMax, nodeCount, relationshipCount",
    "pagerank": "CALL gds.pageRank.write.estimate($g, {writeProperty: 'tmp'}) "
                "YIELD requiredMemory, bytesMin, bytesMax RETURN requiredMemory, bytesMin, bytesMax",
    "betweenness": "CALL gds.betweenness.write.estimate($g, {writeProperty: 'tmp'}) "
                   "YIELD requiredMemory, bytesMin, bytesMax RETURN requiredMemory, bytesMin, bytesMax",
}


def estimate(client: Neo4jClient, graph_name: str, algorithm: str) -> dict[str, Any]:
    """Return GDS's own memory estimate for running ``algorithm`` on ``graph_name``."""
    if algorithm not in ESTIMATES:
        raise ValueError(f"No estimate for {algorithm}; options {list(ESTIMATES)}")
    rows = client.run_read(ESTIMATES[algorithm], {"g": graph_name})
    return rows[0] if rows else {}
