"""Detect whether GDS is installed and which procedures exist."""
from __future__ import annotations

import logging
from typing import Any

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.utils.errors import GDSUnavailableError

logger = logging.getLogger(__name__)

REQUIRED_PROCEDURES = [
    "gds.graph.project", "gds.graph.drop", "gds.graph.list", "gds.louvain.stream",
    "gds.louvain.write", "gds.louvain.mutate", "gds.pageRank.write", "gds.degree.write",
]


def gds_info(client: Neo4jClient) -> dict[str, Any]:
    """Return {'version': ..., 'missing': [...]} or raise GDSUnavailableError."""
    try:
        version = client.run_read("RETURN gds.version() AS version")[0]["version"]
    except Exception as exc:
        raise GDSUnavailableError(
            "GDS is not available. Check NEO4J_PLUGINS='[\"graph-data-science\"]' in "
            "docker-compose.yml and the container logs.") from exc
    names = {r["name"] for r in client.run_read("CALL gds.list() YIELD name RETURN name")}
    missing = [p for p in REQUIRED_PROCEDURES if p not in names]
    if missing:
        logger.warning("GDS %s lacks procedures: %s", version, missing)
    return {"version": version, "procedure_count": len(names), "missing": missing,
            "has_betweenness": "gds.betweenness.write" in names}
