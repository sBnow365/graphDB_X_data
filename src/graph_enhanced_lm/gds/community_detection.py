"""Louvain community detection in stream / mutate / write mode.

* stream : results are RETURNED to Python; nothing is stored.
* mutate : results are added to the IN-MEMORY projection (for chaining algorithms).
* write  : results are WRITTEN as a property on the User nodes in the database.

A Louvain community is a group of users who interact more with each other than with
outsiders. It is NOT evidence of coordination or malicious intent.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def _cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"relationshipWeightProperty": "weight", "maxLevels": int(cfg.get("max_levels", 10)),
            "tolerance": float(cfg.get("tolerance", 1e-4)),
            "includeIntermediateCommunities": bool(cfg.get("include_intermediate_communities", False))}


def louvain_stream(client: Neo4jClient, graph_name: str, cfg: dict[str, Any]) -> pd.DataFrame:
    q = ("CALL gds.louvain.stream($g, $config) YIELD nodeId, communityId "
         "RETURN gds.util.asNode(nodeId).user_id AS user_id, communityId AS community "
         "ORDER BY community, user_id")
    return pd.DataFrame(client.run_read(q, {"g": graph_name, "config": _cfg(cfg)}))


def louvain_mutate(client: Neo4jClient, graph_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    config = {**_cfg(cfg), "mutateProperty": cfg.get("write_property", "louvain_community")}
    q = ("CALL gds.louvain.mutate($g, $config) YIELD communityCount, modularity, ranLevels "
         "RETURN communityCount, modularity, ranLevels")
    return client.run_write_query(q, {"g": graph_name, "config": config})[0]


def louvain_write(client: Neo4jClient, graph_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    config = {**_cfg(cfg), "writeProperty": cfg.get("write_property", "louvain_community")}
    q = ("CALL gds.louvain.write($g, $config) YIELD communityCount, modularity, ranLevels, "
         "nodePropertiesWritten RETURN communityCount, modularity, ranLevels, nodePropertiesWritten")
    res = client.run_write_query(q, {"g": graph_name, "config": config})[0]
    logger.info("Louvain write: %s", res)
    return res


Q_MATERIALIZE = """
MATCH (u:User) WHERE u[$prop] IS NOT NULL
WITH u, u[$prop] AS cid
MERGE (c:Community {community_key: $run_id + ':' + toString(cid)})
  ON CREATE SET c.community_id = cid, c.run_id = $run_id, c.algorithm = 'louvain',
                c.projection = $graph_name, c.created_at = datetime()
MERGE (u)-[m:MEMBER_OF {run_id: $run_id}]->(c)
RETURN count(DISTINCT c) AS communities, count(m) AS memberships
"""


def materialize_communities(client: Neo4jClient, prop: str, run_id: str,
                            graph_name: str) -> dict[str, Any]:
    """Create (:Community) nodes + MEMBER_OF edges (derived analytical relationships)."""
    return client.run_write_query(Q_MATERIALIZE, {"prop": prop, "run_id": run_id,
                                                  "graph_name": graph_name})[0]
