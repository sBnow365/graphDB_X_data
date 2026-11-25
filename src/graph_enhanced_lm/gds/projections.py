"""Named in-memory graph projections (Cypher aggregation projection).

Database storage (on disk) != GDS projection (in RAM). Algorithms run on the
projection; results only reach the database if you use ``write`` mode.

The projection here is an UNDIRECTED, WEIGHTED User-User interaction graph:
  A retweets B's post   -> A--B (weight RETWEETED)
  A replies to B's post -> A--B (weight REPLIED_TO)
  A quotes B's post     -> A--B (weight QUOTED)
  A's post mentions B   -> A--B (weight MENTIONED)
  A RETWEETED_USER B    -> A--B (weight RETWEETED_USER x edge weight; user-level graphs such
                           as Hateful Users; used only when no date/label filter is given,
                           because these edges carry neither)
Each unordered user pair gets ONE edge whose weight is the sum.
"""
from __future__ import annotations

import logging
from typing import Any

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.utils.errors import GDSUnavailableError

logger = logging.getLogger(__name__)

_TIME = "($start IS NULL OR {v}.timestamp >= datetime($start)) AND ($end IS NULL OR {v}.timestamp < datetime($end))"
_LABEL = "(size($labels) = 0 OR any(l IN coalesce(p.labels, []) WHERE l IN $labels))"

INTERACTION_UNION = f"""
  MATCH (a:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(b:User)
  WHERE a <> b AND {_TIME.format(v='r')} AND {_LABEL}
  RETURN a AS source, b AS target, $w_retweet AS w
  UNION ALL
  MATCH (a:User)-[:POSTED]->(x:Post)-[:REPLIED_TO]->(p:Post)<-[:POSTED]-(b:User)
  WHERE a <> b AND {_TIME.format(v='x')} AND {_LABEL}
  RETURN a AS source, b AS target, $w_reply AS w
  UNION ALL
  MATCH (a:User)-[:POSTED]->(x:Post)-[:QUOTED]->(p:Post)<-[:POSTED]-(b:User)
  WHERE a <> b AND {_TIME.format(v='x')} AND {_LABEL}
  RETURN a AS source, b AS target, $w_quote AS w
  UNION ALL
  MATCH (a:User)-[:POSTED]->(p:Post)-[:MENTIONED]->(b:User)
  WHERE a <> b AND {_TIME.format(v='p')} AND {_LABEL}
  RETURN a AS source, b AS target, $w_mention AS w
  UNION ALL
  MATCH (a:User)-[r:RETWEETED_USER]->(b:User)
  WHERE a <> b AND $start IS NULL AND $end IS NULL AND size($labels) = 0
  RETURN a AS source, b AS target, $w_retweet_user * coalesce(r.weight, 1.0) AS w
"""
ISOLATED = """
  UNION ALL
  MATCH (a:User) RETURN a AS source, null AS target, 0.0 AS w
"""

PROJECT_TEMPLATE = """
CALL {{{union}}}
WITH source, target, w
WITH CASE WHEN target IS NULL OR elementId(source) <= elementId(target) THEN source ELSE target END AS s,
     CASE WHEN target IS NULL OR elementId(source) <= elementId(target) THEN target ELSE source END AS t,
     w
WITH s, t, sum(w) AS weight
WITH gds.graph.project($graph_name, s, t,
       {{relationshipProperties: {{weight: weight}}}}{config}) AS g
RETURN g.graphName AS graph_name, g.nodeCount AS node_count,
       g.relationshipCount AS relationship_count
"""

COUNT_TEMPLATE = "CALL {{{union}}} RETURN count(*) AS interactions"


def _params(graph_name: str, weights: dict[str, float], start: str | None, end: str | None,
            labels: list[str] | None) -> dict[str, Any]:
    return {"graph_name": graph_name, "start": start, "end": end, "labels": labels or [],
            "w_retweet": float(weights.get("RETWEETED", 1.0)),
            "w_reply": float(weights.get("REPLIED_TO", 1.0)),
            "w_quote": float(weights.get("QUOTED", 1.0)),
            "w_mention": float(weights.get("MENTIONED", 0.5)),
            "w_retweet_user": float(weights.get("RETWEETED_USER", 1.0))}


def graph_exists(client: Neo4jClient, graph_name: str) -> bool:
    rows = client.run_read("CALL gds.graph.exists($n) YIELD exists RETURN exists", {"n": graph_name})
    return bool(rows and rows[0]["exists"])


def list_graphs(client: Neo4jClient) -> list[dict[str, Any]]:
    return client.run_read("CALL gds.graph.list() YIELD graphName, nodeCount, relationshipCount, "
                           "memoryUsage RETURN graphName, nodeCount, relationshipCount, memoryUsage")


def drop_graph(client: Neo4jClient, graph_name: str) -> bool:
    """Drop a projection if it exists (failIfMissing=false). Frees RAM; data on disk is untouched."""
    rows = client.run_write_query("CALL gds.graph.drop($n, false) YIELD graphName RETURN graphName",
                                  {"n": graph_name})
    return bool(rows)


def count_interactions(client: Neo4jClient, weights: dict[str, float], start: str | None = None,
                       end: str | None = None, labels: list[str] | None = None) -> int:
    q = COUNT_TEMPLATE.format(union=INTERACTION_UNION)
    return int(client.run_read(q, _params("_", weights, start, end, labels))[0]["interactions"])


def project_interaction_graph(client: Neo4jClient, graph_name: str, weights: dict[str, float],
                              start: str | None = None, end: str | None = None,
                              labels: list[str] | None = None, include_isolated: bool = False,
                              replace: bool = True, max_relationships: int | None = None
                              ) -> dict[str, Any]:
    """Create the named projection. ``start``/``end`` give a TEMPORAL subgraph and
    ``labels`` a CONTENT-filtered subgraph (only interactions on posts with those labels)."""
    if max_relationships is not None:
        n = count_interactions(client, weights, start, end, labels)
        if n > max_relationships:
            raise GDSUnavailableError(f"{n} interactions exceed the guard ({max_relationships}); "
                                      "narrow the date range or raise gds.yaml max_relationships_guard")
    if graph_exists(client, graph_name):
        if not replace:
            logger.info("Re-using existing projection %s", graph_name)
            return {"graph_name": graph_name, "reused": True}
        drop_graph(client, graph_name)
    union = INTERACTION_UNION + (ISOLATED if include_isolated else "")
    params = _params(graph_name, weights, start, end, labels)
    try:
        q = PROJECT_TEMPLATE.format(union=union, config=", {undirectedRelationshipTypes: ['*']}")
        rows = client.run_write_query(q, params)
        undirected = True
    except Exception as exc:  # older GDS without that option
        logger.warning("Undirected projection failed (%s); retrying as directed", exc)
        if graph_exists(client, graph_name):
            drop_graph(client, graph_name)
        rows = client.run_write_query(PROJECT_TEMPLATE.format(union=union, config=""), params)
        undirected = False
    result = dict(rows[0]) if rows else {}
    result.update({"undirected": undirected, "start": start, "end": end, "labels": labels or []})
    logger.info("Projection created: %s", result)
    return result
