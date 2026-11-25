"""Read-only Cypher used by the dashboard's graph views (simple, parameterised)."""
from __future__ import annotations

from typing import Any

Q_COUNTS = "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC"
Q_REL_COUNTS = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
Q_COMMUNITIES = ("MATCH (u:User) WHERE u[$prop] IS NOT NULL "
                 "RETURN u[$prop] AS community, count(*) AS users ORDER BY users DESC LIMIT $limit")
Q_COMMUNITY_USERS = """
MATCH (u:User) WHERE u[$prop] = $cid
RETURN u.user_id AS id, u.user_hate_label AS hate_label, u.pagerank AS pagerank, u.degree AS degree,
       u.synthetic_gt_community AS planted
ORDER BY coalesce(u.pagerank, 0) DESC, id LIMIT $limit
"""
Q_USER_INFO = """
MATCH (u:User) WHERE u.user_id IN $ids
OPTIONAL MATCH (u)-[:POSTED]->(p:Post) WHERE NOT coalesce(p.placeholder, false)
WITH u, count(p) AS posts, sum(CASE WHEN 'HATE' IN coalesce(p.labels, []) THEN 1 ELSE 0 END) AS hate_posts,
     sum(CASE WHEN 'SPAM' IN coalesce(p.labels, []) THEN 1 ELSE 0 END) AS spam_posts
RETURN u.user_id AS id, u.user_hate_label AS hate_label, u.louvain_community AS community,
       u.pagerank AS pagerank, posts, hate_posts, spam_posts
"""
Q_EDGES_VIA_POSTS = """
MATCH (a:User)-[:RETWEETED]->(:Post)<-[:POSTED]-(b:User)
WHERE a.user_id IN $ids AND b.user_id IN $ids AND a <> b
RETURN a.user_id AS source, b.user_id AS target, count(*) AS weight
"""
Q_EDGES_USER = """
MATCH (a:User)-[r:RETWEETED_USER]->(b:User)
WHERE a.user_id IN $ids AND b.user_id IN $ids AND a <> b
RETURN a.user_id AS source, b.user_id AS target, sum(coalesce(r.weight, 1.0)) AS weight
"""
Q_EGO_NEIGHBOURS = """
MATCH (u:User {user_id: $uid})
OPTIONAL MATCH (u)-[:RETWEETED]->(:Post)<-[:POSTED]-(a:User)
WITH u, collect(DISTINCT a.user_id) AS l1
OPTIONAL MATCH (b:User)-[:RETWEETED]->(:Post)<-[:POSTED]-(u)
WITH u, l1, collect(DISTINCT b.user_id) AS l2
OPTIONAL MATCH (u)-[:RETWEETED_USER]-(c:User)
WITH l1 + l2 + collect(DISTINCT c.user_id) AS ids
UNWIND ids AS id
WITH DISTINCT id WHERE id IS NOT NULL
RETURN id LIMIT $limit
"""
Q_POSTS_OF = """
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE (u.user_id IN $uids OR p.post_id IN $pids) AND NOT coalesce(p.placeholder, false)
RETURN u.user_id AS user_id, p.post_id AS post_id, p.primary_label AS label
LIMIT $limit
"""
Q_RETWEETS_OF_POSTS = """
MATCH (r:User)-[:RETWEETED]->(p:Post) WHERE p.post_id IN $pids
RETURN r.user_id AS user_id, p.post_id AS post_id LIMIT $limit
"""


def run(client: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    return client.run_read(query, params)


def user_subgraph(client: Any, user_ids: list[str]) -> tuple[list[dict], list[dict]]:
    """Nodes (with labels, community, pagerank, post label counts) and user-user edges."""
    if not user_ids:
        return [], []
    nodes = run(client, Q_USER_INFO, ids=user_ids)
    edges = run(client, Q_EDGES_VIA_POSTS, ids=user_ids) + run(client, Q_EDGES_USER, ids=user_ids)
    return nodes, edges


def community_subgraph(client: Any, prop: str, cid: int, limit: int = 60) -> tuple[list[dict], list[dict]]:
    members = run(client, Q_COMMUNITY_USERS, prop=prop, cid=cid, limit=limit)
    return user_subgraph(client, [m["id"] for m in members])


def ego_subgraph(client: Any, user_id: str, limit: int = 40) -> tuple[list[dict], list[dict]]:
    neighbours = [r["id"] for r in run(client, Q_EGO_NEIGHBOURS, uid=user_id, limit=limit)]
    return user_subgraph(client, [user_id] + neighbours)


def evidence_subgraph(client: Any, records: list[dict[str, Any]], limit: int = 150
                      ) -> tuple[list[dict], list[dict]]:
    """Graph around the users/posts cited as evidence: users, their posts, retweeters."""
    uids = sorted({str(v) for r in records for k, v in r.items()
                   if k in ("user_id", "amplifier", "author", "connected_user_id") and v})
    pids = sorted({str(r["post_id"]) for r in records if r.get("post_id")})
    if not uids and not pids:
        return [], []
    posted = run(client, Q_POSTS_OF, uids=uids, pids=pids, limit=limit)
    rts = run(client, Q_RETWEETS_OF_POSTS, pids=pids or [p["post_id"] for p in posted], limit=limit)
    users = sorted(set(uids) | {p["user_id"] for p in posted} | {r["user_id"] for r in rts})[:limit]
    nodes = [dict(n, kind="User") for n in run(client, Q_USER_INFO, ids=users)]
    nodes += [{"id": p["post_id"], "kind": "Post", "label": p["label"]} for p in posted]
    edges = [{"source": p["user_id"], "target": p["post_id"], "type": "POSTED"} for p in posted]
    edges += [{"source": r["user_id"], "target": r["post_id"], "type": "RETWEETED"} for r in rts]
    return nodes, edges
