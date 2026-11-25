"""Trusted, hand-written, parameterised Cypher templates.

These are the preferred retrieval path for Graph RAG (the LLM-generated path is the
fallback). Every template takes parameters; nothing is string-concatenated.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryTemplate:
    name: str
    description: str
    cypher: str
    params: tuple[str, ...] = field(default_factory=tuple)


TEMPLATES: dict[str, QueryTemplate] = {}


def _t(name: str, description: str, cypher: str, *params: str) -> None:
    TEMPLATES[name] = QueryTemplate(name, description, cypher.strip(), tuple(params))


_t("graph_overview", "Count nodes per label.",
   """
MATCH (n) UNWIND labels(n) AS label
RETURN label, count(*) AS count ORDER BY count DESC
""")

_t("posts_in_range", "Posts between two datetimes.",
   """
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN p.post_id AS post_id, u.user_id AS user_id, p.timestamp AS timestamp,
       p.primary_label AS label
ORDER BY p.timestamp LIMIT $limit
""", "start", "end", "limit")

_t("users_posting_label_in_range",
   "Users who posted content with a given label in a date range, with post counts.",
   """
MATCH (u:User)-[:POSTED]->(p:Post)-[:HAS_LABEL]->(l:ContentLabel {name: $label})
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN u.user_id AS user_id, count(DISTINCT p) AS labelled_posts,
       collect(DISTINCT p.post_id)[0..5] AS example_post_ids
ORDER BY labelled_posts DESC, user_id LIMIT $limit
""", "label", "start", "end", "limit")

_t("label_counts_in_range", "Number of posts per label in a date range.",
   """
MATCH (p:Post)-[:HAS_LABEL]->(l:ContentLabel)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN l.name AS label, count(DISTINCT p) AS posts ORDER BY posts DESC
""", "start", "end")

_t("posts_per_user", "Most active users by number of posts.",
   """
MATCH (u:User)-[:POSTED]->(p:Post)
RETURN u.user_id AS user_id, count(p) AS posts ORDER BY posts DESC, user_id LIMIT $limit
""", "limit")

_t("most_retweeted_users", "Users whose posts received the most retweets.",
   """
MATCH (:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
RETURN author.user_id AS user_id, count(r) AS retweets_received,
       count(DISTINCT p) AS posts_retweeted
ORDER BY retweets_received DESC, user_id LIMIT $limit
""", "limit")

_t("repeated_amplifiers",
   "Users who retweeted the same author's posts at least $min_count times.",
   """
MATCH (amp:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
WHERE amp <> author
WITH amp, author, count(r) AS times, collect(DISTINCT p.post_id)[0..5] AS example_post_ids
WHERE times >= $min_count
RETURN amp.user_id AS amplifier, author.user_id AS author, times, example_post_ids
ORDER BY times DESC, amplifier LIMIT $limit
""", "min_count", "limit")

_t("connected_users", "Users directly interacting with a given user (any direction).",
   """
MATCH (u:User {user_id: $user_id})
OPTIONAL MATCH (u)-[:RETWEETED]->(:Post)<-[:POSTED]-(a:User)
OPTIONAL MATCH (b:User)-[:RETWEETED]->(:Post)<-[:POSTED]-(u)
OPTIONAL MATCH (u)-[:POSTED]->(:Post)-[:MENTIONED]->(c:User)
WITH u, collect(DISTINCT a.user_id) + collect(DISTINCT b.user_id) + collect(DISTINCT c.user_id) AS ids
UNWIND ids AS other
WITH DISTINCT other WHERE other IS NOT NULL AND other <> $user_id
RETURN other AS connected_user_id ORDER BY other LIMIT $limit
""", "user_id", "limit")

_t("community_label_stats",
   "Per-community user/post counts and label shares (needs a community property).",
   """
MATCH (u:User) WHERE u[$community_property] IS NOT NULL
WITH u[$community_property] AS community, collect(u) AS members
UNWIND members AS m
OPTIONAL MATCH (m)-[:POSTED]->(p:Post) WHERE NOT coalesce(p.placeholder, false)
WITH community, size(members) AS users, collect(DISTINCT p) AS posts
WITH community, users, size(posts) AS n_posts,
     size([x IN posts WHERE 'HATE' IN coalesce(x.labels, [])]) AS hate_posts,
     size([x IN posts WHERE 'SPAM' IN coalesce(x.labels, [])]) AS spam_posts,
     size([x IN posts WHERE 'OFFENSIVE' IN coalesce(x.labels, [])]) AS offensive_posts,
     size([x IN posts WHERE 'COMMUNITY_TARGETED' IN coalesce(x.labels, [])]) AS targeted_posts
WHERE users >= $min_users
RETURN community, users, n_posts AS posts, hate_posts, spam_posts, offensive_posts, targeted_posts,
       CASE WHEN n_posts = 0 THEN 0.0 ELSE toFloat(hate_posts) / n_posts END AS hate_share,
       CASE WHEN n_posts = 0 THEN 0.0 ELSE toFloat(spam_posts) / n_posts END AS spam_share,
       CASE WHEN n_posts = 0 THEN 0.0 ELSE toFloat(offensive_posts) / n_posts END AS offensive_share
ORDER BY hate_share DESC, community
""", "community_property", "min_users")

_t("community_user_label_stats",
   "Per-community counts of HUMAN-annotated hateful / normal users (Hateful Users dataset).",
   """
MATCH (u:User) WHERE u[$community_property] IS NOT NULL
WITH u[$community_property] AS community, count(u) AS users,
     sum(CASE WHEN u.user_hate_label = 'HATEFUL' THEN 1 ELSE 0 END) AS hateful_users,
     sum(CASE WHEN u.user_hate_label = 'NORMAL' THEN 1 ELSE 0 END) AS normal_users
WITH community, users, hateful_users, normal_users, hateful_users + normal_users AS annotated_users
WHERE users >= $min_users
RETURN community, users, hateful_users, normal_users, annotated_users,
       CASE WHEN annotated_users = 0 THEN 0.0 ELSE toFloat(hateful_users) / annotated_users END AS hateful_user_share
ORDER BY hateful_user_share DESC, community
""", "community_property", "min_users")

_t("hateful_user_homophily",
   "Retweet edges between annotated users by label pair (who retweets whom).",
   """
MATCH (a:User)-[r:RETWEETED_USER]->(b:User)
WHERE a.user_hate_label IS NOT NULL AND b.user_hate_label IS NOT NULL
RETURN a.user_hate_label AS retweeter, b.user_hate_label AS retweeted, count(r) AS edges
ORDER BY retweeter, retweeted
""")

_t("community_internal_edges", "Interaction edges inside vs leaving each community.",
   """
MATCH (a:User)-[:RETWEETED]->(:Post)<-[:POSTED]-(b:User)
WHERE a[$community_property] IS NOT NULL AND b[$community_property] IS NOT NULL AND a <> b
WITH a[$community_property] AS community,
     sum(CASE WHEN a[$community_property] = b[$community_property] THEN 1 ELSE 0 END) AS internal,
     count(*) AS total
RETURN community, internal, total - internal AS external,
       CASE WHEN total = 0 THEN 0.0 ELSE toFloat(internal) / total END AS internal_ratio
ORDER BY community
""", "community_property")

_t("community_internal_user_edges", "Internal vs external RETWEETED_USER edges per community.",
   """
MATCH (a:User)-[:RETWEETED_USER]->(b:User)
WHERE a[$community_property] IS NOT NULL AND b[$community_property] IS NOT NULL
WITH a[$community_property] AS community,
     sum(CASE WHEN a[$community_property] = b[$community_property] THEN 1 ELSE 0 END) AS internal,
     count(*) AS total
RETURN community, internal, total - internal AS external,
       CASE WHEN total = 0 THEN 0.0 ELSE toFloat(internal) / total END AS internal_ratio
ORDER BY community
""", "community_property")

_t("community_evidence",
   "Supporting posts for one community, optionally filtered by label and date range.",
   """
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE u[$community_property] = $community_id
  AND ($label IS NULL OR $label IN coalesce(p.labels, []))
  AND ($start IS NULL OR p.timestamp >= datetime($start))
  AND ($end IS NULL OR p.timestamp < datetime($end))
OPTIONAL MATCH (rt:User)-[:RETWEETED]->(p)
RETURN p.post_id AS post_id, u.user_id AS user_id, p.timestamp AS timestamp,
       p.primary_label AS label, p.primary_label_source AS label_source,
       p.classification_confidence AS confidence, count(rt) AS retweets, p.text AS text
ORDER BY retweets DESC, timestamp LIMIT $limit
""", "community_property", "community_id", "label", "start", "end", "limit")

_t("top_users_in_community", "Highest-scoring users in a community by a centrality property.",
   """
MATCH (u:User) WHERE u[$community_property] = $community_id AND u[$score_property] IS NOT NULL
RETURN u.user_id AS user_id, u[$score_property] AS score
ORDER BY score DESC, user_id LIMIT $limit
""", "community_property", "community_id", "score_property", "limit")

_t("high_degree_users", "Users with the most interaction relationships (no GDS needed).",
   """
MATCH (u:User)
OPTIONAL MATCH (u)-[r:RETWEETED|POSTED]-()
WITH u, count(r) AS degree
RETURN u.user_id AS user_id, degree ORDER BY degree DESC, user_id LIMIT $limit
""", "limit")

_t("daily_label_counts", "Daily number of posts per label in a date range.",
   """
MATCH (p:Post)-[:HAS_LABEL]->(l:ContentLabel)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN date(p.timestamp) AS day, l.name AS label, count(DISTINCT p) AS posts
ORDER BY day, label
""", "start", "end")

_t("community_activity_in_range",
   "Per-community post counts and label counts in a date range.",
   """
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE u[$community_property] IS NOT NULL
  AND p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN u[$community_property] AS community, count(p) AS posts,
       size([x IN collect(p) WHERE 'HATE' IN coalesce(x.labels, [])]) AS hate_posts,
       size([x IN collect(p) WHERE 'SPAM' IN coalesce(x.labels, [])]) AS spam_posts
ORDER BY community
""", "community_property", "start", "end")


def get_template(name: str) -> QueryTemplate:
    if name not in TEMPLATES:
        raise KeyError(f"Unknown query template '{name}'. Available: {sorted(TEMPLATES)}")
    return TEMPLATES[name]
