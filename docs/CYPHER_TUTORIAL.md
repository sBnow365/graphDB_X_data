# Cypher Tutorial (using this project's graph)

Run these in **Neo4j Browser** (http://localhost:7474) after ingesting the synthetic data, or with
`python scripts/run_cypher.py --query "..."`. Results describe SYNTHETIC data.

## 0. The mental model
Cypher draws patterns with ASCII art:
```
(u:User)-[:POSTED]->(p:Post)
 ^node   ^relationship  ^node        () = node, [] = relationship, -> = direction, :Label / :TYPE
```
`MATCH` finds every place in the graph where the pattern exists; `RETURN` says what to output.
Our schema (details in [GRAPH_SCHEMA.md](GRAPH_SCHEMA.md)):
```
(:User)-[:POSTED]->(:Post)   (:User)-[:RETWEETED]->(:Post)   (:Post)-[:REPLIED_TO|QUOTED]->(:Post)
(:Post)-[:MENTIONED]->(:User) (:Post)-[:USES_HASHTAG]->(:Hashtag) (:Post)-[:HAS_LABEL]->(:ContentLabel)
(:User)-[:MEMBER_OF]->(:Community)   User.louvain_community, User.pagerank (after GDS)
```

### Parameters in Neo4j Browser
Never paste user input into a query string; use `$parameters`. In Browser:
```
:param start => '2025-03-04T18:30:00+00:00'
:param end => '2025-03-14T18:30:00+00:00'
:param label => 'HATE'
:param community_id => 0
:param user_id => 'u_0001'
```
(These two datetimes are 5 Mar 00:00 IST and 15 Mar 00:00 IST: the 10 days before the fictional election.)

---
## Level 1: reading data

**1. List users** (`MATCH`, `RETURN`, `LIMIT`)
```cypher
MATCH (u:User) RETURN u.user_id, u.username LIMIT 10
```
*Interpretation:* ten User nodes. `LIMIT` protects you from printing millions of rows.

**2. List posts, newest first** (`ORDER BY … DESC`)
```cypher
MATCH (p:Post) RETURN p.post_id, p.timestamp, p.primary_label ORDER BY p.timestamp DESC LIMIT 10
```
*Interpretation:* most recent posts and their display label.

**3. Who posted what** (relationship pattern)
```cypher
MATCH (u:User)-[:POSTED]->(p:Post) RETURN u.user_id, p.post_id LIMIT 10
```
*Interpretation:* each row is one authorship edge.

**4. Filter with WHERE**
```cypher
MATCH (p:Post) WHERE p.primary_label = 'SPAM' RETURN p.post_id, p.text LIMIT 10
```
*Interpretation:* posts whose primary label is SPAM. (Multi-label check: example 7.)

**5. Count** (`count`)
```cypher
MATCH (p:Post) RETURN count(p) AS posts
```
*Interpretation:* total Post nodes (including `placeholder` posts from outside a real-data sample).

## Level 2: aggregation

**6. Posts per user** (implicit GROUP BY)
```cypher
MATCH (u:User)-[:POSTED]->(p:Post)
RETURN u.user_id AS user, count(p) AS posts ORDER BY posts DESC LIMIT 10
```
*Interpretation:* the most active accounts. Every non-aggregated column (`u.user_id`) is a grouping key.

**7. Hate-labelled posts** (list membership + label provenance)
```cypher
MATCH (p:Post)-[h:HAS_LABEL]->(:ContentLabel {name: 'HATE'})
RETURN p.post_id, h.label_source, h.model_name, h.confidence LIMIT 20
```
*Interpretation:* `label_source` tells you whether a human, the synthetic ground truth or a model produced the label. Always report this.

**8. Spam-labelled posts via the list property**
```cypher
MATCH (p:Post) WHERE 'SPAM' IN p.labels RETURN count(p) AS spam_posts
```

**9. Users with many retweets of their posts**
```cypher
MATCH (:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
RETURN author.user_id, count(r) AS retweets_received ORDER BY retweets_received DESC LIMIT 10
```
*Interpretation:* reach/amplification received, not intent.

**10. OPTIONAL MATCH** (keep rows without matches, like a LEFT JOIN)
```cypher
MATCH (u:User)
OPTIONAL MATCH (u)-[r:RETWEETED]->(:Post)
RETURN u.user_id, count(r) AS retweets_made ORDER BY retweets_made ASC LIMIT 10
```
*Interpretation:* includes users who never retweeted (count 0). With plain `MATCH` they would disappear.

**11. COLLECT** (aggregate into a list)
```cypher
MATCH (u:User)-[:POSTED]->(p:Post)
RETURN u.user_id, collect(p.post_id)[0..3] AS sample_posts, count(p) AS n LIMIT 5
```

**12. WITH** (filter on an aggregate = SQL HAVING)
```cypher
MATCH (u:User)-[:POSTED]->(p:Post)
WITH u, count(p) AS posts WHERE posts >= 10
RETURN u.user_id, posts ORDER BY posts DESC
```

**13. UNWIND** (list → rows)
```cypher
UNWIND ['HATE','OFFENSIVE','SPAM','COMMUNITY_TARGETED'] AS lab
MATCH (p:Post) WHERE lab IN p.labels
RETURN lab, count(p) AS posts
```

## Level 3: time

**14. Posts in a date range** (parameterised)
```cypher
MATCH (p:Post)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN count(p) AS posts
```
*Interpretation:* half-open interval `[start, end)` so consecutive windows never double-count.

**15. Hate posters around the election** (the project's flagship query)
```cypher
MATCH (u:User)-[:POSTED]->(p:Post)-[:HAS_LABEL]->(:ContentLabel {name: $label})
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN u.user_id, count(DISTINCT p) AS labelled_posts ORDER BY labelled_posts DESC LIMIT 20
```
*Interpretation:* accounts that posted content **labelled** HATE in the window. Labels may be model output; this is not a judgement about the person.

**16. Computing a window from the election date** (`duration`)
```cypher
WITH datetime('2025-03-15T00:00:00+05:30') AS e
MATCH (p:Post) WHERE p.timestamp >= e - duration({days: 10}) AND p.timestamp < e
RETURN count(p) AS posts_last_10_days
```

**17. Daily counts per label** (`date()`)
```cypher
MATCH (p:Post)-[:HAS_LABEL]->(l:ContentLabel)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN date(p.timestamp) AS day, l.name AS label, count(DISTINCT p) AS posts ORDER BY day, label
```
*Note:* `date()` of a UTC datetime gives the UTC day. The Python temporal module converts to your timezone.

## Level 4: paths and subgraphs

**18. Connected users** (one hop through a post)
```cypher
MATCH (u:User {user_id: $user_id})-[:RETWEETED]->(:Post)<-[:POSTED]-(other:User)
RETURN DISTINCT other.user_id LIMIT 25
```

**19. Variable-length paths** (`*1..3`, always bounded!)
```cypher
MATCH path = (u:User {user_id: $user_id})-[:POSTED|RETWEETED*1..3]-(other:User)
WHERE other <> u
RETURN DISTINCT other.user_id, length(path) AS hops ORDER BY hops LIMIT 25
```
*Warning:* unbounded `*` can explode combinatorially on real graphs.

**20. Extract a filtered subgraph** (switch Browser to graph view)
```cypher
MATCH (a:User)-[:POSTED]->(p:Post)<-[r:RETWEETED]-(b:User)
WHERE 'HATE' IN p.labels AND p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN a, p, r, b LIMIT 100
```
Programmatic version: `scripts/extract_subgraph.py` (exports GraphML for Gephi).

**21. Repeated amplification**
```cypher
MATCH (amp:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
WHERE amp <> author
WITH amp, author, count(r) AS times WHERE times >= 3
RETURN amp.user_id, author.user_id, times ORDER BY times DESC LIMIT 20
```
*Interpretation:* frequent re-sharing of one author. Fans do this too; it is not coordination by itself.

**22. High-degree users without GDS**
```cypher
MATCH (u:User) OPTIONAL MATCH (u)-[r:RETWEETED|POSTED]-()
RETURN u.user_id, count(r) AS degree ORDER BY degree DESC LIMIT 10
```

## Level 5: communities (after `run_graph_algorithms.py`)

**23. Community sizes**
```cypher
MATCH (u:User) WHERE u.louvain_community IS NOT NULL
RETURN u.louvain_community AS community, count(*) AS users ORDER BY users DESC
```

**24. Community statistics** (list comprehensions)
```cypher
MATCH (u:User)-[:POSTED]->(p:Post) WHERE u.louvain_community IS NOT NULL
WITH u.louvain_community AS community, collect(DISTINCT u) AS users, collect(p) AS posts
RETURN community, size(users) AS users, size(posts) AS posts,
       round(toFloat(size([x IN posts WHERE 'HATE' IN x.labels])) / size(posts), 3) AS hate_share,
       round(toFloat(size([x IN posts WHERE 'SPAM' IN x.labels])) / size(posts), 3) AS spam_share
ORDER BY hate_share DESC
```
*Interpretation:* a high share makes a community a *candidate exhibiting hate/spam propagation patterns* for review, not a "hate ring".

**25. Supporting evidence for one community**
```cypher
MATCH (u:User {louvain_community: $community_id})-[:POSTED]->(p:Post)
WHERE 'HATE' IN p.labels
OPTIONAL MATCH (rt:User)-[:RETWEETED]->(p)
RETURN p.post_id, u.user_id, p.timestamp, p.primary_label_source, count(rt) AS retweets
ORDER BY retweets DESC LIMIT 20
```

**26. Most central users in a community**
```cypher
MATCH (u:User {louvain_community: $community_id})
RETURN u.user_id, round(u.pagerank, 4) AS pagerank ORDER BY pagerank DESC LIMIT 10
```

**27. Dynamic property access** (how the Python templates avoid string concatenation)
```cypher
:param prop => 'louvain_community'
MATCH (u:User) WHERE u[$prop] IS NOT NULL RETURN u[$prop] AS community, count(*) AS n
```

## Level 6: writing (admin only; never in question-answering mode)

**28. CREATE vs MERGE (incremental, idempotent insert)**
```cypher
MERGE (u:User {user_id: 'demo_user'})
ON CREATE SET u.synthetic = true, u.created_note = 'tutorial'
ON MATCH SET u.seen_again = true
RETURN u
```
Run it twice: the second run finds the node instead of creating a duplicate. `CREATE` would duplicate. Ingestion uses `UNWIND $rows AS row MERGE …` for batches.

**29. SET / REMOVE**
```cypher
MATCH (u:User {user_id: 'demo_user'}) SET u.note = 'updated' REMOVE u.seen_again RETURN u
```

**30. DELETE** ⚠️
```cypher
MATCH (u:User {user_id: 'demo_user'}) DETACH DELETE u
```
`DETACH DELETE` removes the node **and all its relationships**. Always `MATCH` something specific and run the same `MATCH … RETURN count(*)` first. `MATCH (n) DETACH DELETE n` wipes the database.

## Level 7: performance and schema

**31. EXPLAIN / PROFILE**
```cypher
PROFILE MATCH (p:Post) WHERE p.timestamp >= datetime('2025-03-01T00:00:00Z') RETURN count(p)
```
`EXPLAIN` shows the plan without running; `PROFILE` runs it and shows *db hits*. Look for
`NodeIndexSeekByRange` (index used) instead of `NodeByLabelScan` (full scan). The safety
layer uses `EXPLAIN` to ask Neo4j whether a query reads or writes.

**32. Constraints and indexes**
```cypher
SHOW CONSTRAINTS;
SHOW INDEXES;
```
Defined in `cypher/01_constraints.cypher` and `02_indexes.cypher`. A unique constraint on
`User.user_id` makes `MERGE` fast and makes duplicates impossible.

## Exercises
1. Count REPLIED_TO relationships per day in the 10 days before the election.
2. Find hashtags used in both HATE and SPAM posts.
3. For community 0, list users who retweeted posts from *other* communities.
4. Rewrite example 15 for OFFENSIVE posts after the election.
Check your answers against the template library in `src/graph_enhanced_lm/graph/cypher_queries.py`.
