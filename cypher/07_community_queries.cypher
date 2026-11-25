// ============ COMMUNITY QUERIES (after Louvain has written u.louvain_community) ============
// :param community_id => 0

// Q24: community sizes.
MATCH (u:User) WHERE u.louvain_community IS NOT NULL
RETURN u.louvain_community AS community, count(*) AS users ORDER BY users DESC;

// Q25: label shares per community (list comprehension [x IN list WHERE ...]).
MATCH (u:User)-[:POSTED]->(p:Post) WHERE u.louvain_community IS NOT NULL
WITH u.louvain_community AS community, collect(p) AS posts
RETURN community, size(posts) AS posts,
       toFloat(size([x IN posts WHERE 'HATE' IN x.labels])) / size(posts) AS hate_share,
       toFloat(size([x IN posts WHERE 'SPAM' IN x.labels])) / size(posts) AS spam_share
ORDER BY hate_share DESC;

// Q26: supporting posts for one community (evidence retrieval).
MATCH (u:User {louvain_community: $community_id})-[:POSTED]->(p:Post)
WHERE 'HATE' IN p.labels
OPTIONAL MATCH (rt:User)-[:RETWEETED]->(p)
RETURN p.post_id, u.user_id, p.timestamp, count(rt) AS retweets ORDER BY retweets DESC LIMIT 20;

// Q27: most central users in a community (after PageRank write).
MATCH (u:User {louvain_community: $community_id})
RETURN u.user_id, u.pagerank ORDER BY u.pagerank DESC LIMIT 10;

// Q28: materialised Community nodes and memberships.
MATCH (u:User)-[:MEMBER_OF]->(c:Community)
RETURN c.community_key, count(u) AS members ORDER BY members DESC LIMIT 10;

// ============ WRITE EXAMPLES (ADMIN / LEARNING ONLY - never in QA mode) ============
// Q29: idempotent insert with MERGE (running twice creates nothing new).
MERGE (u:User {user_id: 'demo_user'})
ON CREATE SET u.synthetic = true, u.created_note = 'tutorial'
RETURN u;

// Q30: SET a property.
MATCH (u:User {user_id: 'demo_user'}) SET u.note = 'updated' RETURN u;

// Q31: DELETE - WARNING: DETACH DELETE removes the node AND all its relationships.
// Always MATCH something specific; never run "MATCH (n) DETACH DELETE n" on real data.
MATCH (u:User {user_id: 'demo_user'}) DETACH DELETE u;

// Q32: PROFILE shows the executed plan and db hits (use EXPLAIN to plan without running).
PROFILE MATCH (p:Post) WHERE p.timestamp >= datetime('2025-03-01T00:00:00Z') RETURN count(p);

// Q33: list constraints and indexes.
SHOW CONSTRAINTS;
