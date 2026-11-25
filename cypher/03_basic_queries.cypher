// ============ BASIC QUERIES (paste one at a time into Neo4j Browser) ============
// Q1: 10 users. MATCH finds a pattern; RETURN chooses output; LIMIT caps rows.
MATCH (u:User) RETURN u.user_id, u.username LIMIT 10;

// Q2: 10 posts, newest first. ORDER BY ... DESC sorts descending.
MATCH (p:Post) RETURN p.post_id, p.timestamp, p.primary_label ORDER BY p.timestamp DESC LIMIT 10;

// Q3: who posted what - a relationship pattern (User)-[:POSTED]->(Post).
MATCH (u:User)-[:POSTED]->(p:Post) RETURN u.user_id, p.post_id LIMIT 10;

// Q4: posts per user - count() is an aggregation, grouped by the non-aggregated column.
MATCH (u:User)-[:POSTED]->(p:Post)
RETURN u.user_id AS user, count(p) AS posts ORDER BY posts DESC LIMIT 10;

// Q5: OPTIONAL MATCH keeps users even if they have no retweets (like a LEFT JOIN).
MATCH (u:User)
OPTIONAL MATCH (u)-[r:RETWEETED]->(:Post)
RETURN u.user_id, count(r) AS retweets_made ORDER BY retweets_made DESC LIMIT 10;

// Q6: COLLECT builds a list; slicing [0..3] keeps the first three.
MATCH (u:User)-[:POSTED]->(p:Post)
RETURN u.user_id, collect(p.post_id)[0..3] AS sample_posts LIMIT 5;

// Q7: WITH pipes results into the next part (filter on an aggregate).
MATCH (u:User)-[:POSTED]->(p:Post)
WITH u, count(p) AS posts
WHERE posts >= 10
RETURN u.user_id, posts ORDER BY posts DESC;

// Q8: users whose posts were retweeted the most.
MATCH (:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
RETURN author.user_id, count(r) AS retweets_received ORDER BY retweets_received DESC LIMIT 10;

// Q9: UNWIND turns a list into rows.
UNWIND ['HATE', 'SPAM', 'OFFENSIVE'] AS lab
MATCH (p:Post) WHERE lab IN p.labels
RETURN lab, count(p) AS posts;

// Q10: whole-graph overview: count nodes per label.
MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC;
