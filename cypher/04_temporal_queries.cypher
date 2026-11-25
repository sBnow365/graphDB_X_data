// ============ TEMPORAL QUERIES ============
// In Neo4j Browser set parameters first, e.g.:
// :param start => '2025-03-05T00:00:00+05:30'
// :param end   => '2025-03-15T00:00:00+05:30'
// :param election => '2025-03-15'

// Q11: posts in a date range. datetime() converts the ISO string; [start, end) interval.
MATCH (p:Post)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN p.post_id, p.timestamp, p.primary_label ORDER BY p.timestamp LIMIT 25;

// Q12: posts in the 10 days before the election, computed from a date with duration.
WITH datetime($election + 'T00:00:00+05:30') AS e
MATCH (p:Post)
WHERE p.timestamp >= e - duration({days: 10}) AND p.timestamp < e
RETURN count(p) AS posts_last_10_days;

// Q13: daily counts per label. date() truncates a datetime to its (UTC) day.
MATCH (p:Post)-[:HAS_LABEL]->(l:ContentLabel)
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN date(p.timestamp) AS day, l.name AS label, count(DISTINCT p) AS posts
ORDER BY day, label;

// Q14: retweets per day (relationship timestamps).
MATCH (:User)-[r:RETWEETED]->(:Post)
WHERE r.timestamp >= datetime($start) AND r.timestamp < datetime($end)
RETURN date(r.timestamp) AS day, count(r) AS retweets ORDER BY day;
