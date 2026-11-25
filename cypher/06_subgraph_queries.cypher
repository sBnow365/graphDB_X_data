// ============ SUBGRAPH / PATH QUERIES ============
// :param user_id => 'u_0001'

// Q20: users connected to a user through retweets (one hop, either direction).
MATCH (u:User {user_id: $user_id})-[:RETWEETED]->(:Post)<-[:POSTED]-(other:User)
RETURN DISTINCT other.user_id LIMIT 25;

// Q21: variable-length path - users reachable within 1..3 POSTED/RETWEETED hops.
// Variable-length paths explode quickly: ALWAYS bound the length and LIMIT.
MATCH path = (u:User {user_id: $user_id})-[:POSTED|RETWEETED*1..3]-(other:User)
WHERE other <> u
RETURN DISTINCT other.user_id, length(path) AS hops ORDER BY hops LIMIT 25;

// Q22: a content-filtered temporal subgraph (visualise it in Browser's graph view).
MATCH (a:User)-[:POSTED]->(p:Post)<-[r:RETWEETED]-(b:User)
WHERE 'HATE' IN p.labels AND p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN a, p, r, b LIMIT 100;

// Q23: shortest interaction path between two users (bounded).
// :param other_id => 'u_0002'
MATCH (a:User {user_id: $user_id}), (b:User {user_id: $other_id})
MATCH path = shortestPath((a)-[:POSTED|RETWEETED*..6]-(b))
RETURN [n IN nodes(path) | coalesce(n.user_id, n.post_id)] AS hops;
