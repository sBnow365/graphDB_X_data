// ============ CONTENT-LABEL QUERIES ============
// :param label => 'HATE'

// Q15: hate-labelled posts via the HAS_LABEL relationship (with provenance).
MATCH (p:Post)-[h:HAS_LABEL]->(:ContentLabel {name: 'HATE'})
RETURN p.post_id, h.label_source, h.model_name, h.confidence LIMIT 20;

// Q16: spam-labelled posts via the labels list property.
MATCH (p:Post) WHERE 'SPAM' IN p.labels RETURN p.post_id, p.timestamp LIMIT 20;

// Q17: users who posted content with a given label in a range.
MATCH (u:User)-[:POSTED]->(p:Post)-[:HAS_LABEL]->(:ContentLabel {name: $label})
WHERE p.timestamp >= datetime($start) AND p.timestamp < datetime($end)
RETURN u.user_id, count(DISTINCT p) AS n ORDER BY n DESC LIMIT 20;

// Q18: users who repeatedly amplified (retweeted) the same author.
MATCH (amp:User)-[r:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
WHERE amp <> author
WITH amp, author, count(r) AS times WHERE times >= 3
RETURN amp.user_id, author.user_id, times ORDER BY times DESC LIMIT 20;

// Q19: label provenance - how many labels come from humans vs models.
MATCH (p:Post) RETURN p.primary_label_source AS source, count(*) AS posts;
