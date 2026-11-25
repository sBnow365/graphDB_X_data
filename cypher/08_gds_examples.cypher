// ============ GRAPH DATA SCIENCE (GDS 2.x syntax; verify with RETURN gds.version()) ============
RETURN gds.version() AS gds_version;

// G1: project a small User-User retweet graph (Cypher aggregation projection).
MATCH (a:User)-[:RETWEETED]->(:Post)<-[:POSTED]-(b:User)
WHERE a <> b
WITH a, b, count(*) AS weight
WITH gds.graph.project('demo_retweets', a, b, {relationshipProperties: {weight: weight}},
                       {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.graphName, g.nodeCount, g.relationshipCount;

// G2: list projections (they live in RAM, not on disk).
CALL gds.graph.list() YIELD graphName, nodeCount, relationshipCount, memoryUsage
RETURN graphName, nodeCount, relationshipCount, memoryUsage;

// G3: memory estimate BEFORE running.
CALL gds.louvain.write.estimate('demo_retweets', {writeProperty: 'tmp'})
YIELD requiredMemory RETURN requiredMemory;

// G4: STREAM mode - results returned, nothing stored.
CALL gds.louvain.stream('demo_retweets', {relationshipWeightProperty: 'weight'})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).user_id AS user, communityId ORDER BY communityId LIMIT 20;

// G5: MUTATE mode - result added to the in-memory graph only.
CALL gds.louvain.mutate('demo_retweets', {mutateProperty: 'demo_comm', relationshipWeightProperty: 'weight'})
YIELD communityCount, modularity RETURN communityCount, modularity;

// G6: WRITE mode - result written to User nodes in the database.
CALL gds.pageRank.write('demo_retweets', {writeProperty: 'demo_pagerank'})
YIELD nodePropertiesWritten RETURN nodePropertiesWritten;

// G7: drop the projection to free memory (database data is untouched).
CALL gds.graph.drop('demo_retweets', false) YIELD graphName RETURN graphName;
