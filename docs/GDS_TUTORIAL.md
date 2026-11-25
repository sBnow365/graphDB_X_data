# Neo4j Graph Data Science (GDS) Tutorial

## Compatibility (checked 30 Oct 2026)
Neo4j's official matrix pairs **Neo4j 5.26 ↔ GDS 2.13**; newer calendar-versioned releases (2025.x,
2026.x) have their own matching GDS versions. `docker-compose.yml` pins `neo4j:5.26-community` and
lets the official image download the matching GDS plugin (`NEO4J_PLUGINS`). Check with:
```cypher
RETURN gds.version();
```
GDS Community Edition is free; some features (e.g. higher concurrency) are Enterprise-only **[verify
current edition limits in GDS docs]**. Everything in this project uses Community features.

## Three different things (the most important idea)
| | Where it lives | Survives restart? | Created by |
|---|---|---|---|
| **Database** | disk | yes | ingestion (`MERGE`) |
| **Projection** (named graph) | RAM, GDS catalog | **no** | `gds.graph.project(...)` |
| **Algorithm results** | returned (stream), in projection (mutate) or on disk (write) | only if *written* | `gds.<algo>.<mode>` |

Intuition: the database is a library; a projection is a photocopy of only the pages an algorithm
needs, laid out for speed; results are notes you either read aloud (stream), clip to the photocopy
(mutate) or write back into the library books (write).

## Our projection: who interacts with whom
Louvain needs a graph of **users**, but our database connects users through **posts**. The projection
(`src/graph_enhanced_lm/gds/projections.py`) collapses paths into weighted user–user edges:
```
A -RETWEETED-> post <-POSTED- B          ⇒  A — B  (+1.0)
A -POSTED-> reply -REPLIED_TO-> post <-POSTED- B  ⇒  A — B  (+1.0)
A -POSTED-> quote -QUOTED-> post <-POSTED- B      ⇒  A — B  (+1.0)
A -POSTED-> post -MENTIONED-> B           ⇒  A — B  (+0.5)
```
Weights are in `configs/gds.yaml`: they are **analytical choices**. Try different weights and report
whether communities change (sensitivity analysis). Each unordered pair gets one undirected edge
whose weight is the sum. Filters: `--start/--end` (temporal subgraph) and `--labels` (only
interactions on posts with those labels).

```powershell
python scripts/create_gds_projection.py                         # full graph
python scripts/create_gds_projection.py --name hate_only --labels HATE
python scripts/create_gds_projection.py --list
python scripts/create_gds_projection.py --drop hate_only       # frees RAM
```
Before projecting, the script counts interactions and refuses above `max_relationships_guard`.
After projecting it asks GDS for a memory estimate (`gds.louvain.write.estimate`).

## Execution modes
```cypher
// stream: results come back to you, nothing stored
CALL gds.louvain.stream('user_interactions', {relationshipWeightProperty: 'weight'})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).user_id AS user, communityId LIMIT 10;

// mutate: result stored as a property inside the projection (chain algorithms)
CALL gds.louvain.mutate('user_interactions', {mutateProperty: 'c', relationshipWeightProperty: 'weight'})
YIELD communityCount, modularity;

// write: result stored on the User nodes in the database
CALL gds.louvain.write('user_interactions', {writeProperty: 'louvain_community', relationshipWeightProperty: 'weight'})
YIELD communityCount, modularity;
```

## Louvain, intuitively
Start with every user alone. Repeatedly move a user into the neighbouring group that most increases
**modularity**: "how many more edges fall inside groups than we'd expect by chance". Then shrink each
group into a super-node and repeat. Output: an integer community id per user (the number itself
means nothing; 3 is not "worse" than 1). Modularity ranges up to 1; values around 0.3+ usually
indicate clear structure. Louvain can give slightly different results between runs and versions;
report the run id and settings (stored in `outputs/metrics/graph_algorithms_*.json`).

**Caution:** a community = users who interact a lot with each other. Sports fans, families and
newsrooms form communities too. Louvain knows nothing about hate; the hate/spam *shares* come from
the classifier labels of the members' posts.

## Centrality
| Algorithm | Intuition | Use here |
|---|---|---|
| Degree | how many (weighted) interaction partners | activity hubs |
| PageRank | important if connected to important users | structurally influential accounts |
| Betweenness | lies on many shortest paths = bridge between groups | brokers between communities (sampled for >5,000 nodes) |
High centrality is structural position, not intent.

## Community statistics and ranking
`python scripts/run_community_analysis.py` computes, per community: users, posts, hate/spam/offensive
counts and shares, internal vs external retweet edges, and **pre-election growth** (hate+spam posts per
day in the 10→1 window minus the 30→11 window). A weighted sum of min-max normalised components gives
`priority_score` (weights in `configs/gds.yaml`). The score orders communities for **human review**;
there is no validated threshold, and the output says so.

## Synthetic sanity check
On synthetic data the true communities are known (`User.synthetic_gt_community`).
`run_graph_algorithms.py` reports **NMI/ARI** between Louvain and the truth. High values mean the
pipeline works; they say nothing about real data. With the default
`intra_community_interaction_prob: 0.85` the planted communities are very easy (NMI = 1.0 was observed in
a test run); also report a harder setting (e.g. 0.6 in `configs/synthetic.yaml`) to show where Louvain
starts to fail.

## Memory tips
* Docker heap is 2 GB (`docker-compose.yml`). For TwiBot-22 samples raise to 4 GB if your RAM allows.
* Drop projections you no longer need; they hold RAM until dropped or Neo4j restarts.
* Use date/label filters to project smaller subgraphs.
