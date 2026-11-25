# Graph Schema

Source of truth in code: `src/graph_enhanced_lm/graph/schema.py` (also fed to the LLM).

```
                 (:ContentLabel {name})
                         ▲ HAS_LABEL {label_source, model_name, model_version, confidence}
                         │
 (:User)─POSTED {timestamp}─►(:Post {post_id, text, timestamp, labels, primary_label, ...})
   │  ▲                       │   │  └─USES_HASHTAG─►(:Hashtag {tag})
   │  └──────MENTIONED────────┘   └─REPLIED_TO / QUOTED {interaction_id, timestamp}─►(:Post)
   ├─RETWEETED {interaction_id, timestamp}─►(:Post)
   ├─FOLLOWS─►(:User)                       (only when the dataset provides follows)
   └─MEMBER_OF {run_id}─►(:Community {community_key, community_id, run_id, algorithm})
 (:IngestionBatch {batch_id, dataset_source, ingested_at, counts})
```

## Nodes
| Label | Key (unique) | Properties | Notes |
|---|---|---|---|
| User | user_id | username, dataset_source, synthetic, source_bot_label, synthetic_gt_community, louvain_community, pagerank, degree, betweenness, placeholder, created_batch, last_batch | `placeholder=true` = referenced but not in the sample |
| Post | post_id | text, timestamp (DATETIME UTC), language, conversation_id, labels (list), primary_label, primary_label_source, classification_confidence, classifier_model, classifier_version, content_hash, dataset_source, synthetic, placeholder | |
| Hashtag | tag | | lower-case, no `#` |
| ContentLabel | name | | NORMAL, HATE, OFFENSIVE, COMMUNITY_TARGETED, SPAM, UNKNOWN |
| Community | community_key (`run_id:id`) | community_id, run_id, algorithm, projection, created_at | derived; one set per algorithm run |
| IngestionBatch | batch_id | dataset_source, ingested_at, counts | provenance |

## Relationships and provenance
| Relationship | Direction semantics | Provenance |
|---|---|---|
| POSTED | author → post | source-provided |
| RETWEETED | retweeting user → original post | source-provided |
| REPLIED_TO | reply post → replied-to post | source-provided |
| QUOTED | quoting post → quoted post | source-provided |
| MENTIONED | post → mentioned user | source-provided |
| FOLLOWS | follower → followed | source-provided (TwiBot-22 only; never invented) |
| USES_HASHTAG | post → hashtag | inferred (regex on text) |
| HAS_LABEL | post → label | model-inferred / human-annotated / synthetic ground truth (property `label_source`) |
| MEMBER_OF | user → community | derived analytical (GDS Louvain) |

## Example
```
(:User {user_id:'u_0042'})-[:POSTED]->(:Post {post_id:'p_000317', primary_label:'HATE',
   labels:['HATE','COMMUNITY_TARGETED'], primary_label_source:'synthetic_ground_truth', synthetic:true})
(:User {user_id:'u_0007'})-[:RETWEETED {interaction_id:'i_0000123', timestamp:datetime(...)}]->(that post)
```

## Constraints and indexes
See `cypher/01_constraints.cypher` (unique keys) and `cypher/02_indexes.cypher` (timestamp, label,
community, interaction ids).

## Duplicate handling
* Nodes: `MERGE` on the unique key.
* Interaction relationships: `MERGE` on `interaction_id` (so two genuine retweets are two edges, a re-run is not).
* HAS_LABEL: one edge per (post, label, label_source, model_name); stale labels from the *same* source are removed when a post is re-classified.

## Incremental updates
`update_graph.py` sends only new/changed posts (SHA-1 of text+timestamp+author) and new interaction ids,
tracked in `data/processed/<dataset>/_manifest/`. Every write stamps `last_batch`.
Human annotations: `primary_label` is overwritten only if the current source is not `human` or the new one is `human`.

## Schema migration
Additive changes (new property, new relationship type) need no migration: new ingestions populate them.
For renames, write a one-off Cypher script, test it on a copy (`docker compose down` keeps the volume;
use `neo4j-admin database dump` for backups **[verify command for your version]**), and update
`schema.py`, the `.cypher` files, the templates and this document together.
