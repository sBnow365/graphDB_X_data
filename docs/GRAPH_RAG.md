# Graph RAG in this project

**RAG** = Retrieval-Augmented Generation: retrieve relevant information first, then let the LLM answer
using it. **Graph RAG** here = the retrieved information comes from the knowledge graph (Cypher
queries, GDS statistics), not from a document search. It does **not** imply a GNN or learned graph
embeddings.

## Flow (`retrieval/graph_rag.py`)
1. **Intent parsing** (rules, `intent_parser.py`): labels (hate/spam/offensive/targeted), dates
   ("ten days before the election", "between 2025-03-01 and 2025-03-05"), community numbers, user ids,
   and an intent such as `community_stats` or `temporal_comparison`.
2. **Retrieval** (`graph_retriever.py`): a trusted template or analysis for known intents; otherwise
   Text-to-Cypher.
3. **Optional GDS step:** community statistics and ranking, window comparisons.
4. **Evidence** (`evidence_formatter.py`): rows → `[E1] (computed) community=2; users=58; hate_share=0.31 …`.
   Texts truncated to 160 chars; notes added (synthetic data, assumptions, causality caveat).
5. **Answer** (`answer_generation.py`): Qwen must cite `[E#]`, say `INSUFFICIENT_EVIDENCE` when needed,
   separate *Interpretation* from computed facts, and never claim coordination or political affiliation.
   The code checks citations: cites of non-existent evidence are flagged as `unknown_citations`.
   With empty evidence the LLM is **not called** at all.

## Supported question types
| Example | Route |
|---|---|
| Which users posted hate-labelled content during the ten days before the election? | template `users_posting_label_in_range` |
| Which communities contain a high proportion of hate-labelled posts? | analysis `community_stats` |
| Which users have high structural influence in community 2? | template `top_users_in_community` |
| Which communities showed increasing hate/spam activity? / How did the network change before, during and after the election? | analysis `temporal_comparison` |
| Which users repeatedly amplified the same posts? | template `repeated_amplifiers` |
| Explain the graph evidence behind community 2 | template `community_evidence` |
| anything else | Text-to-Cypher |

## Without an LLM
`--no-llm` returns the evidence only. The graph-analysis baseline never depends on Qwen.

## Optional extensions (not implemented)
Embeddings of post text stored as node properties + Neo4j vector index for semantic retrieval;
hybrid "find semantically similar posts, then expand their neighbourhood" retrieval. Document the
embedding model, similarity metric and evaluation if you add them.
