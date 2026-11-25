# Project Overview

**Official title:** Graph-Enhanced Language Models for Scientific Literature
**Proposed subtitle:** A Graph-RAG and Neo4j Framework for Identifying Candidate Hate/Spam Propagation Communities on X Around Election Periods

## The title/application mismatch
The title refers to *scientific literature* (papers, citations, authors). The mentor's described
application concerns *X/Twitter* posts, users and interactions around elections. These are different
research domains with different data, ethics and evaluation. The student reports that the mentor chose
the X/social-media focus; the official title is nevertheless kept unchanged until the mentor decides.

**How they relate:** the *method* (a knowledge graph + graph algorithms + an LLM that retrieves graph
evidence and answers questions about it) is domain-independent. "Graph-enhanced language models" here
means *an LLM whose answers are grounded in graph-structured evidence*. The code can be ported to
scientific literature by writing a new adapter (Author, Paper, CITES, AUTHORED, Topic) without changing
the GDS, retrieval or LLM layers. This project does **not** analyse scientific papers.

## Clarification Required from Mentor
Please confirm which interpretation is intended:
1. **Apply graph-enhanced language-model techniques to X/social-media data** (what is implemented).
2. **Analyse scientific literature about graph-enhanced language models** (e.g. a citation-graph study).
3. **Combine both** (literature review of graph-enhanced LMs + the X application).
And: should the title be changed, or the subtitle added? See also the checklist in [DATASET_RESEARCH.md](DATASET_RESEARCH.md).

## What the system does, in 15 steps (learning order)
1. **System:** data → labels → Neo4j graph → communities/centrality/time windows → question answering grounded in graph evidence.
2. **Technologies:** Python (glue), pandas (tables), Neo4j (graph storage), Cypher (graph queries), GDS (graph algorithms), a BERT-family classifier (content labels), Qwen3-8B (Cypher writing + explanations), Docker (reproducible database), pytest (tests).
3. **Schema:** Users post Posts; users retweet posts; posts reply to/quote posts and mention users; posts carry labels and hashtags ([GRAPH_SCHEMA.md](GRAPH_SCHEMA.md)).
4. **Data:** synthetic (built in), HateXplain/HASOC (classifier), TwiBot-22 (real graph) ([DATASET_RESEARCH.md](DATASET_RESEARCH.md)).
5. **Rows → graph:** adapters convert each dataset into canonical CSV tables; ingestion turns rows into nodes/relationships with `UNWIND … MERGE` ([DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md)).
6. **Cypher:** pattern matching on the graph ([CYPHER_TUTORIAL.md](CYPHER_TUTORIAL.md)).
7. **Projections:** in-memory user–user interaction graphs for algorithms ([GDS_TUTORIAL.md](GDS_TUTORIAL.md)).
8. **Louvain + statistics:** communities, label shares, internal density, pre-election growth, transparent ranking.
9. **Temporal filtering:** timezone-aware windows around a configured election date ([TEMPORAL_ANALYSIS.md](TEMPORAL_ANALYSIS.md)).
10. **Qwen3-8B role:** writes Cypher for questions templates don't cover, and writes answers *only* from evidence ([QWEN_SETUP.md](QWEN_SETUP.md)).
11. **Text-to-Cypher:** generate → validate → EXPLAIN → read-only execution ([TEXT_TO_CYPHER.md](TEXT_TO_CYPHER.md)).
12. **Graph RAG:** intent → retrieval → evidence `[E#]` → grounded answer ([GRAPH_RAG.md](GRAPH_RAG.md)).
13. **Evaluation:** classifier F1, graph recovery/modularity, T2C validity/correctness/safety, RAG grounding ([EVALUATION_PLAN.md](EVALUATION_PLAN.md)).
14. **Running/debugging:** README stages + [TROUBLESHOOTING.md](TROUBLESHOOTING.md); VS Code F5 configurations.
15. **Report:** `reports/project_report.md`, filled from `reports/experiment_results.md` ([REPORT_OUTLINE.md](REPORT_OUTLINE.md)).

## Key design decisions
| Decision | Reason |
|---|---|
| Synthetic-first | the whole pipeline is testable without any external data or access approval |
| Classifier separate from LLM | specialised, measurable, cheap; LLM labelling would be slow and hard to evaluate |
| Templates before Text-to-Cypher | trusted queries for common questions; LLM only for the long tail |
| No GNN in baseline | Graph RAG uses graph *structure as evidence*, not learned embeddings ([OPTIONAL_GNN_EXTENSION.md](OPTIONAL_GNN_EXTENSION.md)) |
| No vector database | Cypher retrieval suffices for structured questions; embeddings are an optional extension |
| Neo4j 5.26 LTS + GDS 2.13 | officially compatible pair; stable for a semester |
