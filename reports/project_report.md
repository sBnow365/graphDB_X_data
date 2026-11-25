# Graph-Enhanced Language Models for Scientific Literature
### A Graph-RAG and Neo4j Framework for Identifying Candidate Hate/Spam Propagation Communities on X Around Election Periods *(proposed subtitle, pending approval)*

**[Student name] · [Roll number] · B.Tech CSE, [Semester] · [Institute]**
**Supervisor:** [Mentor name] · **Date:** [Submission date]

> Draft status: sections marked *(pending)* require experimental results. Every number in
> `[RESULT: …]` must be copied from `reports/experiment_results.md` or `outputs/metrics/`.

---
## 1. Abstract
Online hate speech and spam spread through networks of users, and election periods are a time of
heightened concern. This project builds a reproducible framework that (i) classifies social-media posts
with a Transformer classifier, (ii) represents users, posts and interactions as a Neo4j knowledge graph,
(iii) detects interaction communities with Louvain and measures structural importance with centrality
algorithms, (iv) compares activity across configurable election windows, and (v) lets analysts ask
natural-language questions answered by Qwen3-8B strictly from retrieved graph evidence (Graph RAG with
safe Text-to-Cypher). The pipeline is validated on a synthetic dataset with planted structure
([RESULT: NMI], [RESULT: modularity]); the classifier is evaluated on HateXplain ([RESULT: macro-F1]);
Text-to-Cypher achieves [RESULT: execution/correctness rates] with 100% rejection of the unsafe-query
suite ([RESULT: confirm]). The system outputs *candidate communities exhibiting hate/spam propagation
patterns* for human review and does not infer intent or coordination.

## 2. Introduction
Context of online harm; why network structure matters (content alone does not show how it spreads);
why LLMs help analysts (natural-language access) and why they need grounding (hallucination).

## 3. Background and motivation
Knowledge graphs and property graphs; Cypher; graph algorithms (community detection, centrality);
Transformer classifiers; LLMs and retrieval augmentation. **Relationship to the title:** the official
title concerns scientific literature; this work applies graph-enhanced language-model techniques to
social-media data, as directed by the supervisor (see docs/PROJECT_OVERVIEW.md). The method transfers
to citation graphs via a new data adapter; scientific papers are not analysed here.

## 4. Problem statement
Given posts, users, interactions and timestamps, identify communities where hate-, offensive- or
spam-labelled content is concentrated or increasing near an election, and provide verifiable
natural-language explanations, without attributing intent to individuals.

## 5. Research objectives
O1 reproducible ingestion into a documented knowledge graph; O2 content classification with measured
accuracy; O3 community detection and ranking with transparent criteria; O4 election-window analysis;
O5 safe natural-language querying grounded in graph evidence; O6 rigorous, honest evaluation.

## 6. Research questions
RQ1 How accurately does a pretrained/fine-tuned classifier label hate/offensive content (HateXplain)?
RQ2 Does Louvain on the interaction projection recover planted communities, and how sensitive is it to edge weights?
RQ3 How do label shares and community activity differ across election windows?
RQ4 How valid, correct and safe is LLM-generated Cypher?
RQ5 How often are Graph-RAG answers fully grounded in retrieved evidence?

## 7. Literature review *(to be written; see docs/LITERATURE_REVIEW_PLAN.md)*
7.1 Hate-speech datasets and classifiers · 7.2 Indian-language resources (HASOC) · 7.3 Bot detection and
social graphs (TwiBot-22) · 7.4 Community detection and centrality · 7.5 RAG and Graph RAG · 7.6 Text-to-query generation · 7.7 GNNs (future work).

## 8. Dataset description
Table 1 (from docs/DATASET_RESEARCH.md): HateXplain, HASOC, TwiBot-22, synthetic: content, labels, IDs,
timestamps, edges, licence/access. Staged strategy and why datasets are not joined.
Synthetic dataset: [RESULT: users/posts/interactions], 5 planted communities, fictional election date 2025-03-15.
TwiBot-22: *(pending access)* sample size, sampling method, time range.

## 9. Methodology
Pipeline figure (README architecture). Canonical schema, validation, classifier, graph, GDS, temporal,
LLM. Design principles: synthetic-first, provenance, read-only LLM access, template-first retrieval.

## 10. System architecture
Module table (`src/graph_enhanced_lm/*`) and data flow; configuration-driven design; Docker deployment.

## 11. Data ingestion and preprocessing
Adapters, ID/timestamp normalisation, deduplication, missing values, label mapping, leakage-free splits,
validation report ([RESULT: validation stats]).

## 12. Content classification
Taxonomy (multi-label), mappings (HateXplain majority vote, HASOC fine labels), backend
([checkpoint name], verified label map), batch inference, caching, provenance.

## 13. Knowledge-graph construction
Schema (Figure/Table from docs/GRAPH_SCHEMA.md), relationship provenance, constraints, idempotent MERGE,
incremental updates, protection of human labels. [RESULT: node/relationship counts].

## 14. Neo4j and Cypher implementation
Template library, parameterised queries, example queries with interpretation (3–4 from the tutorial).

## 15. Graph Data Science
Interaction projection and weights; Louvain; degree, PageRank, betweenness; execution modes; memory.
Community statistics and investigation-priority score (weights as analytical choices).

## 16. Temporal analysis
Timezone-aware windows; window comparison; community activity; causality caveat.

## 17. Qwen3-8B integration
Model facts (licence, context), serving (Colab/Ollama, 4-bit), non-thinking mode, replaceable client.

## 18. Text-to-Cypher
Prompt design, parameter discipline, six safety layers, retry, manual mode.

## 19. Graph RAG
Intent parsing, template-first retrieval, evidence formatting, grounded answering, citation checking.

## 20. Experimental design *(pending)*
E1 classifier on HateXplain test; E2 Louvain recovery + weight sensitivity on synthetic; E3 window
comparison (synthetic; TwiBot-22 if available); E4 Text-to-Cypher benchmark ([N] verified items);
E5 Graph-RAG benchmark with manual scoring; E6 baselines A/B/C.

## 21. Evaluation metrics
Per docs/EVALUATION_PLAN.md.

## 22. Results *(pending: fill only from generated files)*
Table: classifier ([RESULT]). Table: graph metrics ([RESULT]). Figure: daily label counts. Table:
window comparison ([RESULT]). Table: community ranking top-5 ([RESULT]; synthetic). Table: T2C
metrics ([RESULT]). Table: RAG grounding ([RESULT]).

## 23. Discussion
What the results show and do not show; synthetic vs real; domain shift; failure cases of T2C; value of templates.

## 24. Ethical considerations
Per docs/ETHICS_AND_LIMITATIONS.md: harm of false accusations, provenance, privacy, licences, no political profiling.

## 25. Limitations
No dataset with hate labels + network + election; English-centric classifier; first-N sampling;
weight sensitivity; small benchmark; single LLM.

## 26. Future work
HASOC-based Hindi/code-mixed classifier; embedding-based hybrid retrieval; GNN user representations;
temporal community tracking; larger human-verified Text-to-Cypher benchmark; application to
scientific-literature graphs (aligning with the original title).

## 27. Conclusion
*(pending)*

## 28. References
*(Add only verified references; see docs/LITERATURE_REVIEW_PLAN.md.)*
