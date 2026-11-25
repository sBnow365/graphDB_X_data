# Implementation Roadmap (≈14 weeks)

| Week | Stage | Tasks | Deliverable / check |
|---|---|---|---|
| 1 | A | Setup (README Stage 0–1), read PROJECT_OVERVIEW, email TwiBot-22 authors, ask mentor the checklist | offline demo runs, `pytest` green |
| 2 | B | Docker + Neo4j, ingestion, Cypher tutorial levels 1–4 | validate_graph OK, 15 queries run |
| 3 | B | GDS tutorial, Louvain/PageRank, community ranking | NMI on synthetic recorded |
| 4 | B | Temporal analysis, windows, plots; verify T2C benchmark gold queries | benchmark `status: verified` |
| 5 | D | HateXplain download/preprocess, classifier evaluation (subset on CPU, full on Colab) | classifier metrics JSON |
| 6 | D | Error analysis; optional HASOC | error-analysis notes |
| 7–8 | C | Colab Qwen, check_llm, ask.py, Text-to-Cypher benchmark | T2C metrics |
| 9 | C | Graph RAG evaluation, manual scoring | RAG metrics |
| 10–11 | D | TwiBot-22 (if access): sample, classify, ingest, repeat analysis | real-data results with caveats |
| 12 | E | Sensitivity analysis, baselines, run_baselines | experiment_results.md |
| 13 | E/F | Report writing, figures, final verification | report draft |
| 14 | | Presentation, demo rehearsal (`run_demo.py`) | |
Fallback if TwiBot-22 access does not arrive: present the method on synthetic data + classifier results on HateXplain, clearly stating the limitation.
