# Evaluation Plan

All numbers must come from scripts run on real outputs. Missing experiments are reported as NOT RUN.

## 1. Content classifier (`evaluate_classifier.py`)
Accuracy, macro precision/recall/F1, weighted F1, per-class P/R/F1, confusion matrix, class distribution.
HateXplain official test split; posts without annotator majority excluded (count them in the report).
Report: checkpoint name + version, max_length, hardware, number of evaluated posts. Add HASOC (Hindi) if
required. Error analysis: inspect 20 false positives and 20 false negatives
(`classifier_*_predictions.csv`); note identity-term and dialect patterns.

## 2. Graph analytics (`run_graph_algorithms.py`, `graph_metrics.py`)
Community count and size distribution, modularity (structure strength, **not** harm), density,
NMI/ARI vs synthetic ground truth at several `intra_community_interaction_prob` values (0.85 is trivially easy; try 0.7, 0.6, 0.5), **sensitivity**: re-run with different weights
(e.g. MENTIONED 0 vs 0.5 vs 1) and with/without reply edges, report NMI between runs.

## 3. Text-to-Cypher (`evaluate_text_to_cypher.py`)
| Metric | Definition |
|---|---|
| syntax/validity rate | passes validator + EXPLAIN |
| execution success | ran without error |
| result correctness | exact multiset match / F1 vs **human-verified** gold query results |
| date-filter correctness | subset of items with dates (category `*date*`) |
| safety | safety-suite accuracy (must be 1.0) + count of unsafe generated queries blocked |
| latency | mean seconds per question (record GPU/endpoint) |
First verify every benchmark gold query by hand and set `status: verified`. Extend to ≥30 items.

## 4. Graph RAG (`evaluate_graph_rag.py`)
| Metric | How |
|---|---|
| intent routing accuracy | automatic vs benchmark labels |
| evidence relevance | manual 0/1 per question |
| answer correctness | manual 0/1 against evidence and gold queries |
| grounding rate | share of answer claims with a valid `[E#]` (manual) |
| unsupported-claim rate | answers with unknown citations (automatic) + manual check |
| missing-evidence handling | questions with empty evidence → must be INSUFFICIENT_EVIDENCE |
Fill `manual_scores` in `outputs/metrics/graph_rag_eval.json`; two raters if possible, report agreement.

## 5. Baseline comparison (`run_baselines.py`)
* **A. Text classification only:** what can be said from labels alone (counts, shares).
* **B. Graph analysis without LLM:** communities, rankings, window changes via scripts/Cypher.
* **C. Graph RAG / Text-to-Cypher:** natural-language access; measured by sections 3–4.
These answer *different* questions; compare on a shared question set (e.g. the 10 RAG benchmark
questions): which configuration can answer each question, with what correctness and effort.
Do **not** claim graph enhancement improves performance unless your numbers show it.
Optional D: a different model size (Qwen3-4B vs 8B) or no-template ablation (`--t2c-only`).
