# Turning results into the report

1. Run Stages 1–5 of the README; keep `outputs/metrics/*.json` and figures.
2. `python scripts/run_baselines.py` → `reports/experiment_results.md` (real numbers only).
3. Copy numbers into `reports/project_report.md` placeholders `[RESULT: …]`; every number must trace to a file.
4. Figures: `outputs/figures/daily_label_counts.png`, confusion matrix, a Neo4j Browser screenshot of a
   community subgraph (synthetic), the architecture diagram from the README.
5. Tables: dataset comparison (DATASET_RESEARCH), schema (GRAPH_SCHEMA), metrics (EVALUATION_PLAN).
6. State clearly what was synthetic, what was real, which labels were model-generated, and which
   sections are planned/pending.
