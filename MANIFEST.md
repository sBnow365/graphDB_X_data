<!-- # MANIFEST

Updated 31 Oct 2026. 183 files.

## Verification actually performed (in a Linux sandbox, Python 3.12)
* `python -m py_compile` on every .py file: OK
* All YAML configs/benchmarks and the Colab notebook JSON parse: OK
* Unit tests: **71 passed, 0 failed** (after the alternative-dataset update); 4 integration tests skipped (need Neo4j). Executed with a minimal
  pytest-compatible runner because pytest could not be installed in the sandbox: run real `pytest` on your machine.
* Scripts executed end-to-end offline: generate_synthetic_data, preprocess_dataset, classify_posts,
  validate_dataset, run_temporal_analysis --source csv, run_demo --offline, evaluate_text_to_cypher --safety-only
  (20/20 correct), evaluate_graph_rag --routing-only (10/10), run_baselines.

## NOT verified (no Neo4j / GPU / internet in the sandbox)
Neo4j ingestion, all GDS procedure calls, EXPLAIN checks against a server, Text-to-Cypher against a real
model, the Colab notebook, the Transformer classifier, HateXplain download, HASOC/TwiBot-22 adapters on real
files (tested only on small fixtures), Windows PowerShell commands (written carefully, not executed).

## Optional / not created
notebooks/01-03 exploration notebooks (use the scripts instead), GNN code, X API adapter, vector search.

## Files
- `.env.example`
- `.gitignore`
- `.vscode/extensions.json`
- `.vscode/launch.json`
- `.vscode/settings.json`
- `MANIFEST.md`
- `Makefile`
- `README.md`
- `benchmarks/cypher_safety_suite.yaml`
- `benchmarks/graph_rag_benchmark.yaml`
- `benchmarks/text_to_cypher_benchmark.yaml`
- `configs/classifier.yaml`
- `configs/dataset.yaml`
- `configs/dataset_mappings/hasoc.yaml`
- `configs/dataset_mappings/hateful_users.yaml`
- `configs/dataset_mappings/hatexplain.yaml`
- `configs/dataset_mappings/russian_trolls.yaml`
- `configs/dataset_mappings/synthetic.yaml`
- `configs/dataset_mappings/twibot22.yaml`
- `configs/evaluation.yaml`
- `configs/gds.yaml`
- `configs/label_taxonomy.yaml`
- `configs/llm.yaml`
- `configs/neo4j.yaml`
- `configs/project.yaml`
- `configs/synthetic.yaml`
- `configs/temporal.yaml`
- `cypher/01_constraints.cypher`
- `cypher/02_indexes.cypher`
- `cypher/03_basic_queries.cypher`
- `cypher/04_temporal_queries.cypher`
- `cypher/05_hate_spam_queries.cypher`
- `cypher/06_subgraph_queries.cypher`
- `cypher/07_community_queries.cypher`
- `cypher/08_gds_examples.cypher`
- `data/README.md`
- `docker-compose.yml`
- `docs/ALTERNATIVE_DATASETS.md`
- `docs/CYPHER_TUTORIAL.md`
- `docs/DATASET_CREATION_GUIDE.md`
- `docs/DATASET_RESEARCH.md`
- `docs/DATA_DICTIONARY.md`
- `docs/ETHICS_AND_LIMITATIONS.md`
- `docs/EVALUATION_PLAN.md`
- `docs/GDS_TUTORIAL.md`
- `docs/GRAPH_RAG.md`
- `docs/GRAPH_SCHEMA.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/LITERATURE_REVIEW_PLAN.md`
- `docs/ONTOLOGY_AND_RDF.md`
- `docs/OPTIONAL_GNN_EXTENSION.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/QWEN_SETUP.md`
- `docs/REPORT_OUTLINE.md`
- `docs/TEMPORAL_ANALYSIS.md`
- `docs/TEXT_TO_CYPHER.md`
- `docs/TROUBLESHOOTING.md`
- `docs/X_API_OPTIONAL.md`
- `notebooks/colab_qwen_server.ipynb`
- `pyproject.toml`
- `reports/experiment_results.md`
- `reports/project_report.md`
- `requirements-classifier.txt`
- `requirements-llm-local.txt`
- `requirements.txt`
- `scripts/_bootstrap.py`
- `scripts/analyze_hateful_users.py`
- `scripts/ask.py`
- `scripts/check_env.py`
- `scripts/check_llm.py`
- `scripts/classify_posts.py`
- `scripts/create_gds_projection.py`
- `scripts/download_dataset.py`
- `scripts/evaluate_classifier.py`
- `scripts/evaluate_graph_rag.py`
- `scripts/evaluate_text_to_cypher.py`
- `scripts/extract_subgraph.py`
- `scripts/generate_synthetic_data.py`
- `scripts/ingest_graph.py`
- `scripts/initialize_neo4j.py`
- `scripts/inspect_dataset.py`
- `scripts/preprocess_dataset.py`
- `scripts/run_baselines.py`
- `scripts/run_community_analysis.py`
- `scripts/run_cypher.py`
- `scripts/run_demo.py`
- `scripts/run_graph_algorithms.py`
- `scripts/run_temporal_analysis.py`
- `scripts/train_classifier.py`
- `scripts/update_graph.py`
- `scripts/validate_dataset.py`
- `scripts/validate_graph.py`
- `src/graph_enhanced_lm/__init__.py`
- `src/graph_enhanced_lm/classification/__init__.py`
- `src/graph_enhanced_lm/classification/base.py`
- `src/graph_enhanced_lm/classification/evaluation.py`
- `src/graph_enhanced_lm/classification/factory.py`
- `src/graph_enhanced_lm/classification/label_mapping.py`
- `src/graph_enhanced_lm/classification/source_label_classifier.py`
- `src/graph_enhanced_lm/classification/transformer_classifier.py`
- `src/graph_enhanced_lm/config.py`
- `src/graph_enhanced_lm/data/__init__.py`
- `src/graph_enhanced_lm/data/dataset_adapters.py`
- `src/graph_enhanced_lm/data/loaders.py`
- `src/graph_enhanced_lm/data/preprocessing.py`
- `src/graph_enhanced_lm/data/schemas.py`
- `src/graph_enhanced_lm/data/splitting.py`
- `src/graph_enhanced_lm/data/synthetic.py`
- `src/graph_enhanced_lm/data/validators.py`
- `src/graph_enhanced_lm/evaluation/__init__.py`
- `src/graph_enhanced_lm/evaluation/benchmark.py`
- `src/graph_enhanced_lm/evaluation/graph_metrics.py`
- `src/graph_enhanced_lm/evaluation/llm_metrics.py`
- `src/graph_enhanced_lm/evaluation/metrics.py`
- `src/graph_enhanced_lm/evaluation/user_label_analysis.py`
- `src/graph_enhanced_lm/gds/__init__.py`
- `src/graph_enhanced_lm/gds/availability.py`
- `src/graph_enhanced_lm/gds/centrality.py`
- `src/graph_enhanced_lm/gds/community_detection.py`
- `src/graph_enhanced_lm/gds/memory_estimation.py`
- `src/graph_enhanced_lm/gds/projections.py`
- `src/graph_enhanced_lm/gds/statistics.py`
- `src/graph_enhanced_lm/graph/__init__.py`
- `src/graph_enhanced_lm/graph/constraints.py`
- `src/graph_enhanced_lm/graph/cypher_queries.py`
- `src/graph_enhanced_lm/graph/incremental_updates.py`
- `src/graph_enhanced_lm/graph/ingestion.py`
- `src/graph_enhanced_lm/graph/neo4j_client.py`
- `src/graph_enhanced_lm/graph/schema.py`
- `src/graph_enhanced_lm/graph/subgraph_extraction.py`
- `src/graph_enhanced_lm/graph/validation.py`
- `src/graph_enhanced_lm/llm/__init__.py`
- `src/graph_enhanced_lm/llm/answer_generation.py`
- `src/graph_enhanced_lm/llm/backends/__init__.py`
- `src/graph_enhanced_lm/llm/backends/hf_transformers.py`
- `src/graph_enhanced_lm/llm/backends/mock.py`
- `src/graph_enhanced_lm/llm/backends/openai_compatible.py`
- `src/graph_enhanced_lm/llm/base.py`
- `src/graph_enhanced_lm/llm/prompts.py`
- `src/graph_enhanced_lm/llm/query_validation.py`
- `src/graph_enhanced_lm/llm/qwen_client.py`
- `src/graph_enhanced_lm/llm/text_to_cypher.py`
- `src/graph_enhanced_lm/pipeline.py`
- `src/graph_enhanced_lm/retrieval/__init__.py`
- `src/graph_enhanced_lm/retrieval/evidence_formatter.py`
- `src/graph_enhanced_lm/retrieval/graph_rag.py`
- `src/graph_enhanced_lm/retrieval/graph_retriever.py`
- `src/graph_enhanced_lm/retrieval/intent_parser.py`
- `src/graph_enhanced_lm/temporal/__init__.py`
- `src/graph_enhanced_lm/temporal/date_filters.py`
- `src/graph_enhanced_lm/temporal/election_windows.py`
- `src/graph_enhanced_lm/temporal/trend_analysis.py`
- `src/graph_enhanced_lm/utils/__init__.py`
- `src/graph_enhanced_lm/utils/errors.py`
- `src/graph_enhanced_lm/utils/logging_config.py`
- `src/graph_enhanced_lm/utils/reproducibility.py`
- `tasks.ps1`
- `tests/conftest.py`
- `tests/fixtures/hasoc/sample.tsv`
- `tests/fixtures/hateful_users/users.edges`
- `tests/fixtures/hateful_users/users_neighborhood_anon.csv`
- `tests/fixtures/hatexplain/dataset.json`
- `tests/fixtures/hatexplain/post_id_divisions.json`
- `tests/fixtures/russian_trolls/tweets.csv`
- `tests/fixtures/russian_trolls/users.csv`
- `tests/fixtures/twibot22/edge.csv`
- `tests/fixtures/twibot22/label.csv`
- `tests/fixtures/twibot22/tweet_0.json`
- `tests/fixtures/twibot22/user.json`
- `tests/test_adapters.py`
- `tests/test_alternative_datasets.py`
- `tests/test_cypher_validation.py`
- `tests/test_dataset_validation.py`
- `tests/test_end_to_end.py`
- `tests/test_env_file.py`
- `tests/test_graph_ingestion.py`
- `tests/test_graph_rag_mock.py`
- `tests/test_intent_and_evidence.py`
- `tests/test_label_mapping.py`
- `tests/test_metrics_and_scoring.py`
- `tests/test_preprocessing.py`
- `tests/test_synthetic_data.py`
- `tests/test_temporal_filters.py` -->



# MANIFEST

Updated 31 Oct 2026. 194 files.

## Verification actually performed (in a Linux sandbox, Python 3.12)
* `python -m py_compile` on every .py file: OK
* All YAML configs/benchmarks and the Colab notebook JSON parse: OK
* Unit tests: **76 passed, 0 failed** (after the dashboard update)
  pytest-compatible runner because pytest could not be installed in the sandbox: run real `pytest` on your machine.
* Scripts executed end-to-end offline: generate_synthetic_data, preprocess_dataset, classify_posts,
  validate_dataset, run_temporal_analysis --source csv, run_demo --offline, evaluate_text_to_cypher --safety-only
  (20/20 correct), evaluate_graph_rag --routing-only (10/10), run_baselines.

* Dashboard: every page executed with a stand-in Streamlit module and a fake Neo4j client (0 failures);
  the real Streamlit rendering, pyvis drawings and plotly charts were not run in the sandbox.

## NOT verified (no Neo4j / GPU / internet in the sandbox)
Neo4j ingestion, all GDS procedure calls, EXPLAIN checks against a server, Text-to-Cypher against a real
model, the Colab notebook, the Transformer classifier, HateXplain download, HASOC/TwiBot-22 adapters on real
files (tested only on small fixtures), Windows PowerShell commands (written carefully, not executed).

## Optional / not created
notebooks/01-03 exploration notebooks (use the scripts instead), GNN code, X API adapter, vector search.

## Files
- `.env.example`
- `.gitignore`
- `.streamlit/config.toml`
- `.vscode/extensions.json`
- `.vscode/launch.json`
- `.vscode/settings.json`
- `MANIFEST.md`
- `Makefile`
- `README.md`
- `app/streamlit_app.py`
- `benchmarks/cypher_safety_suite.yaml`
- `benchmarks/graph_rag_benchmark.yaml`
- `benchmarks/text_to_cypher_benchmark.yaml`
- `configs/classifier.yaml`
- `configs/dataset.yaml`
- `configs/dataset_mappings/hasoc.yaml`
- `configs/dataset_mappings/hateful_users.yaml`
- `configs/dataset_mappings/hatexplain.yaml`
- `configs/dataset_mappings/russian_trolls.yaml`
- `configs/dataset_mappings/synthetic.yaml`
- `configs/dataset_mappings/twibot22.yaml`
- `configs/evaluation.yaml`
- `configs/gds.yaml`
- `configs/label_taxonomy.yaml`
- `configs/llm.yaml`
- `configs/neo4j.yaml`
- `configs/project.yaml`
- `configs/synthetic.yaml`
- `configs/temporal.yaml`
- `cypher/01_constraints.cypher`
- `cypher/02_indexes.cypher`
- `cypher/03_basic_queries.cypher`
- `cypher/04_temporal_queries.cypher`
- `cypher/05_hate_spam_queries.cypher`
- `cypher/06_subgraph_queries.cypher`
- `cypher/07_community_queries.cypher`
- `cypher/08_gds_examples.cypher`
- `data/README.md`
- `docker-compose.yml`
- `docs/ALTERNATIVE_DATASETS.md`
- `docs/CYPHER_TUTORIAL.md`
- `docs/DASHBOARD.md`
- `docs/DATASET_CREATION_GUIDE.md`
- `docs/DATASET_RESEARCH.md`
- `docs/DATA_DICTIONARY.md`
- `docs/ETHICS_AND_LIMITATIONS.md`
- `docs/EVALUATION_PLAN.md`
- `docs/GDS_TUTORIAL.md`
- `docs/GRAPH_RAG.md`
- `docs/GRAPH_SCHEMA.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/LITERATURE_REVIEW_PLAN.md`
- `docs/ONTOLOGY_AND_RDF.md`
- `docs/OPTIONAL_GNN_EXTENSION.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/QWEN_SETUP.md`
- `docs/REPORT_OUTLINE.md`
- `docs/TEMPORAL_ANALYSIS.md`
- `docs/TEXT_TO_CYPHER.md`
- `docs/TROUBLESHOOTING.md`
- `docs/X_API_OPTIONAL.md`
- `notebooks/colab_qwen_server.ipynb`
- `pyproject.toml`
- `reports/experiment_results.md`
- `reports/project_report.md`
- `requirements-classifier.txt`
- `requirements-llm-local.txt`
- `requirements-ui.txt`
- `requirements.txt`
- `scripts/_bootstrap.py`
- `scripts/analyze_hateful_users.py`
- `scripts/ask.py`
- `scripts/check_env.py`
- `scripts/check_llm.py`
- `scripts/classify_posts.py`
- `scripts/create_gds_projection.py`
- `scripts/download_dataset.py`
- `scripts/evaluate_classifier.py`
- `scripts/evaluate_graph_rag.py`
- `scripts/evaluate_text_to_cypher.py`
- `scripts/extract_subgraph.py`
- `scripts/generate_synthetic_data.py`
- `scripts/ingest_graph.py`
- `scripts/initialize_neo4j.py`
- `scripts/inspect_dataset.py`
- `scripts/preprocess_dataset.py`
- `scripts/run_baselines.py`
- `scripts/run_community_analysis.py`
- `scripts/run_cypher.py`
- `scripts/run_demo.py`
- `scripts/run_graph_algorithms.py`
- `scripts/run_temporal_analysis.py`
- `scripts/train_classifier.py`
- `scripts/update_graph.py`
- `scripts/validate_dataset.py`
- `scripts/validate_graph.py`
- `src/graph_enhanced_lm/__init__.py`
- `src/graph_enhanced_lm/classification/__init__.py`
- `src/graph_enhanced_lm/classification/base.py`
- `src/graph_enhanced_lm/classification/evaluation.py`
- `src/graph_enhanced_lm/classification/factory.py`
- `src/graph_enhanced_lm/classification/label_mapping.py`
- `src/graph_enhanced_lm/classification/source_label_classifier.py`
- `src/graph_enhanced_lm/classification/transformer_classifier.py`
- `src/graph_enhanced_lm/config.py`
- `src/graph_enhanced_lm/data/__init__.py`
- `src/graph_enhanced_lm/data/dataset_adapters.py`
- `src/graph_enhanced_lm/data/loaders.py`
- `src/graph_enhanced_lm/data/preprocessing.py`
- `src/graph_enhanced_lm/data/schemas.py`
- `src/graph_enhanced_lm/data/splitting.py`
- `src/graph_enhanced_lm/data/synthetic.py`
- `src/graph_enhanced_lm/data/validators.py`
- `src/graph_enhanced_lm/evaluation/__init__.py`
- `src/graph_enhanced_lm/evaluation/benchmark.py`
- `src/graph_enhanced_lm/evaluation/graph_metrics.py`
- `src/graph_enhanced_lm/evaluation/llm_metrics.py`
- `src/graph_enhanced_lm/evaluation/metrics.py`
- `src/graph_enhanced_lm/evaluation/user_label_analysis.py`
- `src/graph_enhanced_lm/gds/__init__.py`
- `src/graph_enhanced_lm/gds/availability.py`
- `src/graph_enhanced_lm/gds/centrality.py`
- `src/graph_enhanced_lm/gds/community_detection.py`
- `src/graph_enhanced_lm/gds/memory_estimation.py`
- `src/graph_enhanced_lm/gds/projections.py`
- `src/graph_enhanced_lm/gds/statistics.py`
- `src/graph_enhanced_lm/graph/__init__.py`
- `src/graph_enhanced_lm/graph/constraints.py`
- `src/graph_enhanced_lm/graph/cypher_queries.py`
- `src/graph_enhanced_lm/graph/incremental_updates.py`
- `src/graph_enhanced_lm/graph/ingestion.py`
- `src/graph_enhanced_lm/graph/neo4j_client.py`
- `src/graph_enhanced_lm/graph/schema.py`
- `src/graph_enhanced_lm/graph/subgraph_extraction.py`
- `src/graph_enhanced_lm/graph/validation.py`
- `src/graph_enhanced_lm/llm/__init__.py`
- `src/graph_enhanced_lm/llm/answer_generation.py`
- `src/graph_enhanced_lm/llm/backends/__init__.py`
- `src/graph_enhanced_lm/llm/backends/hf_transformers.py`
- `src/graph_enhanced_lm/llm/backends/mock.py`
- `src/graph_enhanced_lm/llm/backends/openai_compatible.py`
- `src/graph_enhanced_lm/llm/base.py`
- `src/graph_enhanced_lm/llm/prompts.py`
- `src/graph_enhanced_lm/llm/query_validation.py`
- `src/graph_enhanced_lm/llm/qwen_client.py`
- `src/graph_enhanced_lm/llm/text_to_cypher.py`
- `src/graph_enhanced_lm/pipeline.py`
- `src/graph_enhanced_lm/retrieval/__init__.py`
- `src/graph_enhanced_lm/retrieval/evidence_formatter.py`
- `src/graph_enhanced_lm/retrieval/graph_rag.py`
- `src/graph_enhanced_lm/retrieval/graph_retriever.py`
- `src/graph_enhanced_lm/retrieval/intent_parser.py`
- `src/graph_enhanced_lm/temporal/__init__.py`
- `src/graph_enhanced_lm/temporal/date_filters.py`
- `src/graph_enhanced_lm/temporal/election_windows.py`
- `src/graph_enhanced_lm/temporal/trend_analysis.py`
- `src/graph_enhanced_lm/ui/__init__.py`
- `src/graph_enhanced_lm/ui/config_edit.py`
- `src/graph_enhanced_lm/ui/data_views.py`
- `src/graph_enhanced_lm/ui/graph_viz.py`
- `src/graph_enhanced_lm/ui/queries.py`
- `src/graph_enhanced_lm/ui/runner.py`
- `src/graph_enhanced_lm/utils/__init__.py`
- `src/graph_enhanced_lm/utils/errors.py`
- `src/graph_enhanced_lm/utils/logging_config.py`
- `src/graph_enhanced_lm/utils/reproducibility.py`
- `tasks.ps1`
- `tests/conftest.py`
- `tests/fixtures/hasoc/sample.tsv`
- `tests/fixtures/hateful_users/users.edges`
- `tests/fixtures/hateful_users/users_neighborhood_anon.csv`
- `tests/fixtures/hatexplain/dataset.json`
- `tests/fixtures/hatexplain/post_id_divisions.json`
- `tests/fixtures/russian_trolls/tweets.csv`
- `tests/fixtures/russian_trolls/users.csv`
- `tests/fixtures/twibot22/edge.csv`
- `tests/fixtures/twibot22/label.csv`
- `tests/fixtures/twibot22/tweet_0.json`
- `tests/fixtures/twibot22/user.json`
- `tests/test_adapters.py`
- `tests/test_alternative_datasets.py`
- `tests/test_cypher_validation.py`
- `tests/test_dataset_validation.py`
- `tests/test_end_to_end.py`
- `tests/test_env_file.py`
- `tests/test_graph_ingestion.py`
- `tests/test_graph_rag_mock.py`
- `tests/test_intent_and_evidence.py`
- `tests/test_label_mapping.py`
- `tests/test_metrics_and_scoring.py`
- `tests/test_preprocessing.py`
- `tests/test_synthetic_data.py`
- `tests/test_temporal_filters.py`
- `tests/test_ui_helpers.py`
