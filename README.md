<!-- # Graph-Enhanced Language Models for Scientific Literature

**Proposed subtitle (pending mentor approval):** *A Graph-RAG and Neo4j Framework for Identifying Candidate Hate/Spam Propagation Communities on X Around Election Periods*

> **Title note.** The official title mentions *scientific literature*, while the implemented
> application analyses X/Twitter-style social-network data (scope chosen from mentor
> discussion, as reported by the student). The title is intentionally unchanged. See
> [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) → *Clarification Required from Mentor*.

## What this system does

1. Loads social-media posts, users and interactions (retweets, replies, quotes, mentions) from a
   **synthetic** dataset (built in) or a real dataset (TwiBot-22).
2. Labels posts (NORMAL / HATE / OFFENSIVE / COMMUNITY_TARGETED / SPAM / UNKNOWN) with an
   existing-label pass-through or a Transformer classifier trained on HateXplain/HASOC.
3. Stores everything in a **Neo4j** knowledge graph and teaches you **Cypher** to query it.
4. Runs **Neo4j Graph Data Science**: Louvain communities, degree, PageRank, betweenness.
5. Compares activity in **election windows** (e.g. 30→11 days before, 10→1 days before, election day, after).
6. Lets you ask questions in English: **Qwen3-8B** writes safe read-only Cypher
   (Text-to-Cypher) or the system uses trusted templates, then Qwen answers **only from the
   retrieved graph evidence** (Graph RAG), citing `[E1]`, `[E2]`, …

It identifies *candidate communities exhibiting hate/spam propagation patterns* for human
review. A graph community is **not** evidence of coordination or intent.

```
 data (synthetic | TwiBot-22) ─► adapters ─► validation/cleaning ─► classifier ─► Neo4j graph
                                                                                    │
       answer with [E#] citations ◄─ Qwen3-8B ◄─ evidence ◄─ templates / Text-to-Cypher
                                                                                    │
                                              GDS (Louvain, PageRank) + temporal windows
```

---

## Requirements

| Component | Needed for | Notes |
|---|---|---|
| Windows 10/11 + **VS Code** | everything | Extensions: Python, Docker (see `.vscode/extensions.json`) |
| **Python 3.11** (3.10–3.12 OK) | everything | python.org installer → tick *Add python.exe to PATH* |
| **Docker Desktop** (WSL2 backend) | Neo4j + GDS | ≥ 8 GB RAM machine recommended; Neo4j uses ~3 GB |
| Internet (once) | pip install, Docker image, GDS plugin download | |
| Google account + **Colab** | running Qwen3-8B on a free GPU | or local Ollama with a smaller model |
| PyTorch + `requirements-classifier.txt` | *optional* Transformer classifier | CPU is fine for evaluation subsets |

Nothing requires a paid API or an X API key.

---

## Step-by-step: Windows PowerShell in VS Code

Open the folder in VS Code (**File → Open Folder**), then **Terminal → New Terminal** (PowerShell).

### Stage 0: Python environment (once)
```powershell
py -3.11 --version                      # must print Python 3.11.x
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                        # makes `graph_enhanced_lm` importable everywhere
Copy-Item .env.example .env             # then open .env and set NEO4J_PASSWORD (8+ chars)
```
In VS Code press `Ctrl+Shift+P → Python: Select Interpreter → .venv`. (Shortcut: `.\tasks.ps1 setup` does all of this.)

### Stage 1: offline baseline (no database, no LLM)
```powershell
python scripts/generate_synthetic_data.py        # data/synthetic/*.csv  (seed 42)
python scripts/preprocess_dataset.py --dataset synthetic   # clean + validate -> data/processed/synthetic
python scripts/classify_posts.py --dataset synthetic       # predictions.csv (ground-truth labels)
python scripts/run_temporal_analysis.py --source csv       # window table + outputs/figures/daily_label_counts.png
python scripts/run_demo.py --offline                       # all of the above + NetworkX Louvain
pytest -q                                                  # unit tests (integration tests auto-skip)
```
Or press **F5 → "1. Offline demo"**.

### Stage 2: Neo4j + Graph Data Science
```powershell
docker compose up -d                    # first start downloads Neo4j 5.26 + GDS plugin (minutes)
docker compose logs -f neo4j            # wait for "Started." then Ctrl+C
python scripts/initialize_neo4j.py      # [OK] connected, [OK] GDS x.y, constraints created
python scripts/ingest_graph.py --dataset synthetic
python scripts/validate_graph.py
```
Open **http://localhost:7474**, log in as `neo4j` / your password, and try queries from
`cypher/03_basic_queries.cypher`. Learn them with [docs/CYPHER_TUTORIAL.md](docs/CYPHER_TUTORIAL.md).

### Stage 3: communities, centrality and election windows
```powershell
python scripts/run_graph_algorithms.py --mode stream    # look first (nothing written)
python scripts/run_graph_algorithms.py                  # Louvain + degree + PageRank -> User nodes
python scripts/run_community_analysis.py                # outputs/metrics/community_ranking.md
python scripts/run_temporal_analysis.py --source neo4j  # per-community activity per window
python scripts/create_gds_projection.py --name pre10 --start 2025-03-05 --end 2025-03-14   # temporal subgraph
python scripts/extract_subgraph.py --start 2025-03-05 --end 2025-03-14 --labels HATE SPAM  # GraphML for Gephi
```

### Stage 4: Qwen3-8B (Colab GPU) + Graph RAG
1. Follow [docs/QWEN_SETUP.md](docs/QWEN_SETUP.md): open `notebooks/colab_qwen_server.ipynb` in Colab, run all cells, copy the printed URL.
2. In `.env` set `LLM_BASE_URL=https://<printed-url>/v1` and `LLM_MODEL=qwen3:8b`.
```powershell
python scripts/check_llm.py
python scripts/ask.py --no-llm "Which communities contain a high proportion of hate-labelled posts?"
python scripts/ask.py "Which users posted hate-labelled content during the ten days before the election?"
python scripts/ask.py --t2c-only "What are the 5 most used hashtags in spam posts?"
python scripts/run_demo.py                               # full end-to-end demo
```

### Stage 5: evaluation and report numbers
```powershell
python scripts/evaluate_text_to_cypher.py --safety-only  # offline safety suite
python scripts/evaluate_graph_rag.py --routing-only      # offline intent routing
python scripts/evaluate_text_to_cypher.py                # benchmark (Neo4j + LLM)
python scripts/evaluate_graph_rag.py                     # answers saved for manual scoring
python scripts/run_baselines.py                          # -> reports/experiment_results.md
```

### Stage 6: real data (see [docs/DATASET_CREATION_GUIDE.md](docs/DATASET_CREATION_GUIDE.md))
```powershell
python scripts/download_dataset.py --dataset hatexplain
python scripts/preprocess_dataset.py --dataset hatexplain
pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install -r requirements-classifier.txt
python scripts/evaluate_classifier.py --dataset hatexplain --split test --max-rows 500
# TwiBot-22, after access is granted:
python scripts/download_dataset.py --dataset twibot22 --from-path "D:\TwiBot-22"
python scripts/preprocess_dataset.py --dataset twibot22
python scripts/classify_posts.py --dataset twibot22 --backend transformer
# set active_graph_dataset: twibot22 in configs/dataset.yaml, then repeat Stages 2-5
```

No TwiBot-22 access? Use the free alternatives in [docs/ALTERNATIVE_DATASETS.md](docs/ALTERNATIVE_DATASETS.md) (Hateful Users on Twitter, Russian troll tweets).

Linux/macOS: use `source .venv/bin/activate`, `python3`, and `make <target>` instead of `tasks.ps1`.

---

## What needs external setup

| Item | External requirement | Without it |
|---|---|---|
| Stages 2–3 | Docker Desktop running Neo4j | Offline demo and CSV temporal analysis still work |
| GDS algorithms | GDS plugin (downloaded automatically by Docker) | Ingestion + Cypher still work |
| Qwen answers / Text-to-Cypher | Colab notebook or local Ollama | `--no-llm` gives evidence without prose |
| Transformer classifier | PyTorch + model download | `source_labels` backend (synthetic only) |
| HASOC, TwiBot-22 | Manual download / access application | Synthetic + HateXplain |

## Repository map
```
configs/      all settings (YAML)          src/graph_enhanced_lm/  library code
scripts/      command-line entry points    cypher/   learning + schema Cypher files
tests/        pytest (unit + integration)  benchmarks/  T2C, safety, RAG benchmarks
docs/         tutorials and guides         reports/  report draft + generated results
notebooks/    Colab Qwen server            data/, outputs/   generated (git-ignored)
```
Full file list: [MANIFEST.md](MANIFEST.md). Problems: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Honesty rules built into the project
* Synthetic data is flagged `synthetic=true` everywhere and printed as such.
* Model-generated labels are stored with `label_source=model` and never overwrite human labels.
* TwiBot-22 bot labels are never used as hate labels; datasets are never joined.
* Generated Cypher is validated, EXPLAIN-checked and executed read-only; unsafe queries return a controlled error.
* Scripts report only metrics computed from real outputs; missing experiments show "NOT RUN". -->



# Graph-Enhanced Language Models for Scientific Literature

**Proposed subtitle (pending mentor approval):** *A Graph-RAG and Neo4j Framework for Identifying Candidate Hate/Spam Propagation Communities on X Around Election Periods*

> **Title note.** The official title mentions *scientific literature*, while the implemented
> application analyses X/Twitter-style social-network data (scope chosen from mentor
> discussion, as reported by the student). The title is intentionally unchanged. See
> [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) → *Clarification Required from Mentor*.

## What this system does

1. Loads social-media posts, users and interactions (retweets, replies, quotes, mentions) from a
   **synthetic** dataset (built in) or a real dataset (TwiBot-22).
2. Labels posts (NORMAL / HATE / OFFENSIVE / COMMUNITY_TARGETED / SPAM / UNKNOWN) with an
   existing-label pass-through or a Transformer classifier trained on HateXplain/HASOC.
3. Stores everything in a **Neo4j** knowledge graph and teaches you **Cypher** to query it.
4. Runs **Neo4j Graph Data Science**: Louvain communities, degree, PageRank, betweenness.
5. Compares activity in **election windows** (e.g. 30→11 days before, 10→1 days before, election day, after).
6. Lets you ask questions in English: **Qwen3-8B** writes safe read-only Cypher
   (Text-to-Cypher) or the system uses trusted templates, then Qwen answers **only from the
   retrieved graph evidence** (Graph RAG), citing `[E1]`, `[E2]`, …

It identifies *candidate communities exhibiting hate/spam propagation patterns* for human
review. A graph community is **not** evidence of coordination or intent.

```
 data (synthetic | TwiBot-22) ─► adapters ─► validation/cleaning ─► classifier ─► Neo4j graph
                                                                                    │
       answer with [E#] citations ◄─ Qwen3-8B ◄─ evidence ◄─ templates / Text-to-Cypher
                                                                                    │
                                              GDS (Louvain, PageRank) + temporal windows
```

---

## Requirements

| Component | Needed for | Notes |
|---|---|---|
| Windows 10/11 + **VS Code** | everything | Extensions: Python, Docker (see `.vscode/extensions.json`) |
| **Python 3.11** (3.10–3.12 OK) | everything | python.org installer → tick *Add python.exe to PATH* |
| **Docker Desktop** (WSL2 backend) | Neo4j + GDS | ≥ 8 GB RAM machine recommended; Neo4j uses ~3 GB |
| Internet (once) | pip install, Docker image, GDS plugin download | |
| Google account + **Colab** | running Qwen3-8B on a free GPU | or local Ollama with a smaller model |
| PyTorch + `requirements-classifier.txt` | *optional* Transformer classifier | CPU is fine for evaluation subsets |

Nothing requires a paid API or an X API key.

---

## Step-by-step: Windows PowerShell in VS Code

Open the folder in VS Code (**File → Open Folder**), then **Terminal → New Terminal** (PowerShell).

### Stage 0: Python environment (once)
```powershell
py -3.11 --version                      # must print Python 3.11.x
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                        # makes `graph_enhanced_lm` importable everywhere
Copy-Item .env.example .env             # then open .env and set NEO4J_PASSWORD (8+ chars)
```
In VS Code press `Ctrl+Shift+P → Python: Select Interpreter → .venv`. (Shortcut: `.\tasks.ps1 setup` does all of this.)

### Stage 1: offline baseline (no database, no LLM)
```powershell
python scripts/generate_synthetic_data.py        # data/synthetic/*.csv  (seed 42)
python scripts/preprocess_dataset.py --dataset synthetic   # clean + validate -> data/processed/synthetic
python scripts/classify_posts.py --dataset synthetic       # predictions.csv (ground-truth labels)
python scripts/run_temporal_analysis.py --source csv       # window table + outputs/figures/daily_label_counts.png
python scripts/run_demo.py --offline                       # all of the above + NetworkX Louvain
pytest -q                                                  # unit tests (integration tests auto-skip)
```
Or press **F5 → "1. Offline demo"**.

### Stage 2: Neo4j + Graph Data Science
```powershell
docker compose up -d                    # first start downloads Neo4j 5.26 + GDS plugin (minutes)
docker compose logs -f neo4j            # wait for "Started." then Ctrl+C
python scripts/initialize_neo4j.py      # [OK] connected, [OK] GDS x.y, constraints created
python scripts/ingest_graph.py --dataset synthetic
python scripts/validate_graph.py
```
Open **http://localhost:7474**, log in as `neo4j` / your password, and try queries from
`cypher/03_basic_queries.cypher`. Learn them with [docs/CYPHER_TUTORIAL.md](docs/CYPHER_TUTORIAL.md).

### Stage 3: communities, centrality and election windows
```powershell
python scripts/run_graph_algorithms.py --mode stream    # look first (nothing written)
python scripts/run_graph_algorithms.py                  # Louvain + degree + PageRank -> User nodes
python scripts/run_community_analysis.py                # outputs/metrics/community_ranking.md
python scripts/run_temporal_analysis.py --source neo4j  # per-community activity per window
python scripts/create_gds_projection.py --name pre10 --start 2025-03-05 --end 2025-03-14   # temporal subgraph
python scripts/extract_subgraph.py --start 2025-03-05 --end 2025-03-14 --labels HATE SPAM  # GraphML for Gephi
```

### Stage 4: Qwen3-8B (Colab GPU) + Graph RAG
1. Follow [docs/QWEN_SETUP.md](docs/QWEN_SETUP.md): open `notebooks/colab_qwen_server.ipynb` in Colab, run all cells, copy the printed URL.
2. In `.env` set `LLM_BASE_URL=https://<printed-url>/v1` and `LLM_MODEL=qwen3:8b`.
```powershell
python scripts/check_llm.py
python scripts/ask.py --no-llm "Which communities contain a high proportion of hate-labelled posts?"
python scripts/ask.py "Which users posted hate-labelled content during the ten days before the election?"
python scripts/ask.py --t2c-only "What are the 5 most used hashtags in spam posts?"
python scripts/run_demo.py                               # full end-to-end demo
```

### Stage 5: evaluation and report numbers
```powershell
python scripts/evaluate_text_to_cypher.py --safety-only  # offline safety suite
python scripts/evaluate_graph_rag.py --routing-only      # offline intent routing
python scripts/evaluate_text_to_cypher.py                # benchmark (Neo4j + LLM)
python scripts/evaluate_graph_rag.py                     # answers saved for manual scoring
python scripts/run_baselines.py                          # -> reports/experiment_results.md
```

### Stage 6: real data (see [docs/DATASET_CREATION_GUIDE.md](docs/DATASET_CREATION_GUIDE.md))
```powershell
python scripts/download_dataset.py --dataset hatexplain
python scripts/preprocess_dataset.py --dataset hatexplain
pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install -r requirements-classifier.txt
python scripts/evaluate_classifier.py --dataset hatexplain --split test --max-rows 500
# TwiBot-22, after access is granted:
python scripts/download_dataset.py --dataset twibot22 --from-path "D:\TwiBot-22"
python scripts/preprocess_dataset.py --dataset twibot22
python scripts/classify_posts.py --dataset twibot22 --backend transformer
# set active_graph_dataset: twibot22 in configs/dataset.yaml, then repeat Stages 2-5
```

No TwiBot-22 access? Use the free alternatives in [docs/ALTERNATIVE_DATASETS.md](docs/ALTERNATIVE_DATASETS.md) (Hateful Users on Twitter, Russian troll tweets).

Linux/macOS: use `source .venv/bin/activate`, `python3`, and `make <target>` instead of `tasks.ps1`.

---

## Dashboard (demo for presentations)
```powershell
pip install -r requirements-ui.txt
streamlit run app/streamlit_app.py
```
Everything above (datasets, graph, Cypher, communities, temporal, Qwen Q&A, evaluation) is available as buttons
and forms. See [docs/DASHBOARD.md](docs/DASHBOARD.md).

## What needs external setup

| Item | External requirement | Without it |
|---|---|---|
| Stages 2–3 | Docker Desktop running Neo4j | Offline demo and CSV temporal analysis still work |
| GDS algorithms | GDS plugin (downloaded automatically by Docker) | Ingestion + Cypher still work |
| Qwen answers / Text-to-Cypher | Colab notebook or local Ollama | `--no-llm` gives evidence without prose |
| Transformer classifier | PyTorch + model download | `source_labels` backend (synthetic only) |
| HASOC, TwiBot-22 | Manual download / access application | Synthetic + HateXplain |

## Repository map
```
configs/      all settings (YAML)          src/graph_enhanced_lm/  library code
scripts/      command-line entry points    cypher/   learning + schema Cypher files
tests/        pytest (unit + integration)  benchmarks/  T2C, safety, RAG benchmarks
docs/         tutorials and guides         reports/  report draft + generated results
notebooks/    Colab Qwen server            data/, outputs/   generated (git-ignored)
```
Full file list: [MANIFEST.md](MANIFEST.md). Problems: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Honesty rules built into the project
* Synthetic data is flagged `synthetic=true` everywhere and printed as such.
* Model-generated labels are stored with `label_source=model` and never overwrite human labels.
* TwiBot-22 bot labels are never used as hate labels; datasets are never joined.
* Generated Cypher is validated, EXPLAIN-checked and executed read-only; unsafe queries return a controlled error.
* Scripts report only metrics computed from real outputs; missing experiments show "NOT RUN".
