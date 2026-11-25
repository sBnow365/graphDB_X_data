# Dashboard (Streamlit)

A browser front-end for demonstrating the whole project without typing commands.
Every button runs the same code as the scripts (the exact command is shown under it).

## Install (once)
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ui.txt        # streamlit, plotly, pyvis
```

## Run
```powershell
streamlit run app/streamlit_app.py
```
Or `.\tasks.ps1 dashboard`, or VS Code **F5 → "0. Dashboard (Streamlit)"**. It opens http://localhost:8501.
Run it from the project root (that is where `.streamlit/config.toml` with the theme lives).
Stop it with `Ctrl+C` in the terminal.

Before a demo: start Docker Desktop → `docker compose up -d` → (optional) start the Colab Qwen notebook.

## Pages
| Page | What you can do |
|---|---|
| Overview | live status of Neo4j, GDS, Qwen and the active dataset; pipeline checklist; architecture; graph composition |
| Datasets | pick any dataset; generate synthetic data (seed, size, difficulty); download / copy files; preprocess; classify; evaluate classifiers; preview tables; validation report; make a dataset the active graph dataset |
| Knowledge Graph | initialise schema, ingest, validate, clear the graph (type DELETE), draw a user's neighbourhood or a community |
| Cypher Lab | run any trusted template with form inputs; write your own query (EXPLAIN must say read-only); run the tutorial `.cypher` files |
| Communities | run Louvain + centrality, rank communities, explain a community (graph + central users + supporting posts); hateful-user homophily for the Hateful Users dataset |
| Temporal | set election date/timezone (presets for synthetic and US 2016), window table, daily chart with election line, label shares, per-community heatmap |
| Ask the Graph | chat with Qwen: answer with [E#] citations, intent/method, evidence table, generated Cypher, evidence subgraph; toggle LLM / force Text-to-Cypher; update the Colab URL in .env from the page |
| Evaluation | classifier metrics + confusion matrix, Text-to-Cypher safety/benchmark, Graph-RAG routing/full run, collected results and downloads |
| Settings & Logs | masked .env, config files, run logs, cache reset |

## Switching dataset in the demo
Datasets → choose → **Make active graph dataset** → Knowledge Graph → **Clear graph** (type DELETE) →
**Ingest** → Communities → **Run Louvain + centrality** → **Rank communities** → Temporal → set the election preset.

## Safety
The dashboard only runs read-only queries you type (Neo4j EXPLAIN must report `r`). Writes happen only through
the pipeline buttons (ingest, GDS write, clear graph with confirmation). Passwords are never displayed.

## Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` not recognised | activate `.venv`, `pip install -r requirements-ui.txt` |
| Graphs show as a table | install `pyvis` (interactive) or `plotly` (static fallback) |
| Graph area blank | pyvis loads its JS from a CDN: the browser needs internet |
| "Neo4j is not reachable" | Docker Desktop running? `docker compose up -d`, then **Retry connection** |
| Qwen 🔴 | Colab stopped or URL changed: paste the new URL on Ask the Graph → Save |
| Long buttons (ingest, classify) | the page waits until the script finishes; output streams live |
