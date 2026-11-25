# Troubleshooting (Windows-first)

| Problem | Cause / fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen terminal |
| `py` not found | install Python 3.11 from python.org with "Add to PATH"; or use full path |
| `ModuleNotFoundError: graph_enhanced_lm` | activate `.venv`, run `pip install -e .`; in VS Code select the `.venv` interpreter |
| pip dependency conflicts | fresh venv; `pip install -r requirements.txt` before optional extras; install torch from pytorch.org selector first |
| `pip install torch` very large/slow | use CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Docker: `WSL 2 installation is incomplete` | `wsl --install` in admin PowerShell, reboot, start Docker Desktop |
| `NEO4J_PASSWORD` required by compose | create `.env` from `.env.example`; run compose from the project folder |
| Neo4j won't start, password error | password must be ≥8 chars; if you changed it after first start: `docker compose down -v` (**deletes graph data**) then `up -d` |
| `Cannot connect to Neo4j` | `docker compose ps` (healthy?), wait 30–60 s, check port 7687 not used by Neo4j Desktop |
| `Unauthorized` | `.env` password differs from the one used at first container start (see above) |
| `There is no procedure with the name gds...` | GDS plugin not loaded: `docker compose logs neo4j`; first start needs internet to fetch the plugin; check `NEO4J_PLUGINS` line |
| Projection fails on `undirectedRelationshipTypes` | code retries as directed automatically; upgrade GDS for undirected |
| Out of memory in GDS | drop old projections (`create_gds_projection.py --list/--drop`), narrow date range, raise heap in compose |
| Cypher `SyntaxError` in Browser | parameters missing: set `:param` first; `$` params cannot be used for labels/relationship types |
| `No processed data in data/processed/...` | run `preprocess_dataset.py` (and `classify_posts.py`) first |
| `File not found data/raw/...` | wrong path; check `configs/dataset_mappings/<name>.yaml` |
| HASOC `lacks columns` | edit `columns:` in `hasoc.yaml` to match your file header (`inspect_dataset.py` shows it) |
| Many posts without timestamps | check `tweet_fields.created_at` mapping; see warnings in logs |
| `Model labels [...] have no mapping` | read the checkpoint's model card; add explicit mapping in `label_taxonomy.yaml` |
| Model download fails | internet/proxy; set `HF_TOKEN` for rate limits; download once and set `HF_HOME` |
| CUDA out of memory | lower `batch_size`/`max_length`; use CPU; for Qwen use 4-bit or a smaller model |
| `Cannot reach LLM endpoint` | Colab session ended or URL changed; see QWEN_SETUP.md |
| LLM returns invalid Cypher | inspect with `ask.py --t2c-only --json`; add few-shot examples; larger model |
| Generated query returns no rows | check parameters (date window covers data? synthetic data spans 2025-01-29…2025-03-30), label names are upper-case, Louvain run before community questions |
| Answer says INSUFFICIENT_EVIDENCE | evidence empty or irrelevant: correct behaviour; inspect evidence with `--no-llm` |
| Tests: integration tests skipped | expected; set `$env:RUN_INTEGRATION="1"` with Neo4j running |
