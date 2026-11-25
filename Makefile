# Linux/macOS/Colab equivalent of tasks.ps1
PY ?= .venv/bin/python
setup: ; python3 -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . && cp -n .env.example .env || true
synthetic: ; $(PY) scripts/generate_synthetic_data.py
offline-demo: ; $(PY) scripts/run_demo.py --offline
neo4j-up: ; docker compose --env-file .env up -d
init: ; $(PY) scripts/initialize_neo4j.py
ingest: ; $(PY) scripts/preprocess_dataset.py --dataset synthetic && $(PY) scripts/classify_posts.py --dataset synthetic && $(PY) scripts/ingest_graph.py --dataset synthetic
gds: ; $(PY) scripts/run_graph_algorithms.py && $(PY) scripts/run_community_analysis.py
demo: ; $(PY) scripts/run_demo.py
test: ; $(PY) -m pytest -q
.PHONY: setup synthetic offline-demo neo4j-up init ingest gds demo test
