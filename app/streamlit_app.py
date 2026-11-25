"""Graph-Enhanced Language Models: project dashboard.

Run from the project root:   streamlit run app/streamlit_app.py
Every button runs the same code as the scripts in scripts/ (the command is shown under it).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

st.set_page_config(page_title="Graph-Enhanced LM Dashboard", page_icon="🕸️", layout="wide",
                   initial_sidebar_state="expanded")

try:
    import plotly.express as px  # noqa: E402
    import plotly.graph_objects as go  # noqa: E402
    HAVE_PLOTLY = True
except ImportError:  # pragma: no cover
    HAVE_PLOTLY = False

from graph_enhanced_lm.config import ENV_FILE, PROJECT_ROOT, get_config  # noqa: E402
from graph_enhanced_lm.graph.cypher_queries import TEMPLATES  # noqa: E402
from graph_enhanced_lm.graph.schema import LABEL_VALUES  # noqa: E402
from graph_enhanced_lm.ui import data_views as dv  # noqa: E402
from graph_enhanced_lm.ui import queries as uq  # noqa: E402
from graph_enhanced_lm.ui.config_edit import env_summary, set_yaml_scalar, update_env_value  # noqa: E402
from graph_enhanced_lm.ui.graph_viz import build_graph, to_plotly, to_pyvis_html  # noqa: E402
from graph_enhanced_lm.ui.runner import display_command, stream_script  # noqa: E402

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #0f766e 0%, #155e75 55%, #1e3a8a 100%); color: #fff;
       padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color:#fff; font-size: 1.75rem; margin: 0 0 .3rem 0;}
.hero p {color:#e0f2f1; margin: .15rem 0; font-size: .95rem;}
.card {background:#fff; border:1px solid #e2e8f0; border-radius: 14px; padding: 1rem 1.1rem; height: 100%;}
.card h4 {margin: 0 0 .35rem 0; font-size: .95rem; color:#334155;}
.big {font-size: 1.35rem; font-weight: 700; color:#0f172a;}
.ok {color:#15803d; font-weight:600;} .bad {color:#b91c1c; font-weight:600;} .warn {color:#b45309; font-weight:600;}
.pill {display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem; margin-right:.3rem;
       background:#e0f2f1; color:#0f766e; border:1px solid #99f6e4;}
.pill.red {background:#fee2e2; color:#991b1b; border-color:#fecaca;}
.pill.gray {background:#f1f5f9; color:#334155; border-color:#e2e8f0;}
.note {background:#fff7ed; border-left:4px solid #f59e0b; padding:.6rem .9rem; border-radius:8px; font-size:.9rem;}
section[data-testid="stSidebar"] {background: #0f172a;}
section[data-testid="stSidebar"] * {color: #e2e8f0;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- shared helpers
def cfg():
    return get_config()


def active_dataset() -> str:
    return cfg().dataset["active_graph_dataset"]


def community_prop() -> str:
    return cfg().gds["louvain"]["write_property"]


@st.cache_resource(show_spinner=False)
def _client(uri: str, user: str, password: str, database: str):
    from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

    c = Neo4jClient(uri, user, password, database)
    c.verify()
    return c


def get_client():
    """Return (client, error)."""
    try:
        s = cfg().neo4j_settings()
        return _client(s.uri, s.user, s.password, s.database), None
    except Exception as exc:  # shown to the user
        return None, str(exc)


def need_client():
    client, err = get_client()
    if client is None:
        st.error("Neo4j is not reachable, so this page cannot run.")
        with st.expander("Details and fix"):
            st.code(err or "unknown error")
            st.markdown("1. Start **Docker Desktop**  \n2. `docker compose up -d`  \n"
                        "3. Check `.env` with `python scripts/check_env.py`")
        if st.button("🔄 Retry connection"):
            _client.clear()
            st.rerun()
        st.stop()
    return client


@st.cache_data(ttl=60, show_spinner=False)
def gds_status(_key: str):
    client, _ = get_client()
    if client is None:
        return None
    try:
        from graph_enhanced_lm.gds.availability import gds_info

        return gds_info(client)
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=60, show_spinner=False)
def llm_status(backend: str, base_url: str, model: str):
    if backend == "mock":
        return {"ok": True, "detail": "mock backend"}
    if backend != "openai_compatible":
        return {"ok": None, "detail": f"{backend}: checked when used"}
    try:
        from graph_enhanced_lm.llm.backends.openai_compatible import OpenAICompatibleClient

        models = OpenAICompatibleClient(base_url, model, os.getenv("LLM_API_KEY") or None, timeout=10).health()
        return {"ok": True, "detail": f"serving {models}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:200]}


@st.cache_data(ttl=30, show_spinner=False)
def graph_counts(_key: str):
    client, _ = get_client()
    if client is None:
        return None, None
    try:
        return (pd.DataFrame(uq.run(client, uq.Q_COUNTS)), pd.DataFrame(uq.run(client, uq.Q_REL_COUNTS)))
    except Exception:
        return None, None


def refresh_caches():
    gds_status.clear()
    llm_status.clear()
    graph_counts.clear()
    get_config.cache_clear()


def run_script(script: str, *args: str, stdin_text: str | None = None, key: str = "") -> int:
    """Run scripts/<script> with live output; returns the exit code."""
    st.caption(f"Command: `{display_command(script, *args)}`")
    box = st.empty()
    lines: list[str] = []
    rc = 1
    with st.spinner(f"Running {script} …"):
        for item in stream_script(script, *args, stdin_text=stdin_text):
            if isinstance(item, int):
                rc = item
                break
            lines.append(item)
            box.code("\n".join(lines[-250:]) or "…", language="text")
    (st.success if rc == 0 else st.error)(f"{script}: {'finished' if rc == 0 else f'failed (exit code {rc})'}")
    refresh_caches()
    return rc


def show_graph(nodes, edges, color_by: str = "community", height: int = 560, key: str = "g"):
    if not nodes and not edges:
        st.info("Nothing to draw for this selection.")
        return
    g = build_graph(nodes, edges, color_by)
    st.caption(f"{g.number_of_nodes()} nodes · {g.number_of_edges()} edges · drag to move, scroll to zoom, hover for details")
    try:
        components.html(to_pyvis_html(g, height), height=height + 20, scrolling=False)
    except ImportError:
        if HAVE_PLOTLY:
            st.plotly_chart(to_plotly(g), key=f"plotly_{key}")
        else:
            st.dataframe(pd.DataFrame(edges))
    legend = {"community": "Colour = Louvain community; size = PageRank.",
              "hate_label": "Red = HATEFUL user (human label), blue = NORMAL, grey = unannotated.",
              "content": "Red = user posted hate-labelled content, blue = other."}
    st.caption(legend.get(color_by, "") + " Squares are posts (red HATE, orange OFFENSIVE, grey NORMAL).")


def synthetic_banner():
    if active_dataset() == "synthetic":
        st.markdown('<div class="note">⚠️ Active graph dataset is <b>SYNTHETIC</b>: results demonstrate the '
                    'method and are not real-world evidence.</div>', unsafe_allow_html=True)


def card(title: str, value: str, sub: str = "", cls: str = ""):
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="big {cls}">{value}</div>'
                f'<div style="color:#64748b;font-size:.85rem">{sub}</div></div>', unsafe_allow_html=True)


def bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "", key: str = ""):
    if df is None or df.empty:
        return
    if HAVE_PLOTLY:
        fig = px.bar(df, x=x, y=y, color=color, title=title, template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, key=key or f"bar_{title}_{x}_{y}")
    else:
        st.bar_chart(df.set_index(x)[y])


# ----------------------------------------------------------------------------- pages
def page_overview():
    p = cfg().project
    st.markdown(f"""<div class="hero"><h1>{p['official_title']}</h1>
    <p><b>Subtitle (proposed):</b> {p['proposed_subtitle']}</p>
    <p>Knowledge graph (Neo4j) · Graph Data Science · Election-window analysis · Qwen3-8B Graph RAG</p></div>""",
                unsafe_allow_html=True)
    client, err = get_client()
    gds = gds_status(str(client is not None)) if client else None
    llm = cfg().llm_settings()
    ls = llm_status(llm["backend"], llm.get("base_url", ""), llm.get("model", ""))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Neo4j", "Connected" if client else "Offline", "bolt://localhost:7687" if client else (err or "")[:80],
             "ok" if client else "bad")
    with c2:
        v = gds.get("version") if isinstance(gds, dict) else None
        card("Graph Data Science", v or "Unavailable", "plugin inside Neo4j", "ok" if v else "warn")
    with c3:
        card("Qwen LLM", {True: "Reachable", False: "Unreachable", None: "Not checked"}[ls["ok"]],
             f"{llm.get('model')} · {ls['detail'][:60]}", {True: "ok", False: "bad", None: "warn"}[ls["ok"]])
    with c4:
        card("Active graph dataset", dv.DATASETS[active_dataset()]["title"], dv.DATASETS[active_dataset()]["role"])
    st.write("")
    if st.button("🔄 Refresh status"):
        refresh_caches()
        _client.clear()
        st.rerun()
    synthetic_banner()

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Pipeline progress")
        name = active_dataset()
        status = dv.dataset_status(name)
        nodes, rels = graph_counts(str(client is not None))
        has_posts = nodes is not None and not nodes.empty and (nodes["label"].isin(["Post", "User"])).any()
        has_comm = False
        if client:
            try:
                has_comm = bool(uq.run(client, uq.Q_COMMUNITIES, prop=community_prop(), limit=1))
            except Exception:
                pass
        steps = [("Data preprocessed", status["processed"]),
                 ("Posts classified", status["predictions"] or name == "hateful_users"),
                 ("Neo4j connected", client is not None),
                 ("Knowledge graph loaded", bool(has_posts)),
                 ("Louvain communities computed", has_comm),
                 ("Qwen reachable", bool(ls["ok"]))]
        for label, done in steps:
            st.markdown(f"{'✅' if done else '⬜'} {label}")
    with right:
        st.subheader("Graph composition")
        if nodes is not None and not nodes.empty:
            bar(nodes, "label", "count", title="Nodes", key="ov_nodes")
            bar(rels, "type", "count", title="Relationships", key="ov_rels")
        else:
            st.info("Graph empty or Neo4j offline. Use **Knowledge Graph → Ingest**.")
    st.subheader("Architecture")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#ecfeff", color="#0f766e", fontname="Helvetica"];
      data [label="Datasets\\nsynthetic · Hateful Users · Russian trolls\\nHateXplain · HASOC"];
      prep [label="Adapters +\\nvalidation"]; clf [label="BERT classifier\\n(content labels)"];
      kg [label="Neo4j\\nknowledge graph", fillcolor="#ccfbf1"]; gds [label="GDS\\nLouvain · PageRank"];
      tmp [label="Election\\nwindows"]; ret [label="Templates /\\nText-to-Cypher\\n(read-only)"];
      llm [label="Qwen3-8B\\n(grounded answer)", fillcolor="#e0e7ff"]; ans [label="Answer +\\n[E#] evidence", shape=note];
      data->prep->clf->kg; kg->gds; kg->tmp; gds->ret; tmp->ret; kg->ret; ret->llm->ans; }""")


def page_datasets():
    st.header("📦 Datasets")
    names = list(dv.DATASETS)
    name = st.selectbox("Dataset", names, index=names.index(active_dataset()),
                        format_func=lambda n: f"{dv.DATASETS[n]['title']}  ·  {n}")
    info = dv.DATASETS[name]
    status = dv.dataset_status(name)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**{info['title']}**: {info['about']}")
        st.markdown(f"<span class='pill'>{info['role']}</span><span class='pill gray'>{info['access']}</span>"
                    + (f" [source]({info['link']})" if info["link"] else ""), unsafe_allow_html=True)
    with c2:
        st.markdown(f"Raw files: **{len(status['raw_files'])}**  \nProcessed: **{'yes' if status['processed'] else 'no'}**"
                    f"  \nPredictions: **{'yes' if status['predictions'] else 'no'}**")
        if name in dv.GRAPH_DATASETS:
            if name == active_dataset():
                st.success("This is the active graph dataset")
            elif st.button("⭐ Make active graph dataset", key="mk_active"):
                set_yaml_scalar("dataset", "active_graph_dataset", name)
                refresh_caches()
                st.toast(f"Active graph dataset → {name}. Clear the graph and ingest it on the Knowledge Graph page.")
                st.rerun()

    t1, t2, t3, t4 = st.tabs(["1 · Get data", "2 · Prepare & classify", "3 · Preview", "4 · Validation"])
    with t1:
        if name == "synthetic":
            s = cfg().synthetic
            a, b, c, d = st.columns(4)
            seed = a.number_input("Seed", value=int(s["seed"]), step=1)
            n_users = b.number_input("Users", value=int(s["n_users"]), min_value=50, max_value=5000, step=50)
            intra = c.slider("Intra-community interaction prob.", 0.3, 0.95, float(s["intra_community_interaction_prob"]),
                             0.05, help="Lower = harder communities to detect")
            e_date = d.date_input("Fictional election date", date.fromisoformat(str(s["election_date"])))
            if st.button("⚙️ Generate synthetic data", type="primary"):
                set_yaml_scalar("synthetic", "intra_community_interaction_prob", float(intra))
                set_yaml_scalar("synthetic", "election_date", e_date.isoformat())
                run_script("generate_synthetic_data.py", "--seed", str(seed), "--n-users", str(n_users))
        elif name in ("hatexplain", "russian_trolls"):
            st.write("This dataset can be downloaded automatically from its public source.")
            if st.button("⬇️ Download", type="primary"):
                run_script("download_dataset.py", "--dataset", name)
        else:
            st.markdown(f"Download manually from **{info['link']}** ({info['access']}), then copy the file(s) in:")
            path = st.text_input("Local file or folder", placeholder=r"C:\Users\you\Downloads\...")
            if st.button("📥 Copy into project", disabled=not path):
                run_script("download_dataset.py", "--dataset", name, "--from-path", path)
        if status["raw_files"]:
            st.caption("In data/raw/" + name + ": " + ", ".join(status["raw_files"]))
    with t2:
        if st.button("🧹 Preprocess + validate", type="primary", key="prep"):
            run_script("preprocess_dataset.py", "--dataset", name)
        if name in dv.GRAPH_DATASETS and info["classify"]:
            st.divider()
            backend = st.radio("Classifier backend", ["source_labels", "transformer"],
                               index=0 if info["classify"] == "source_labels" else 1, horizontal=True,
                               help="source_labels = labels already in the data; transformer = BERT model")
            if st.button("🏷️ Classify posts", key="classify"):
                run_script("classify_posts.py", "--dataset", name, "--backend", backend)
        elif name == "hateful_users":
            st.info("No posts to classify: this dataset carries human user-level labels.")
        if name in dv.CLASSIFIER_DATASETS:
            st.divider()
            a, b = st.columns(2)
            split = a.selectbox("Split", ["test", "val", "train"])
            max_rows = b.number_input("Max rows (0 = all)", value=300, min_value=0, step=100)
            if st.button("📊 Evaluate classifier", key="evalclf"):
                args = ["--dataset", name, "--split", split] + (["--max-rows", str(max_rows)] if max_rows else [])
                run_script("evaluate_classifier.py", *args)
    with t3:
        d = dv.processed_path(name)
        tables = [t for t in ["users", "posts", "interactions", "user_edges", "predictions", "classification"]
                  if (d / f"{t}.csv").exists()]
        if not tables:
            st.info("No processed data yet: run step 2.")
        else:
            tab = st.radio("Table", tables, horizontal=True)
            df = dv.read_csv_safe(d / f"{tab}.csv", nrows=300)
            st.dataframe(df, hide_index=True)
        dist = dv.label_distribution(name)
        if not dist.empty:
            ycol = "users" if "users" in dist.columns else "posts"
            xcol = dist.columns[0]
            color = dist.columns[1] if len(dist.columns) > 2 else None
            bar(dist, xcol, ycol, color, "Label distribution", key=f"dist_{name}")
    with t4:
        if status["validation"]:
            v = status["validation"]
            (st.success if v.get("ok") else st.error)(f"Validation {'PASS' if v.get('ok') else 'FAIL'}")
            for w in v.get("warnings", []):
                st.warning(w)
            for e in v.get("errors", []):
                st.error(e)
            st.json(v.get("stats", {}), expanded=False)
        else:
            st.info("No validation report yet.")


def page_graph():
    st.header("🧠 Knowledge Graph")
    client = need_client()
    synthetic_banner()
    name = active_dataset()
    st.markdown(f"Active graph dataset: <span class='pill'>{name}</span>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        if st.button("🧱 Initialize schema", help="constraints + indexes"):
            run_script("initialize_neo4j.py")
    with b:
        if st.button(f"📥 Ingest '{name}'", type="primary"):
            run_script("ingest_graph.py", "--dataset", name)
    with c:
        if st.button("✅ Validate graph"):
            from graph_enhanced_lm.graph.validation import validate_graph

            res = validate_graph(client)
            (st.success if res["ok"] else st.error)("Graph OK" if res["ok"] else f"Problems: {res['problems']}")
            st.json({k: v for k, v in res.items() if k not in ("node_counts", "relationship_counts")}, expanded=False)
    nodes, rels = graph_counts("graph")
    if nodes is not None and not nodes.empty:
        x, y = st.columns(2)
        with x:
            bar(nodes, "label", "count", title="Nodes", key="kg_nodes")
        with y:
            bar(rels, "type", "count", title="Relationships", key="kg_rels")
    with st.expander("⚠️ Danger zone: clear the whole graph (needed before switching datasets)"):
        st.write("Deletes every node and relationship and drops GDS projections. Constraints stay. "
                 "Your CSV files are not touched; you can re-ingest any time.")
        confirm = st.text_input("Type DELETE to confirm")
        if st.button("🗑️ Clear graph", disabled=confirm != "DELETE"):
            from graph_enhanced_lm.gds.projections import drop_graph, list_graphs

            with st.spinner("Deleting in batches…"):
                try:
                    for g in list_graphs(client):
                        drop_graph(client, g["graphName"])
                except Exception:
                    pass
                client.run_schema("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")
                for m in cfg().path("processed").glob("*/_manifest"):
                    for f in m.glob("*.csv"):
                        f.unlink()
            refresh_caches()
            st.success("Graph cleared.")
    st.subheader("Graph explorer")
    color_by = st.radio("Colour users by", ["community", "content", "hate_label"], horizontal=True,
                        format_func={"community": "Louvain community", "content": "posted labels",
                                     "hate_label": "human hate label (Hateful Users)"}.get)
    t1, t2 = st.tabs(["User neighbourhood", "Community"])
    with t1:
        default_user = {"hateful_users": "hu_0", "russian_trolls": ""}.get(name, "u_0001")
        uid = st.text_input("User id", value=default_user)
        lim = st.slider("Max neighbours", 10, 120, 40)
        if uid and st.button("Draw neighbourhood"):
            n, e = uq.ego_subgraph(client, uid, lim)
            show_graph(n, e, color_by, key="ego")
    with t2:
        comms = pd.DataFrame(uq.run(client, uq.Q_COMMUNITIES, prop=community_prop(), limit=50))
        if comms.empty:
            st.info("No communities yet: run them on the **Communities** page.")
        else:
            cid = st.selectbox("Community", comms["community"].tolist(),
                               format_func=lambda c: f"{c}  ({int(comms.set_index('community').loc[c, 'users'])} users)")
            size = st.slider("Users to draw (top PageRank)", 10, 150, 60)
            if st.button("Draw community"):
                n, e = uq.community_subgraph(client, community_prop(), int(cid), size)
                show_graph(n, e, color_by, key="comm")


def _template_inputs(tname: str) -> dict:
    t = TEMPLATES[tname]
    from graph_enhanced_lm.temporal.election_windows import build_windows

    try:
        w = {x.name: x for x in build_windows(cfg().temporal)}
        default_range = (w.get("pre_10_to_1") or list(w.values())[0]).as_params()
    except Exception:
        default_range = {"start": "1970-01-01T00:00:00+00:00", "end": "2100-01-01T00:00:00+00:00"}
    nullable = {"label", "start", "end"} if tname == "community_evidence" else set()
    params: dict = {}
    cols = st.columns(3)
    for i, p in enumerate(t.params):
        col = cols[i % 3]
        if p in ("start", "end"):
            v = col.text_input(p, value=default_range[p], key=f"tp_{tname}_{p}")
            params[p] = (v or None) if p in nullable else v
        elif p == "label":
            opts = (["(any)"] if p in nullable else []) + LABEL_VALUES
            v = col.selectbox(p, opts, index=opts.index("HATE"), key=f"tp_{tname}_{p}")
            params[p] = None if v == "(any)" else v
        elif p in ("limit", "min_count", "min_users"):
            dflt = {"limit": 25, "min_count": 2, "min_users": 5}[p]
            params[p] = int(col.number_input(p, value=dflt, min_value=1, step=1, key=f"tp_{tname}_{p}"))
        elif p == "community_property":
            params[p] = col.text_input(p, value=community_prop(), key=f"tp_{tname}_{p}")
        elif p == "community_id":
            params[p] = int(col.number_input(p, value=0, step=1, key=f"tp_{tname}_{p}",
                                             help="Louvain ids are arbitrary integers: see Communities page"))
        elif p == "score_property":
            params[p] = col.selectbox(p, ["pagerank", "degree", "betweenness"], key=f"tp_{tname}_{p}")
        else:
            params[p] = col.text_input(p, value="u_0001" if p == "user_id" else "", key=f"tp_{tname}_{p}")
    return params


def page_cypher():
    st.header("🔎 Cypher Lab")
    client = need_client()
    synthetic_banner()
    t1, t2, t3 = st.tabs(["Trusted query templates", "Write your own (read-only)", "Tutorial files"])
    with t1:
        tname = st.selectbox("Template", sorted(TEMPLATES), format_func=lambda n: f"{n} - {TEMPLATES[n].description}")
        st.code(TEMPLATES[tname].cypher, language="cypher")
        params = _template_inputs(tname)
        if st.button("▶ Run template", type="primary"):
            try:
                rows = client.run_read(TEMPLATES[tname].cypher, params, max_rows=1000)
                st.success(f"{len(rows)} row(s)")
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True)
                st.download_button("Download CSV", df.to_csv(index=False), f"{tname}.csv", "text/csv")
            except Exception as exc:
                st.error(str(exc))
    with t2:
        q = st.text_area("Cypher", height=160, value="MATCH (u:User)-[:POSTED]->(p:Post)\nRETURN u.user_id AS user, count(p) AS posts\nORDER BY posts DESC LIMIT 10")
        ptxt = st.text_area("Parameters (JSON)", value="{}", height=80)
        if st.button("▶ Run query", type="primary", key="own"):
            try:
                params = json.loads(ptxt or "{}")
                qtype = client.explain_query_type(q, params)
                if qtype != "r":
                    st.error(f"Neo4j reports query type '{qtype}'. The dashboard is read-only; "
                             "use `python scripts/run_cypher.py --write` in the terminal for admin writes.")
                else:
                    rows = client.run_read(q, params, max_rows=1000)
                    st.success(f"{len(rows)} row(s) · query type r (read-only)")
                    st.dataframe(pd.DataFrame(rows), hide_index=True)
            except Exception as exc:
                st.error(str(exc))
    with t3:
        from graph_enhanced_lm.graph.constraints import parse_cypher_file

        files = sorted((PROJECT_ROOT / "cypher").glob("*.cypher"))
        f = st.selectbox("File", files, format_func=lambda p: p.name)
        stmts = parse_cypher_file(f)
        i = st.selectbox("Statement", range(len(stmts)), format_func=lambda k: f"#{k + 1}: {stmts[k].splitlines()[0][:90]}")
        st.code(stmts[i], language="cypher")
        ptxt2 = st.text_area("Parameters (JSON)", value='{"label": "HATE", "user_id": "u_0001", "community_id": 0}',
                             height=70, key="tutparams")
        if st.button("▶ Run statement (reads only)"):
            try:
                params = json.loads(ptxt2 or "{}")
                qtype = client.explain_query_type(stmts[i], params)
                if qtype != "r":
                    st.warning(f"Query type '{qtype}': write/schema statements are not executed from the dashboard.")
                else:
                    st.dataframe(pd.DataFrame(client.run_read(stmts[i], params, max_rows=500)), hide_index=True)
            except Exception as exc:
                st.error(str(exc))


def page_communities():
    st.header("🕸️ Communities & centrality (Neo4j GDS)")
    client = need_client()
    synthetic_banner()
    g = gds_status("comm")
    if isinstance(g, dict) and g.get("version"):
        st.markdown(f"<span class='pill'>GDS {g['version']}</span>", unsafe_allow_html=True)
    else:
        st.warning("GDS not available: check the Docker plugin (Settings → troubleshooting).")
    a, b, c = st.columns([1, 1, 1])
    betw = a.checkbox("Include betweenness (slower)", value=active_dataset() != "hateful_users")
    if b.button("🧮 Run Louvain + centrality", type="primary"):
        run_script("run_graph_algorithms.py", *(["--betweenness"] if betw else []))
    if c.button("🏁 Rank communities"):
        run_script("run_community_analysis.py")
    m = dv.load_json(cfg().path("metrics") / "graph_algorithms_latest.json")
    if m:
        lv = m.get("louvain", {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Communities", lv.get("communityCount", "–"))
        k2.metric("Modularity", f"{lv.get('modularity', 0):.3f}" if lv.get("modularity") is not None else "–")
        k3.metric("Projected users", m.get("projection", {}).get("node_count", "–"))
        rec = m.get("synthetic_ground_truth_recovery")
        k4.metric("Planted-community NMI", rec.get("nmi") if rec else "n/a", help="synthetic data only")
    rk_path = cfg().path("metrics") / "community_ranking.csv"
    if rk_path.exists():
        rk = pd.read_csv(rk_path)
        st.subheader("Investigation-priority ranking")
        st.markdown("<div class='note'>Ranking for <b>human review</b>. A high score reflects label shares and "
                    "interaction structure; it is <b>not</b> evidence of coordination or intent. Weights: "
                    f"<code>{cfg().gds['community_analysis']['priority_weights']}</code></div>", unsafe_allow_html=True)
        show = [col for col in ["priority_rank", "community", "users", "posts", "hate_share",
                                "offensive_share", "hateful_user_share", "pre_election_growth", "internal_ratio",
                                "priority_score", "designation"] if col in rk.columns]
        st.dataframe(rk[show], hide_index=True)
        if HAVE_PLOTLY:
            colour = "hateful_user_share" if "hateful_user_share" in rk else ("hate_share" if "hate_share" in rk else None)
            fig = px.bar(rk.head(20), x=rk.head(20)["community"].astype(str), y="priority_score", color=colour,
                         color_continuous_scale="Reds", template="plotly_white", title="Priority score (top 20)")
            fig.update_layout(height=340, xaxis_title="community", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, key="rank_chart")
        st.subheader("Explain a community")
        cid = st.selectbox("Community", rk["community"].tolist())
        x, y = st.columns([1.4, 1])
        with x:
            color_by = "hate_label" if active_dataset() == "hateful_users" else "content"
            n, e = uq.community_subgraph(client, community_prop(), int(cid), 60)
            show_graph(n, e, color_by, key="comm_expl")
        with y:
            st.markdown("**Most central users (PageRank)**")
            st.dataframe(pd.DataFrame(uq.run(client, uq.Q_COMMUNITY_USERS, prop=community_prop(), cid=int(cid),
                                             limit=10)), hide_index=True)
            if active_dataset() != "hateful_users":
                st.markdown("**Supporting posts (hate-labelled)**")
                ev = client.run_read(TEMPLATES["community_evidence"].cypher,
                                     {"community_property": community_prop(), "community_id": int(cid),
                                      "label": "HATE", "start": None, "end": None, "limit": 15})
                st.dataframe(pd.DataFrame(ev).drop(columns=["text"], errors="ignore"), hide_index=True)
    if active_dataset() == "hateful_users":
        st.subheader("Hateful users: homophily and concentration")
        if st.button("👥 Run hateful-user analysis"):
            run_script("analyze_hateful_users.py")
        res = dv.load_json(cfg().path("metrics") / "hateful_users_analysis.json")
        if res:
            h = res.get("homophily", {})
            k1, k2 = st.columns(2)
            k1.metric("P(retweet hateful | hateful user)", h.get("p_retweet_hateful_given_hateful"))
            k2.metric("P(retweet hateful | normal user)", h.get("p_retweet_hateful_given_normal"))
            if h.get("pairs"):
                st.dataframe(pd.DataFrame(h["pairs"]).T.reset_index(names="pair"), hide_index=True)
            if res.get("concentration", {}).get("available"):
                st.json(res["concentration"], expanded=False)


def page_temporal():
    st.header("📈 Temporal & election-window analysis")
    t = cfg().temporal
    with st.container(border=True):
        st.markdown("**Election settings** (use an officially verified date for real data)")
        a, b, c, d = st.columns([1, 1, 1, 1])
        e_date = a.date_input("Election date", date.fromisoformat(str(t["election_date"])))
        tzs = ["Asia/Kolkata", "America/New_York", "UTC", "Europe/London"]
        tz = b.selectbox("Timezone", tzs, index=tzs.index(t["timezone"]) if t["timezone"] in tzs else 0)
        if c.button("💾 Save"):
            set_yaml_scalar("temporal", "election_date", e_date.isoformat())
            set_yaml_scalar("temporal", "timezone", tz)
            st.rerun()
        preset = d.selectbox("Preset", ["-", "Synthetic (2025-03-15, IST)", "US 2016 (2016-11-08, New York)"])
        if preset != "-" and d.button("Apply preset"):
            dt, z = ("2025-03-15", "Asia/Kolkata") if preset.startswith("Synthetic") else ("2016-11-08", "America/New_York")
            set_yaml_scalar("temporal", "election_date", dt)
            set_yaml_scalar("temporal", "timezone", z)
            st.rerun()
    from graph_enhanced_lm.temporal.election_windows import build_windows
    from graph_enhanced_lm.temporal.trend_analysis import (
        CAUSALITY_NOTE, community_window_activity_neo4j, daily_label_counts_df, window_comparison_df,
        window_comparison_neo4j,
    )

    windows = build_windows(cfg().temporal)
    st.dataframe(pd.DataFrame([{"window": w.name, "start (UTC)": w.start.isoformat(), "end (UTC)": w.end.isoformat(),
                                "days": round(w.days, 1)} for w in windows]), hide_index=True)
    name = active_dataset()
    if not dv.DATASETS[name]["temporal"]:
        st.info(f"'{name}' has no timestamps, so temporal analysis is not available for it.")
        return
    source = st.radio("Source", ["CSV files (fast, no database)", "Neo4j graph (+ per community)"], horizontal=True)
    if not st.button("▶ Run analysis", type="primary"):
        return
    if source.startswith("CSV"):
        from graph_enhanced_lm.data.loaders import read_table
        from graph_enhanced_lm.pipeline import load_processed, merge_predictions, processed_dir

        try:
            ds = load_processed(cfg(), name)
            posts = merge_predictions(ds, read_table(processed_dir(cfg(), name) / "predictions.csv"))
        except Exception as exc:
            st.error(f"{exc}. Preprocess and classify the dataset first (Datasets page).")
            return
        overall = window_comparison_df(posts, windows)
        daily = daily_label_counts_df(posts, tz=cfg().temporal["timezone"])
        if HAVE_PLOTLY and not daily.empty:
            daily["day"] = pd.to_datetime(daily["day"])
            fig = px.line(daily, x="day", y="posts", color="label", template="plotly_white",
                          title="Daily posts per label" + (" (SYNTHETIC)" if name == "synthetic" else ""))
            ed = pd.Timestamp(cfg().temporal["election_date"])
            fig.add_shape(type="line", x0=ed, x1=ed, y0=0, y1=1, yref="paper", line=dict(dash="dash", color="black"))
            fig.add_annotation(x=ed, y=1, yref="paper", text="election day", showarrow=False, yanchor="bottom")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, key="daily")
    else:
        client = need_client()
        overall = window_comparison_neo4j(client, windows)
        comm = community_window_activity_neo4j(client, windows, community_prop())
        if not comm.empty and HAVE_PLOTLY:
            pv = comm.pivot_table(index="community", columns="window", values="hate_posts", fill_value=0)
            pv = pv[[w.name for w in windows if w.name in pv.columns]]
            fig = px.imshow(pv, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                            title="Hate-labelled posts per community and window")
            fig.update_layout(height=max(300, 26 * len(pv)), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, key="heat")
    cols = ["window", "days", "posts", "posts_per_day", "hate_share", "offensive_share"]
    st.dataframe(overall[[c for c in cols if c in overall.columns]], hide_index=True)
    if HAVE_PLOTLY and not overall.empty:
        long = overall.melt(id_vars="window", value_vars=["hate_share", "offensive_share"],
                            var_name="label", value_name="share")
        st.plotly_chart(px.bar(long, x="window", y="share", color="label", barmode="group",
                               template="plotly_white", title="Label shares per window"), key="shares")
    st.markdown(f"<div class='note'>{CAUSALITY_NOTE}</div>", unsafe_allow_html=True)


EXAMPLES = [
    "Which users posted hate-labelled content during the ten days before the election?",
    "Which communities contain a high proportion of hate-labelled posts?",
    "Which users repeatedly amplified the same posts?",
    "How did the network change before, during, and after the election?",
    "Who are the most active users?",
    "What are the 5 most used hashtags in hate-labelled posts?",
]


def _render_answer(d: dict, client, idx: int):
    ans = d.get("answer") or ""
    st.markdown(ans if ans else "_(no answer)_")
    pills = f"<span class='pill'>intent: {d.get('intent')}</span><span class='pill gray'>method: {d.get('method')}</span>"
    if d.get("unknown_citations"):
        pills += f"<span class='pill red'>invalid citations: {d['unknown_citations']}</span>"
    st.markdown(pills, unsafe_allow_html=True)
    ev = pd.DataFrame(d.get("evidence") or [])
    with st.expander(f"📑 Evidence ({len(ev)} items)", expanded=False):
        if ev.empty:
            st.write("No evidence retrieved.")
        else:
            st.dataframe(ev, hide_index=True)
        for n in d.get("evidence_notes") or []:
            st.caption("NOTE: " + n)
    if d.get("cypher"):
        with st.expander("🧾 Cypher that produced the evidence"):
            st.code(d["cypher"], language="cypher")
            st.json(d.get("params") or {}, expanded=False)
    if d.get("errors"):
        with st.expander("⚠️ Errors / retries"):
            for e in d["errors"]:
                st.write(e)
    if client is not None and d.get("evidence"):
        with st.expander("🕸️ Evidence subgraph"):
            if st.button("Draw", key=f"draw_{idx}"):
                recs = d["evidence"]
                n, e = uq.evidence_subgraph(client, recs)
                comm = next((r["community"] for r in recs if r.get("community") is not None), None)
                if not n and comm is not None:
                    n, e = uq.community_subgraph(client, community_prop(), int(comm), 50)
                show_graph(n, e, "content" if active_dataset() != "hateful_users" else "hate_label", key=f"ev{idx}")


def page_ask():
    st.header("💬 Ask the graph (Graph RAG with Qwen)")
    synthetic_banner()
    client = need_client()
    s = cfg().llm_settings()
    ls = llm_status(s["backend"], s.get("base_url", ""), s.get("model", ""))
    with st.expander(f"LLM: {s['backend']} · {s.get('model')} · "
                     f"{ {True: '🟢 reachable', False: '🔴 unreachable', None: '🟡 not checked'}[ls['ok']] }",
                     expanded=ls["ok"] is False):
        st.caption(ls["detail"])
        url = st.text_input("LLM_BASE_URL (Colab tunnel URL + /v1, or http://localhost:11434/v1)", value=s.get("base_url", ""))
        model = st.text_input("LLM_MODEL", value=s.get("model", ""))
        if st.button("💾 Save to .env and re-check"):
            update_env_value("LLM_BASE_URL", url.strip())
            update_env_value("LLM_MODEL", model.strip())
            os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"] = url.strip(), model.strip()
            refresh_caches()
            st.rerun()
    o1, o2, o3 = st.columns(3)
    use_llm = o1.toggle("Use Qwen for the answer", value=bool(ls["ok"]))
    force_t2c = o2.toggle("Force Text-to-Cypher", value=False, help="skip trusted templates")
    if o3.button("🧽 Clear chat"):
        st.session_state["chat"] = []
    st.caption("Examples:")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 3].button(ex, key=f"ex{i}"):
            st.session_state["pending_q"] = ex
    chat = st.session_state.setdefault("chat", [])
    for i, (q, d) in enumerate(chat):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant", avatar="🕸️"):
            _render_answer(d, client, i)
    typed = st.chat_input("Ask about users, communities, labels, time windows…")
    q = typed or st.session_state.pop("pending_q", None)
    if q:
        from graph_enhanced_lm.llm.qwen_client import build_llm_client
        from graph_enhanced_lm.retrieval.graph_rag import build_graph_rag

        with st.spinner("Retrieving evidence and generating the answer…"):
            try:
                llm = build_llm_client(s) if use_llm else None
                rag = build_graph_rag(client, llm, cfg())
                if force_t2c and llm is not None:
                    rag.retriever.retrieve = lambda intent: None  # type: ignore[assignment]
                d = rag.ask(q).to_dict()
            except Exception as exc:
                d = {"answer": f"Error: {exc}", "intent": None, "method": "error", "evidence": [], "errors": [str(exc)]}
        chat.append((q, d))
        st.rerun()


def page_evaluation():
    st.header("🧪 Evaluation & results")
    t1, t2, t3, t4 = st.tabs(["Classifier", "Text-to-Cypher", "Graph RAG", "Results & report"])
    mdir = cfg().path("metrics")
    with t1:
        files = sorted(p for p in mdir.glob("classifier_*_*.json")) if mdir.exists() else []
        if not files:
            st.info("No classifier evaluation yet: Datasets → HateXplain/HASOC → Evaluate classifier.")
        else:
            f = st.selectbox("Result file", files, format_func=lambda p: p.name)
            r = dv.load_json(f)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Accuracy", f"{r['accuracy']:.3f}")
            k2.metric("Macro-F1", f"{r['macro_f1']:.3f}")
            k3.metric("Posts", r["n"])
            k4.metric("Model", str(r.get("model", ""))[-28:])
            pc = pd.DataFrame(r["per_class"]).T.reset_index(names="label")
            st.dataframe(pc, hide_index=True)
            if HAVE_PLOTLY:
                cm = pd.DataFrame(r["confusion_matrix"], index=r["labels"], columns=r["labels"])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                                labels=dict(x="predicted", y="true", color="posts"), title="Confusion matrix")
                fig.update_layout(height=380)
                st.plotly_chart(fig, key="cm")
    with t2:
        a, b = st.columns(2)
        if a.button("🛡️ Safety suite (offline)"):
            run_script("evaluate_text_to_cypher.py", "--safety-only")
        if b.button("🧾 Full benchmark (Neo4j + Qwen)"):
            run_script("evaluate_text_to_cypher.py")
        r = dv.load_json(mdir / "text_to_cypher_eval.json")
        if r:
            st.metric("Safety-suite accuracy", r["safety_suite"]["accuracy"])
            b2 = r.get("benchmark")
            if b2:
                cols = st.columns(5)
                for c, k in zip(cols, ["syntax_valid_rate", "execution_success_rate", "exact_result_match_rate",
                                       "mean_result_f1", "mean_latency_seconds"]):
                    c.metric(k.replace("_", " "), b2[k])
                st.dataframe(pd.DataFrame(b2["items"])[["id", "question", "valid", "executed", "attempts",
                                                        "latency_seconds"]], hide_index=True)
    with t3:
        a, b = st.columns(2)
        if a.button("🧭 Routing accuracy (offline)"):
            run_script("evaluate_graph_rag.py", "--routing-only")
        if b.button("💬 Full RAG run (Neo4j + Qwen)"):
            run_script("evaluate_graph_rag.py")
        r = dv.load_json(mdir / "graph_rag_eval.json")
        if r:
            st.metric("Intent routing accuracy", r["routing_accuracy"])
            st.dataframe(pd.DataFrame(r["routing"]), hide_index=True)
            if r.get("automatic_grounding"):
                st.json(r["automatic_grounding"])
    with t4:
        if st.button("📋 Collect all results (run_baselines)", type="primary"):
            run_script("run_baselines.py")
        md = dv.results_markdown()
        if md:
            st.markdown(md)
        st.subheader("All metric files")
        for p in dv.metric_files():
            st.download_button(f"⬇️ {p.name}", p.read_bytes(), p.name, key=f"dl_{p.name}")


def page_settings():
    st.header("⚙️ Settings & logs")
    st.subheader(".env (secrets masked)")
    st.json(env_summary(ENV_FILE))
    if st.button("🔍 Run check_env.py"):
        run_script("check_env.py")
    st.subheader("Configuration files")
    yfiles = sorted((PROJECT_ROOT / "configs").rglob("*.yaml"))
    f = st.selectbox("File", yfiles, format_func=lambda p: str(p.relative_to(PROJECT_ROOT)))
    st.code(f.read_text(encoding="utf-8"), language="yaml")
    st.subheader("Recent run logs")
    logs = sorted(cfg().path("logs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
    if logs:
        lf = st.selectbox("Log", logs, format_func=lambda p: p.name)
        st.json(dv.load_json(lf), expanded=False)
    if st.button("♻️ Clear dashboard caches"):
        refresh_caches()
        _client.clear()
        st.success("Caches cleared.")


PAGES = {
    "🏠 Overview": page_overview,
    "📦 Datasets": page_datasets,
    "🧠 Knowledge Graph": page_graph,
    "🔎 Cypher Lab": page_cypher,
    "🕸️ Communities": page_communities,
    "📈 Temporal": page_temporal,
    "💬 Ask the Graph": page_ask,
    "🧪 Evaluation": page_evaluation,
    "⚙️ Settings & Logs": page_settings,
}

with st.sidebar:
    st.markdown("### 🕸️ Graph-Enhanced LM")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Active dataset:** `{active_dataset()}`")
    _c, _ = get_client()
    st.markdown(f"Neo4j: {'🟢' if _c else '🔴'}")
    st.caption("Results on synthetic data are not real-world evidence. Communities are candidates for review, "
               "not proof of coordination.")
    if not HAVE_PLOTLY:
        st.warning("pip install -r requirements-ui.txt for charts")

PAGES[choice]()
