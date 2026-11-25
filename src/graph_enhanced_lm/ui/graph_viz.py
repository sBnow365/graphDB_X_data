"""Turn node/edge lists into an interactive network (pyvis) or a static fallback (plotly)."""
from __future__ import annotations

import html
from typing import Any

import networkx as nx

PALETTE = ["#0f766e", "#7c3aed", "#ea580c", "#2563eb", "#db2777", "#65a30d", "#0891b2",
           "#ca8a04", "#9333ea", "#dc2626", "#475569", "#16a34a"]
LABEL_COLORS = {"HATE": "#dc2626", "OFFENSIVE": "#ea580c", "SPAM": "#ca8a04",
                "COMMUNITY_TARGETED": "#db2777", "NORMAL": "#64748b", "UNKNOWN": "#94a3b8"}
HATE_USER_COLORS = {"HATEFUL": "#dc2626", "NORMAL": "#2563eb"}


def node_color(n: dict[str, Any], color_by: str) -> str:
    if n.get("kind") == "Post":
        return LABEL_COLORS.get(str(n.get("label") or "UNKNOWN"), "#94a3b8")
    if color_by == "hate_label":
        return HATE_USER_COLORS.get(str(n.get("hate_label")), "#cbd5e1")
    if color_by == "content":
        if (n.get("hate_posts") or 0) > 0:
            return "#dc2626"
        if (n.get("spam_posts") or 0) > 0:
            return "#ca8a04"
        return "#2563eb"
    comm = n.get("community")
    if comm is None:
        return "#94a3b8"
    key = int(comm) if str(comm).lstrip("-").isdigit() else sum(map(ord, str(comm)))
    return PALETTE[key % len(PALETTE)]


def build_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                color_by: str = "community") -> nx.DiGraph:
    """Nodes need 'id'; edges need 'source','target' (optional 'weight','type')."""
    g = nx.DiGraph()
    pr = [float(n.get("pagerank") or 0) for n in nodes]
    top = max(pr) if pr and max(pr) > 0 else 1.0
    for n in nodes:
        nid = str(n["id"])
        kind = n.get("kind", "User")
        size = 8 if kind == "Post" else 12 + 28 * (float(n.get("pagerank") or 0) / top)
        tip = "<br>".join(f"{k}: {html.escape(str(v))}" for k, v in n.items() if v not in (None, ""))
        g.add_node(nid, kind=kind, color=node_color(n, color_by), size=size, title=tip,
                   label=nid if kind == "User" else "")
    for e in edges:
        s, t = str(e["source"]), str(e["target"])
        for x in (s, t):
            if x not in g:
                g.add_node(x, kind="User", color="#cbd5e1", size=10, title=x, label=x)
        w = float(e.get("weight") or 1.0)
        if g.has_edge(s, t):
            g[s][t]["weight"] += w
        else:
            g.add_edge(s, t, weight=w, type=e.get("type", "INTERACTS"))
    return g


def to_pyvis_html(g: nx.DiGraph, height: int = 560, dark: bool = False) -> str:
    """Interactive HTML (drag, zoom, hover). Raises ImportError if pyvis is missing."""
    from pyvis.network import Network  # type: ignore

    net = Network(height=f"{height}px", width="100%", directed=True, cdn_resources="remote",
                  bgcolor="#0f172a" if dark else "#ffffff", font_color="#e2e8f0" if dark else "#0f172a")
    for nid, a in g.nodes(data=True):
        net.add_node(nid, label=a.get("label", ""), title=a.get("title", nid), color=a.get("color"),
                     size=a.get("size", 10), shape="square" if a.get("kind") == "Post" else "dot")
    for s, t, a in g.edges(data=True):
        net.add_edge(s, t, value=a.get("weight", 1.0), title=f"{a.get('type')} x{a.get('weight', 1):g}",
                     color="#94a3b8")
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -6000, "springLength": 120},'
                    ' "stabilization": {"iterations": 150}}, "edges": {"arrows": {"to": {"scaleFactor": 0.4}},'
                    ' "smooth": false}, "interaction": {"hover": true, "navigationButtons": true}}')
    return net.generate_html()


def to_plotly(g: nx.DiGraph):
    """Static fallback figure (needs plotly)."""
    import plotly.graph_objects as go  # type: ignore

    pos = nx.spring_layout(g.to_undirected(), seed=42, k=None)
    ex, ey = [], []
    for s, t in g.edges():
        ex += [pos[s][0], pos[t][0], None]
        ey += [pos[s][1], pos[t][1], None]
    nodes = list(g.nodes(data=True))
    fig = go.Figure([
        go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#cbd5e1"), hoverinfo="skip"),
        go.Scatter(x=[pos[n][0] for n, _ in nodes], y=[pos[n][1] for n, _ in nodes], mode="markers",
                   marker=dict(size=[a["size"] for _, a in nodes], color=[a["color"] for _, a in nodes],
                               line=dict(width=0.5, color="#ffffff")),
                   text=[a["title"].replace("<br>", "\n") for _, a in nodes], hoverinfo="text"),
    ])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=520,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white")
    return fig
