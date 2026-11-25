"""INTEGRATION: full synthetic pipeline on Neo4j + GDS (no LLM needed)."""
import pytest

from graph_enhanced_lm.gds.community_detection import louvain_write
from graph_enhanced_lm.gds.projections import drop_graph, project_interaction_graph
from graph_enhanced_lm.gds.statistics import fetch_community_stats
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.retrieval.graph_rag import build_graph_rag

pytestmark = pytest.mark.integration


def test_projection_louvain_stats_and_rag_without_llm(cfg):
    with Neo4jClient.from_config(cfg) as c:
        proj = project_interaction_graph(c, "test_projection", cfg.gds["projection"]["weights"])
        assert proj["node_count"] > 0
        res = louvain_write(c, "test_projection", {**cfg.gds["louvain"], "write_property": "test_louvain"})
        assert res["communityCount"] >= 2
        stats = fetch_community_stats(c, "test_louvain", 1)
        assert not stats.empty and {"hate_share", "spam_share"} <= set(stats.columns)
        drop_graph(c, "test_projection")
        rag = build_graph_rag(c, None, cfg)
        out = rag.ask("Which users posted hate-labelled content during the ten days before the election?")
        assert out.method.startswith("template:")
