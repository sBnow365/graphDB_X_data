import networkx as nx
import pandas as pd
import pytest

from graph_enhanced_lm.classification.evaluation import evaluate_single_label
from graph_enhanced_lm.evaluation.graph_metrics import (
    ground_truth_recovery, interaction_graph_from_tables, modularity_of, offline_louvain,
)
from graph_enhanced_lm.evaluation.llm_metrics import result_set_match
from graph_enhanced_lm.gds.statistics import add_pre_election_growth, score_communities


def test_classifier_metrics_known_values():
    r = evaluate_single_label(["HATE", "NORMAL", "NORMAL", "HATE"], ["HATE", "NORMAL", "HATE", "HATE"],
                              ["HATE", "NORMAL"])
    assert r["accuracy"] == 0.75
    assert r["per_class"]["NORMAL"]["recall"] == 0.5
    assert r["confusion_matrix"] == [[2, 0], [1, 1]]


def test_metrics_reject_empty():
    with pytest.raises(ValueError):
        evaluate_single_label([], [])


def test_result_set_match_order_insensitive():
    g = [{"user_id": "a"}, {"user_id": "b"}]
    p = [{"uid": "b"}, {"uid": "a"}]
    assert result_set_match(g, p)["exact"]
    assert result_set_match(g, [{"x": "a"}])["recall"] == 0.5


def test_offline_louvain_recovers_planted_structure(synth_ds, cfg):
    g = interaction_graph_from_tables(synth_ds.posts, synth_ds.interactions, cfg.gds["projection"]["weights"])
    assign = offline_louvain(g)
    rec = ground_truth_recovery(assign, synth_ds.users)
    assert rec["n"] > 0 and rec["nmi"] > 0.5      # planted communities are largely recovered
    assert modularity_of(g, assign) > 0.3


def test_score_is_transparent_and_labelled():
    stats = pd.DataFrame({"community": [0, 1, 2], "hate_share": [0.3, 0.0, 0.05],
                          "spam_share": [0.0, 0.0, 0.6], "offensive_share": [0.1, 0.0, 0.0],
                          "internal_ratio": [0.9, 0.5, 0.8]})
    out = score_communities(stats, {"hate_share": 0.5, "spam_share": 0.5})
    assert list(out["community"])[:2] in ([0, 2], [2, 0])
    assert out.set_index("community").loc[1, "designation"] == "no hate/spam signal observed"
    assert "not evidence of coordination" in out["interpretation"].iloc[0]


def test_pre_election_growth():
    stats = pd.DataFrame({"community": [0, 1]})
    act = pd.DataFrame({"community": [0, 0, 1], "window": ["near", "far", "far"],
                        "hate_posts": [20, 10, 5], "spam_posts": [0, 0, 0]})
    out = add_pre_election_growth(stats, act, "near", "far", 10, 20).set_index("community")
    assert out.loc[0, "pre_election_growth"] == 1.5     # 20/10 - 10/20
    assert out.loc[1, "pre_election_growth"] == -0.25
