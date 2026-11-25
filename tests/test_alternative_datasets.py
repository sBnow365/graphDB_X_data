from pathlib import Path

import pandas as pd

from graph_enhanced_lm.config import load_dataset_mapping
from graph_enhanced_lm.data.dataset_adapters import HatefulUsersAdapter, RussianTrollsAdapter
from graph_enhanced_lm.data.preprocessing import preprocess_canonical
from graph_enhanced_lm.data.validators import validate_canonical
from graph_enhanced_lm.evaluation.user_label_analysis import concentration, homophily_table
from graph_enhanced_lm.gds.projections import INTERACTION_UNION
from graph_enhanced_lm.gds.statistics import score_communities

FIX = Path(__file__).parent / "fixtures"


def _req(cfg):
    return {k: v["required"] for k, v in cfg.dataset["canonical_tables"].items() if k != "classification_posts"}


def hateful():
    m = load_dataset_mapping("hateful_users")
    m["files"] = {"users": str(FIX / "hateful_users/users_neighborhood_anon.csv"),
                  "edges": str(FIX / "hateful_users/users.edges")}
    return preprocess_canonical(HatefulUsersAdapter(m).load())


def trolls():
    m = load_dataset_mapping("russian_trolls")
    m["files"] = {"tweets": str(FIX / "russian_trolls/tweets.csv"), "users": str(FIX / "russian_trolls/users.csv")}
    return preprocess_canonical(RussianTrollsAdapter(m).load())


def test_hateful_users_labels_and_edges(cfg):
    ds = hateful()
    lab = ds.users.set_index("user_id")["user_hate_label"]
    assert lab["hu_0"] == "HATEFUL" and lab["hu_2"] == "NORMAL" and pd.isna(lab["hu_4"])
    # self-loop removed, duplicate 0->1 merged into weight 2
    e = ds.user_edges.set_index(["source_user_id", "target_user_id"])
    assert ("hu_0", "hu_0") not in e.index
    assert e.loc[("hu_0", "hu_1"), "weight"] == 2.0
    assert ds.posts.empty and ds.interactions.empty
    assert validate_canonical(ds, _req(cfg), cfg.dataset["interaction_types"]).ok


def test_hateful_users_comma_separated_edges(tmp_path):
    (tmp_path / "users.edges").write_text("0,1\n2,3\n")
    (tmp_path / "u.csv").write_text((FIX / "hateful_users/users_neighborhood_anon.csv").read_text())
    m = load_dataset_mapping("hateful_users")
    m["files"] = {"users": str(tmp_path / "u.csv"), "edges": str(tmp_path / "users.edges")}
    assert len(HatefulUsersAdapter(m).load().user_edges) == 2


def test_homophily_lift():
    ds = hateful()
    h = homophily_table(ds.users, ds.user_edges)
    assert h["pairs"]["HATEFUL->HATEFUL"]["edges"] == 2          # 0->1 (weight 2 counts once) and 1->0
    assert h["p_retweet_hateful_given_hateful"] == 1.0
    assert h["p_retweet_hateful_given_normal"] == 0.0


def test_concentration_and_scoring():
    stats = pd.DataFrame({"community": [7, 9], "users": [3, 3], "hateful_users": [2, 0],
                          "annotated_users": [2, 2], "hateful_user_share": [1.0, 0.0]})
    c = concentration(stats, top_k=1)
    assert c["share_of_hateful_in_top_1"] == 1.0 and c["overall_hateful_rate"] == 0.5
    out = score_communities(stats, {"hateful_user_share": 1.0})
    assert out.iloc[0]["community"] == 7
    assert out.set_index("community").loc[9, "designation"] == "no hate/spam signal observed"


def test_russian_trolls_mapping(cfg):
    ds = trolls()
    posts = ds.posts.set_index("post_id")
    assert "105" not in posts.index                         # outside date window
    assert {"101", "102", "103", "104"} <= set(posts.index)
    assert posts.loc["999", "is_stub"] and posts.loc["999", "user_id"] == "u_outsiderc"  # inferred RT author
    assert not posts.loc["101", "is_stub"]                   # real post beats stub with same id
    i = ds.interactions
    rt = i[i.interaction_type == "RETWEETED"]
    assert set(zip(rt.source_user_id, rt.target_post_id)) == {("u_trollb", "101"), ("u_trollb", "999")}
    mn = i[i.interaction_type == "MENTIONED"]
    # RT author is not double-counted as a mention; other mentions kept
    assert ("102", "u_trolla") not in set(zip(mn.source_post_id, mn.target_user_id))
    assert ("101", "u_candidatex") in set(zip(mn.source_post_id, mn.target_user_id))
    rp = i[i.interaction_type == "REPLIED_TO"].iloc[0]
    assert rp.source_post_id == "104" and rp.target_post_id == "102"
    assert (ds.users["source_bot_label"] == "ira_linked").all()
    assert ds.posts.loc[~ds.posts.is_stub, "timestamp"].str.startswith("2016-10").all()
    assert validate_canonical(ds, _req(cfg), cfg.dataset["interaction_types"]).ok


def test_projection_has_user_edge_branch():
    assert "RETWEETED_USER" in INTERACTION_UNION and "$w_retweet_user" in INTERACTION_UNION


def test_classifier_skips_posts_without_text(cfg, tmp_path, monkeypatch):
    from graph_enhanced_lm import pipeline
    ds = trolls()
    monkeypatch_dir = tmp_path / "russian_trolls"
    monkeypatch_dir.mkdir()
    ds.posts.assign(source_labels="NORMAL").to_csv(monkeypatch_dir / "posts.csv", index=False)
    ds.users.to_csv(monkeypatch_dir / "users.csv", index=False)
    ds.interactions.to_csv(monkeypatch_dir / "interactions.csv", index=False)
    monkeypatch.setattr(pipeline, "processed_dir", lambda c, n: monkeypatch_dir)
    preds = pipeline.classify_dataset(cfg, "russian_trolls", "source_labels").set_index("post_id")
    assert preds.loc["999", "label_source"] == "unlabeled" and preds.loc["999", "labels"] == "UNKNOWN"
    assert preds.loc["101", "primary_label"] == "NORMAL"
