from pathlib import Path

from graph_enhanced_lm.classification.label_mapping import LabelMapper
from graph_enhanced_lm.config import load_dataset_mapping
from graph_enhanced_lm.data.dataset_adapters import HASOCAdapter, HateXplainAdapter, TwiBot22Adapter

FIX = Path(__file__).parent / "fixtures"


def test_hatexplain_majority_and_target(cfg):
    m = load_dataset_mapping("hatexplain")
    m["files"] = {"dataset_json": str(FIX / "hatexplain/dataset.json"),
                  "divisions_json": str(FIX / "hatexplain/post_id_divisions.json")}
    df = HateXplainAdapter(m, LabelMapper.from_taxonomy(cfg.label_taxonomy, "hatexplain")).load().set_index("post_id")
    assert df.loc["1_twitter", "source_labels"] == "HATE;COMMUNITY_TARGETED"
    assert df.loc["2_gab", "source_labels"] == "NORMAL"
    assert df.loc["3_twitter", "source_labels"] == "UNKNOWN"          # 3-way tie, no majority
    assert df.loc["3_twitter", "split"] == "test"
    assert df["group_id"].isna().all()                               # no user ids in HateXplain


def test_hasoc_columns_and_fine_labels(cfg):
    m = load_dataset_mapping("hasoc")
    m["files"] = {"data_file": str(FIX / "hasoc/sample.tsv")}
    df = HASOCAdapter(m, LabelMapper.from_taxonomy(cfg.label_taxonomy, "hasoc")).load().set_index("post_id")
    assert df.loc["hasoc_h1", "source_labels"] == "HATE"
    assert df.loc["hasoc_h2", "source_labels"] == "NORMAL"
    assert df.loc["hasoc_h3", "source_labels"] == "OFFENSIVE"


def test_twibot_graph_mapping():
    m = load_dataset_mapping("twibot22")
    m["files"] = {"users": str(FIX / "twibot22/user.json"), "tweets_glob": str(FIX / "twibot22/tweet_*.json"),
                  "edges": str(FIX / "twibot22/edge.csv"), "labels": str(FIX / "twibot22/label.csv")}
    ds = TwiBot22Adapter(m).load()
    assert set(ds.posts["post_id"]) == {"t1", "t2", "t3"}
    assert ds.posts.set_index("post_id").loc["t1", "user_id"] == "u1"
    assert ds.posts["source_labels"].isna().all()                    # bot labels are NOT hate labels
    types = dict(zip(ds.interactions["interaction_type"], ds.interactions.index))
    assert set(types) == {"RETWEETED", "REPLIED_TO", "MENTIONED"}
    rt = ds.interactions[ds.interactions["interaction_type"] == "RETWEETED"].iloc[0]
    assert rt["source_user_id"] == "u2" and rt["target_post_id"] == "t1"
    assert ds.users.set_index("user_id").loc["u2", "source_bot_label"] == "bot"
    assert len(ds.follows) == 1 and ds.follows.iloc[0]["source_user_id"] == "u2"
