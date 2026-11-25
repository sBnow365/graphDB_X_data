import pandas as pd

from graph_enhanced_lm.data.preprocessing import (
    clean_interactions, clean_posts, extract_hashtags, normalize_id, normalize_timestamp,
)
from graph_enhanced_lm.data.splitting import assign_splits, check_group_leakage


def test_normalize_id():
    assert normalize_id(123.0) == "123"
    assert normalize_id("  abc ") == "abc"
    assert normalize_id(float("nan")) is None
    assert normalize_id("") is None


def test_normalize_timestamp_formats():
    assert normalize_timestamp("2025-03-01T10:00:00Z") == "2025-03-01T10:00:00+00:00"
    assert normalize_timestamp("2025-03-01 15:30:00", assume_tz="Asia/Kolkata") == "2025-03-01T10:00:00+00:00"
    assert normalize_timestamp("Wed Oct 10 20:19:24 +0000 2018") == "2018-10-10T20:19:24+00:00"
    assert normalize_timestamp("garbage") is None
    assert normalize_timestamp(None) is None


def test_hashtags():
    assert extract_hashtags("Hi #Cricket and #cricket #jobs!") == ["cricket", "jobs"]
    assert extract_hashtags(None) == []


def test_dedupe_posts():
    df = pd.DataFrame({"post_id": ["1", "1", "2"], "user_id": ["a", "a", "b"],
                       "timestamp": ["2025-01-01", "2025-01-01", None], "synthetic": [True] * 3})
    out = clean_posts(df)
    assert list(out["post_id"]) == ["1", "2"]
    assert out.loc[out.post_id == "2", "timestamp"].isna().all()


def test_interaction_id_generated_deterministically():
    df = pd.DataFrame({"interaction_type": ["retweeted"], "source_user_id": ["u1"], "target_post_id": ["p1"]})
    a, b = clean_interactions(df), clean_interactions(df)
    assert a["interaction_id"].iloc[0] == b["interaction_id"].iloc[0]
    assert a["interaction_type"].iloc[0] == "RETWEETED"


def test_group_split_has_no_leakage():
    df = pd.DataFrame({"post_id": [str(i) for i in range(200)], "group_id": [f"g{i % 20}" for i in range(200)]})
    out = assign_splits(df, 0.8, 0.1, 42, "group_id")
    assert check_group_leakage(out, "group_id") == 0
    assert set(out["split"]) == {"train", "val", "test"}


def test_existing_split_respected():
    df = pd.DataFrame({"post_id": ["a", "b"], "split": ["test", "train"]})
    assert list(assign_splits(df)["split"]) == ["test", "train"]
