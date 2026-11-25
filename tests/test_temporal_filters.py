from datetime import datetime, timezone

import pandas as pd
import pytest

from graph_enhanced_lm.temporal.date_filters import DateRange, day_range, offset_range
from graph_enhanced_lm.temporal.election_windows import build_windows, nearest_pre_windows
from graph_enhanced_lm.temporal.trend_analysis import daily_label_counts_df, window_comparison_df


def test_offset_range_timezone():
    r = offset_range("2025-03-15", -10, 0, "Asia/Kolkata")
    assert r.start.isoformat() == "2025-03-04T18:30:00+00:00"   # 5 Mar 00:00 IST
    assert r.end.isoformat() == "2025-03-14T18:30:00+00:00"     # 15 Mar 00:00 IST
    assert r.days == 10


def test_day_range_inclusive_end():
    r = day_range("2025-03-01", "2025-03-01", "UTC")
    assert r.days == 1
    assert r.contains(datetime(2025, 3, 1, 23, 59, tzinfo=timezone.utc))
    assert not r.contains(datetime(2025, 3, 2, 0, 0, tzinfo=timezone.utc))


def test_invalid_range():
    with pytest.raises(ValueError):
        DateRange(datetime(2025, 1, 2, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        DateRange(datetime(2025, 1, 1), datetime(2025, 1, 2))   # naive


def test_build_windows(cfg):
    ws = build_windows(cfg.temporal)
    assert [w.name for w in ws] == [w["name"] for w in cfg.temporal["windows"]]
    far, near = nearest_pre_windows(ws)
    assert near.name == "pre_10_to_1" and far.name == "pre_30_to_11"


def test_build_windows_requires_date(cfg):
    t = dict(cfg.temporal, election_date=None)
    with pytest.raises(ValueError):
        build_windows(t)


def test_window_comparison_and_missing_timestamps(cfg):
    posts = pd.DataFrame({
        "post_id": ["a", "b", "c", "d"],
        "timestamp": ["2025-03-10T10:00:00+00:00", "2025-03-12T10:00:00+00:00",
                      "2025-03-20T10:00:00+00:00", None],
        "labels": ["HATE;COMMUNITY_TARGETED", "NORMAL", "SPAM", "HATE"]})
    ws = build_windows(cfg.temporal)
    out = window_comparison_df(posts, ws).set_index("window")
    assert out.loc["pre_10_to_1", "posts"] == 2
    assert out.loc["pre_10_to_1", "hate_share"] == 0.5
    assert out.loc["post_1_to_10", "spam_posts"] == 1
    assert out["posts"].sum() == 3      # the post without timestamp is excluded


def test_daily_counts():
    posts = pd.DataFrame({"post_id": ["a", "b"], "timestamp": ["2025-03-01T20:00:00Z", "2025-03-01T21:00:00Z"],
                          "labels": ["HATE", "HATE"]})
    d = daily_label_counts_df(posts, tz="Asia/Kolkata")    # both are 2 March in IST
    assert len(d) == 1 and str(d.iloc[0]["day"]) == "2025-03-02" and d.iloc[0]["posts"] == 2
