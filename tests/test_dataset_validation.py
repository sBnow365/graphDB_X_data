import pandas as pd

from graph_enhanced_lm.data.preprocessing import preprocess_canonical
from graph_enhanced_lm.data.schemas import CanonicalDataset
from graph_enhanced_lm.data.validators import validate_canonical


def _req(cfg):
    return {k: v["required"] for k, v in cfg.dataset["canonical_tables"].items() if k != "classification_posts"}


def test_clean_synthetic_passes(cfg, synth_ds):
    rep = validate_canonical(preprocess_canonical(synth_ds), _req(cfg), cfg.dataset["interaction_types"])
    assert rep.ok, rep.pretty()


def test_duplicate_posts_detected(cfg, synth_ds):
    posts = synth_ds.posts.copy()
    dup = posts.iloc[[0]]
    bad = CanonicalDataset("bad", synth_ds.users, pd.concat([posts, dup]), synth_ds.interactions)
    rep = validate_canonical(bad, _req(cfg), cfg.dataset["interaction_types"])
    assert not rep.ok and any("duplicate" in e for e in rep.errors)


def test_missing_column_detected(cfg, synth_ds):
    bad = CanonicalDataset("bad", synth_ds.users, synth_ds.posts.drop(columns=["timestamp"]),
                           synth_ds.interactions)
    rep = validate_canonical(bad, _req(cfg), cfg.dataset["interaction_types"])
    assert not rep.ok and "missing required columns" in rep.errors[0]


def test_unknown_interaction_type(cfg, synth_ds):
    inter = synth_ds.interactions.copy()
    inter.loc[inter.index[0], "interaction_type"] = "LIKED"
    rep = validate_canonical(CanonicalDataset("x", synth_ds.users, synth_ds.posts, inter),
                             _req(cfg), cfg.dataset["interaction_types"])
    assert any("unknown interaction types" in e for e in rep.errors)


def test_unparseable_timestamp(cfg, synth_ds):
    posts = synth_ds.posts.copy()
    posts.loc[posts.index[0], "timestamp"] = "not-a-date"
    rep = validate_canonical(CanonicalDataset("x", synth_ds.users, posts, synth_ds.interactions),
                             _req(cfg), cfg.dataset["interaction_types"])
    assert any("unparseable" in e for e in rep.errors)
