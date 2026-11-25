import pandas as pd

from graph_enhanced_lm.data.synthetic import generate_synthetic_dataset


def test_reproducible_with_same_seed(small_synth_cfg):
    a = generate_synthetic_dataset(small_synth_cfg)
    b = generate_synthetic_dataset(small_synth_cfg)
    pd.testing.assert_frame_equal(a.posts, b.posts)
    pd.testing.assert_frame_equal(a.interactions, b.interactions)


def test_different_seed_differs(small_synth_cfg):
    other = dict(small_synth_cfg, seed=small_synth_cfg["seed"] + 1)
    a = generate_synthetic_dataset(small_synth_cfg)
    b = generate_synthetic_dataset(other)
    assert not a.posts["source_labels"].equals(b.posts["source_labels"])


def test_everything_marked_synthetic(synth_ds):
    assert synth_ds.users["synthetic"].all()
    assert synth_ds.posts["synthetic"].all()
    assert synth_ds.interactions["synthetic"].all()
    assert synth_ds.posts["text"].str.startswith("[SYNTHETIC]").all()


def test_has_communities_labels_and_interactions(synth_ds, small_synth_cfg):
    assert synth_ds.users["synthetic_gt_community"].nunique() == small_synth_cfg["n_communities"]
    labels = set(";".join(synth_ds.posts["source_labels"]).split(";"))
    assert {"NORMAL", "HATE", "SPAM", "OFFENSIVE"} <= labels
    assert set(synth_ds.interactions["interaction_type"]) == {"RETWEETED", "REPLIED_TO", "QUOTED", "MENTIONED"}


def test_interaction_endpoints_present(synth_ds):
    i = synth_ds.interactions
    rt = i[i["interaction_type"] == "RETWEETED"]
    assert rt["source_user_id"].notna().all() and rt["target_post_id"].notna().all()
    mn = i[i["interaction_type"] == "MENTIONED"]
    assert mn["source_post_id"].notna().all() and mn["target_user_id"].notna().all()


def test_no_self_retweets(synth_ds):
    author = dict(zip(synth_ds.posts["post_id"], synth_ds.posts["user_id"]))
    rt = synth_ds.interactions[synth_ds.interactions["interaction_type"] == "RETWEETED"]
    assert all(author[p] != u for u, p in zip(rt["source_user_id"], rt["target_post_id"]))


def test_pre_election_hate_ramp_planted(cfg):
    ds = generate_synthetic_dataset(cfg.synthetic)
    users = ds.users.set_index("user_id")["synthetic_gt_community"]
    posts = ds.posts.assign(comm=ds.posts["user_id"].map(users),
                            ts=pd.to_datetime(ds.posts["timestamp"], utc=True))
    hostile = posts[posts["comm"] == "hostile_cluster"]
    e = pd.Timestamp(cfg.synthetic["election_date"], tz="UTC")
    near = hostile[(hostile.ts >= e - pd.Timedelta(days=10)) & (hostile.ts < e)]
    far = hostile[(hostile.ts >= e - pd.Timedelta(days=40)) & (hostile.ts < e - pd.Timedelta(days=10))]
    share = lambda d: d["source_labels"].str.contains("HATE").mean()
    assert share(near) > share(far)
