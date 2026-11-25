"""Reproducible SYNTHETIC social-network dataset generator.

Every record carries ``synthetic=True`` and every post text starts with ``[SYNTHETIC]``.
The texts are neutral placeholders: no real hateful language is produced. Planted
structure (communities, a pre-election hate ramp, a spam amplification network) exists
so the pipeline can be tested against a KNOWN answer. Never present results computed on
this data as real-world evidence.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from graph_enhanced_lm.data.schemas import (
    INTERACTION_COLUMNS, POST_COLUMNS, USER_COLUMNS, CanonicalDataset, join_labels,
)

logger = logging.getLogger(__name__)

PRIMARY = ["NORMAL", "OFFENSIVE", "HATE", "SPAM"]
TOPICS = ["transport", "water_supply", "cricket", "local_markets", "exams", "festival",
          "jobs", "roads", "healthcare", "weather"]
GROUPS = ["GroupA", "GroupB", "GroupC"]  # fictional identity groups
TEMPLATES = {
    "NORMAL": "[SYNTHETIC] Sharing an update about {topic} in the city today #{topic}",
    "OFFENSIVE": "[SYNTHETIC] Rude insult placeholder aimed at another account about {topic} #{topic}",
    "HATE": "[SYNTHETIC] Hostile generalisation placeholder targeting {group} #{topic}",
    "SPAM": "[SYNTHETIC] Limited offer placeholder, click example.invalid now #deal #{topic}",
}
TYPE_PROBS = {"RETWEETED": 0.5, "REPLIED_TO": 0.25, "QUOTED": 0.1, "MENTIONED": 0.15}


def _label_probs(profile: dict[str, Any], ts: datetime, election: datetime) -> np.ndarray:
    """Per-post label probabilities; the HATE share ramps up 10 days pre-election."""
    probs = np.array([float(profile[k]) for k in PRIMARY])
    ramp = float(profile.get("pre_election_hate_ramp", 0.0))
    if ramp and election - timedelta(days=10) <= ts < election:
        probs[PRIMARY.index("HATE")] += ramp
    return probs / probs.sum()


def generate_synthetic_dataset(cfg: dict[str, Any]) -> CanonicalDataset:
    """Generate users, posts and interactions from ``configs/synthetic.yaml`` settings."""
    rng = np.random.default_rng(int(cfg["seed"]))
    profiles = cfg["community_profiles"]
    n_comm = int(cfg["n_communities"])
    if n_comm != len(profiles):
        raise ValueError("n_communities must equal the number of community_profiles")
    election = datetime.fromisoformat(cfg["election_date"]).replace(tzinfo=timezone.utc)
    start = election - timedelta(days=int(cfg["days_before"]))
    span_seconds = int((int(cfg["days_before"]) + int(cfg["days_after"])) * 86400)
    end = start + timedelta(seconds=span_seconds)
    p_ct = float(cfg["community_targeted_given_hate"])

    # ---- users ----
    n_users = int(cfg["n_users"])
    communities = np.array([i % n_comm for i in range(n_users)])
    rng.shuffle(communities)
    users = pd.DataFrame({
        "user_id": [f"u_{i:04d}" for i in range(n_users)],
        "username": [f"synthetic_user_{i:04d}" for i in range(n_users)],
        "dataset_source": "synthetic",
        "synthetic": True,
        "source_bot_label": None,
        "synthetic_gt_community": [profiles[c]["name"] for c in communities],
        "user_hate_label": None,
    })
    members = {c: np.where(communities == c)[0] for c in range(n_comm)}

    posts: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    counter = {"post": 0, "inter": 0}

    def make_post(author: int, ts: datetime, conversation: str | None) -> dict[str, Any]:
        comm = int(communities[author])
        probs = _label_probs(profiles[comm], ts, election)
        primary = PRIMARY[int(rng.choice(len(PRIMARY), p=probs))]
        labels = [primary]
        if primary == "HATE" and rng.random() < p_ct:
            labels.append("COMMUNITY_TARGETED")
        topic = TOPICS[int(rng.integers(len(TOPICS)))]
        group = GROUPS[int(rng.integers(len(GROUPS)))]
        pid = f"p_{counter['post']:06d}"
        counter["post"] += 1
        return {
            "post_id": pid, "user_id": users.at[author, "user_id"],
            "text": TEMPLATES[primary].format(topic=topic, group=group),
            "timestamp": ts.replace(microsecond=0).isoformat(), "language": "en",
            "source_labels": join_labels(labels), "conversation_id": conversation or pid,
            "dataset_source": "synthetic", "synthetic": True, "is_stub": False,
        }

    def pick_actor(author: int) -> int:
        comm = int(communities[author])
        if rng.random() < float(cfg["intra_community_interaction_prob"]):
            pool = members[comm]
        else:
            pool = np.array([u for c, m in members.items() if c != comm for u in m])
        actor = int(rng.choice(pool))
        return actor if actor != author else int(rng.choice(pool))

    # ---- original posts ----
    originals: list[tuple[dict[str, Any], int]] = []
    for author in range(n_users):
        n_posts = max(1, int(rng.poisson(float(cfg["mean_posts_per_user"]))))
        for _ in range(n_posts):
            ts = start + timedelta(seconds=int(rng.integers(span_seconds)))
            post = make_post(author, ts, None)
            posts.append(post)
            originals.append((post, author))

    # ---- interactions (and reply / quote posts) ----
    type_names = list(TYPE_PROBS)
    type_p = np.array(list(TYPE_PROBS.values()))
    for post, author in originals:
        mean = float(cfg["interactions_per_post_mean"])
        if "SPAM" in post["source_labels"]:
            mean *= 3  # planted amplification of spam posts
        base_ts = datetime.fromisoformat(post["timestamp"])
        for _ in range(int(rng.poisson(mean))):
            actor = pick_actor(author)
            if actor == author:
                continue
            itype = type_names[int(rng.choice(len(type_names), p=type_p))]
            ts = min(base_ts + timedelta(seconds=int(rng.integers(60, 48 * 3600))), end)
            iid = f"i_{counter['inter']:07d}"
            counter["inter"] += 1
            row = {"interaction_id": iid, "interaction_type": itype,
                   "source_user_id": None, "source_post_id": None, "target_post_id": None,
                   "target_user_id": None, "timestamp": ts.replace(microsecond=0).isoformat(),
                   "dataset_source": "synthetic", "synthetic": True}
            if itype == "RETWEETED":
                row.update(source_user_id=users.at[actor, "user_id"],
                           target_post_id=post["post_id"])
            elif itype in ("REPLIED_TO", "QUOTED"):
                new_post = make_post(actor, ts, post["conversation_id"])
                posts.append(new_post)
                row.update(source_user_id=users.at[actor, "user_id"],
                           source_post_id=new_post["post_id"], target_post_id=post["post_id"])
            else:  # MENTIONED: the original post mentions the actor
                row.update(source_user_id=users.at[author, "user_id"],
                           source_post_id=post["post_id"],
                           target_user_id=users.at[actor, "user_id"],
                           timestamp=post["timestamp"])
            interactions.append(row)

    ds = CanonicalDataset(
        name="synthetic",
        users=users[USER_COLUMNS],
        posts=pd.DataFrame(posts, columns=POST_COLUMNS),
        interactions=pd.DataFrame(interactions, columns=INTERACTION_COLUMNS),
        capabilities={"text": True, "user_ids": True, "timestamps": True,
                      "interactions": True, "follows": False, "labels": True},
        notes=["SYNTHETIC DATA - planted structure for testing only; not real-world evidence."],
    )
    logger.info("Generated synthetic dataset: %s", ds.summary())
    return ds
