"""Source-specific adapters: each converts one dataset into the canonical schema.

Source-specific field names live in ``configs/dataset_mappings/<name>.yaml``; this
module contains only the conversion LOGIC. Datasets are never joined with each other.
"""
from __future__ import annotations

import glob
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from graph_enhanced_lm.classification.label_mapping import LabelMapper
from graph_enhanced_lm.config import PROJECT_ROOT, load_dataset_mapping
from graph_enhanced_lm.data.loaders import iter_json_array, read_table
from graph_enhanced_lm.data.preprocessing import normalize_id, normalize_timestamp
from graph_enhanced_lm.data.schemas import (
    CLASSIFICATION_COLUMNS, FOLLOW_COLUMNS, INTERACTION_COLUMNS, POST_COLUMNS, USER_COLUMNS,
    USER_EDGE_COLUMNS, CanonicalDataset, ensure_columns, join_labels,
)
from graph_enhanced_lm.utils.errors import DatasetError

logger = logging.getLogger(__name__)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class GraphDatasetAdapter(ABC):
    """Adapter producing users/posts/interactions for the knowledge graph."""

    def __init__(self, mapping: dict[str, Any]):
        self.mapping = mapping

    @abstractmethod
    def load(self) -> CanonicalDataset: ...


class ClassificationDatasetAdapter(ABC):
    """Adapter producing a labelled text table for classifier training/evaluation."""

    def __init__(self, mapping: dict[str, Any], mapper: LabelMapper):
        self.mapping = mapping
        self.mapper = mapper

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


# --------------------------------------------------------------------- synthetic
class SyntheticAdapter(GraphDatasetAdapter):
    """Reads the CSVs written by scripts/generate_synthetic_data.py."""

    def load(self) -> CanonicalDataset:
        files = self.mapping["files"]
        for key in ("users", "posts", "interactions"):
            if not _abs(files[key]).exists():
                raise DatasetError(f"{files[key]} not found. Run: python scripts/generate_synthetic_data.py")
        return CanonicalDataset(
            name="synthetic",
            users=ensure_columns(read_table(_abs(files["users"])), USER_COLUMNS),
            posts=ensure_columns(read_table(_abs(files["posts"])), POST_COLUMNS),
            interactions=ensure_columns(read_table(_abs(files["interactions"])), INTERACTION_COLUMNS),
            capabilities=self.mapping.get("capabilities", {}),
            notes=["SYNTHETIC DATA - not real-world evidence."],
        )


# --------------------------------------------------------------------- HateXplain
class HateXplainAdapter(ClassificationDatasetAdapter):
    """HateXplain dataset.json -> classification table.

    Each post has several annotators. The project label is the MAJORITY label; a 3-way
    tie becomes UNKNOWN. If a majority of annotators named a target community (not
    'None') on a hate/offensive post, COMMUNITY_TARGETED is added.
    """

    def load(self) -> pd.DataFrame:
        files, fields = self.mapping["files"], self.mapping["fields"]
        path = _abs(files["dataset_json"])
        if not path.exists():
            raise DatasetError(f"{path} not found. See docs/DATASET_CREATION_GUIDE.md (HateXplain).")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        division: dict[str, str] = {}
        div_path = _abs(files.get("divisions_json", ""))
        if div_path.exists() and div_path.is_file():
            with div_path.open("r", encoding="utf-8") as fh:
                for split, ids in json.load(fh).items():
                    for pid in ids:
                        division[str(pid)] = {"val": "val", "validation": "val"}.get(split, split)
        no_target = set(self.mapping.get("no_target_values", ["None"]))
        rows = []
        for key, rec in data.items():
            pid = str(rec.get("post_id", key))
            anns = rec.get(fields["annotators"], []) or []
            votes = Counter(a.get(fields["annotator_label"]) for a in anns)
            top = votes.most_common()
            original = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
            labels = self.mapper.map_labels([original]) if original else ["UNKNOWN"]
            target_votes = Counter(t for a in anns for t in (a.get(fields["annotator_target"]) or [])
                                   if t not in no_target)
            majority_targets = [t for t, c in target_votes.items() if c >= (len(anns) // 2 + 1)]
            if majority_targets and original in ("hatespeech", "offensive"):
                labels.append("COMMUNITY_TARGETED")
            rows.append({
                "post_id": pid,
                "text": " ".join(rec.get(fields["tokens"], []) or []),
                "source_labels": join_labels(labels),
                "original_label": original or "no_majority",
                "group_id": None,  # HateXplain has no user/conversation ids
                "split": division.get(pid),
                "language": "en",
                "dataset_source": "hatexplain",
                "targets": ";".join(sorted(majority_targets)),
            })
        df = pd.DataFrame(rows)
        logger.info("HateXplain: %d posts loaded (%d with official split)",
                    len(df), int(df["split"].notna().sum()))
        return ensure_columns(df, CLASSIFICATION_COLUMNS)


# --------------------------------------------------------------------- HASOC
class HASOCAdapter(ClassificationDatasetAdapter):
    """HASOC TSV/CSV/XLSX -> classification table (column names from hasoc.yaml)."""

    def load(self) -> pd.DataFrame:
        files, cols = self.mapping["files"], self.mapping["columns"]
        path = _abs(files["data_file"])
        if not path.exists():
            raise DatasetError(f"{path} not found. See docs/DATASET_CREATION_GUIDE.md (HASOC).")
        opts = dict(self.mapping.get("read_options", {}) or {})
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            opts.pop("sep", None)
        raw = read_table(path, **opts)
        needed = [cols["id"], cols["text"], cols["coarse_label"]]
        missing = [c for c in needed if c not in raw.columns]
        if missing:
            raise DatasetError(f"HASOC file lacks columns {missing}; found {list(raw.columns)}. "
                               "Edit configs/dataset_mappings/hasoc.yaml")
        fine = cols.get("fine_label")
        rows = []
        for _, r in raw.iterrows():
            coarse = str(r[cols["coarse_label"]]).strip().upper()
            fine_val = str(r[fine]).strip().upper() if fine and fine in raw.columns and pd.notna(r[fine]) else None
            original = fine_val if fine_val and fine_val != "NONE" else coarse
            rows.append({
                "post_id": f"hasoc_{normalize_id(r[cols['id']])}",
                "text": r[cols["text"]],
                "source_labels": join_labels(self.mapper.map_labels([original])),
                "original_label": original,
                "group_id": None,
                "split": None,
                "language": self.mapping.get("language"),
                "dataset_source": "hasoc",
            })
        return ensure_columns(pd.DataFrame(rows), CLASSIFICATION_COLUMNS)


# --------------------------------------------------------------------- TwiBot-22
class TwiBot22Adapter(GraphDatasetAdapter):
    """TwiBot-22 (user.json, tweet_*.json, edge.csv) -> canonical graph tables.

    Streams tweets up to ``sampling.max_tweets`` (the first N encountered; this sample is
    NOT random and that bias must be reported). Bot/human labels are kept only as the
    user property ``source_bot_label`` and are NEVER converted into hate labels.
    Posts have no content labels: they must be produced by the classifier.
    """

    def load(self) -> CanonicalDataset:
        m = self.mapping
        files, tf, uf = m["files"], m["tweet_fields"], m["user_fields"]
        max_tweets = int(m["sampling"]["max_tweets"])
        tweet_files = sorted(glob.glob(str(_abs(files["tweets_glob"]))))
        if not tweet_files:
            raise DatasetError(f"No tweet files match {files['tweets_glob']}. "
                               "TwiBot-22 requires an access application (see DATASET_RESEARCH.md).")
        posts, tweet_author = [], {}
        for tfile in tweet_files:
            for rec in iter_json_array(tfile):
                tid = normalize_id(rec.get(tf["id"]))
                author = normalize_id(rec.get(tf["author_id"]))
                if not tid:
                    continue
                if author and not author.startswith("u"):
                    author = f"u{author}"
                tweet_author[tid] = author
                posts.append({"post_id": tid, "user_id": author, "text": rec.get(tf["text"]),
                              "timestamp": normalize_timestamp(rec.get(tf["created_at"])),
                              "language": rec.get(tf.get("lang", "lang")), "source_labels": None,
                              "conversation_id": None, "dataset_source": "twibot22",
                              "synthetic": False})
                if len(posts) >= max_tweets:
                    break
            if len(posts) >= max_tweets:
                break
        post_ids = set(tweet_author)
        post_ts = {p["post_id"]: p["timestamp"] for p in posts}

        rel_map = m["relation_map"]
        ec = m["edge_columns"]
        interactions, follows, user_ids = [], [], {a for a in tweet_author.values() if a}
        for chunk in pd.read_csv(_abs(files["edges"]), dtype=str, chunksize=1_000_000):
            chunk = chunk[chunk[ec["relation"]].isin(rel_map)]
            for src, rel, tgt in zip(chunk[ec["source"]], chunk[ec["relation"]], chunk[ec["target"]]):
                kind = rel_map[rel]
                if kind == "POSTED":
                    if tgt in post_ids and not tweet_author.get(tgt):
                        tweet_author[tgt] = src
                        user_ids.add(src)
                elif kind == "RETWEETED" and src in post_ids:
                    interactions.append(self._inter("RETWEETED", source_user_id=tweet_author.get(src),
                                                    target_post_id=tgt, ts=post_ts.get(src)))
                elif kind in ("REPLIED_TO", "QUOTED") and src in post_ids:
                    interactions.append(self._inter(kind, source_user_id=tweet_author.get(src),
                                                    source_post_id=src, target_post_id=tgt,
                                                    ts=post_ts.get(src)))
                elif kind == "MENTIONED" and src in post_ids:
                    interactions.append(self._inter("MENTIONED", source_user_id=tweet_author.get(src),
                                                    source_post_id=src, target_user_id=tgt,
                                                    ts=post_ts.get(src)))
                    user_ids.add(tgt)
                elif kind in ("FOLLOWS", "FOLLOWS_REVERSED") and (src in user_ids or tgt in user_ids):
                    a, b = (src, tgt) if kind == "FOLLOWS" else (tgt, src)
                    follows.append({"source_user_id": a, "target_user_id": b,
                                    "dataset_source": "twibot22", "synthetic": False})
        for p in posts:  # fill authors found via 'post' edges
            p["user_id"] = p["user_id"] or tweet_author.get(p["post_id"])

        bot_labels = {}
        if _abs(files["labels"]).exists():
            lab = pd.read_csv(_abs(files["labels"]), dtype=str)
            bot_labels = dict(zip(lab.iloc[:, 0], lab.iloc[:, 1]))
        usernames = {}
        if _abs(files["users"]).exists():
            for rec in iter_json_array(_abs(files["users"])):
                uid = normalize_id(rec.get(uf["id"]))
                if uid in user_ids:
                    usernames[uid] = rec.get(uf["username"])
        users = pd.DataFrame([{"user_id": u, "username": usernames.get(u), "dataset_source": "twibot22",
                               "synthetic": False, "source_bot_label": bot_labels.get(u),
                               "synthetic_gt_community": None} for u in sorted(user_ids)])
        ds = CanonicalDataset(
            name="twibot22", users=ensure_columns(users, USER_COLUMNS),
            posts=ensure_columns(pd.DataFrame(posts), POST_COLUMNS),
            interactions=ensure_columns(pd.DataFrame(interactions), INTERACTION_COLUMNS),
            follows=ensure_columns(pd.DataFrame(follows), FOLLOW_COLUMNS),
            capabilities=m.get("capabilities", {}),
            notes=[f"First {max_tweets} tweets encountered (non-random sample).",
                   "Posts carry NO hate labels; run the classifier. Bot labels are not hate labels."],
        )
        logger.info("TwiBot-22 loaded: %s", ds.summary())
        return ds

    @staticmethod
    def _inter(kind: str, ts: str | None, **ends: Any) -> dict[str, Any]:
        row = {c: None for c in INTERACTION_COLUMNS}
        row.update(ends)
        row.update(interaction_type=kind, timestamp=ts, dataset_source="twibot22", synthetic=False)
        return row


# --------------------------------------------------------------------- Hateful Users
class HatefulUsersAdapter(GraphDatasetAdapter):
    """Hateful Users on Twitter (Ribeiro et al., ICWSM 2018; Kaggle, CC0).

    * ``users_neighborhood_anon.csv``: one row per user; column ``hate`` is
      'hateful' / 'normal' (crowd-annotated, ~5k users) or 'other' (not annotated).
    * ``users.edges``: retweet graph, one edge per line, "u v" meaning u retweeted v.
    There are NO tweets, timestamps or post-level labels: the graph is User->User and the
    labels describe USERS (human annotation), not posts.
    """

    def load(self) -> CanonicalDataset:
        m = self.mapping
        files, cols = m["files"], m["columns"]
        upath, epath = _abs(files["users"]), _abs(files["edges"])
        for p in (upath, epath):
            if not p.exists():
                raise DatasetError(f"{p} not found. Download from Kaggle (see docs/ALTERNATIVE_DATASETS.md).")
        wanted = {cols["id"], cols["label"]}
        users_raw = pd.read_csv(upath, usecols=lambda c: c in wanted, dtype=str)
        label_map = {k.lower(): v for k, v in m["label_map"].items()}
        prefix = m.get("id_prefix", "hu_")
        users = pd.DataFrame({
            "user_id": prefix + users_raw[cols["id"]].astype(str).str.strip(),
            "username": None, "dataset_source": "hateful_users", "synthetic": False,
            "source_bot_label": None, "synthetic_gt_community": None,
            "user_hate_label": users_raw[cols["label"]].astype(str).str.strip().str.lower().map(label_map),
        })
        sep = self._sniff_separator(epath)
        edges = pd.read_csv(epath, sep=sep, header=None, names=["s", "t"], usecols=[0, 1],
                            dtype=str, comment="#", engine="python" if sep == r"\s+" else "c")
        edges = edges.dropna()
        edges = edges[edges["s"].str.strip().str.lstrip("-").str.isdigit()]  # drops a header line if any
        user_edges = pd.DataFrame({
            "source_user_id": prefix + edges["s"].str.strip(),
            "target_user_id": prefix + edges["t"].str.strip(),
            "edge_type": "RETWEETED_USER", "weight": 1.0,
            "dataset_source": "hateful_users", "synthetic": False,
        })
        ds = CanonicalDataset(
            name="hateful_users", users=ensure_columns(users, USER_COLUMNS),
            user_edges=ensure_columns(user_edges, USER_EDGE_COLUMNS),
            capabilities=m.get("capabilities", {}),
            notes=["User-level human labels (hateful/normal) for ~5k users; others unannotated.",
                   "Edges: u RETWEETED_USER v = u retweeted v. No posts, no timestamps."])
        logger.info("Hateful Users loaded: %s", ds.summary())
        return ds

    @staticmethod
    def _sniff_separator(path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    break
        if "," in line:
            return ","
        if "\t" in line:
            return "\t"
        return r"\s+"


# --------------------------------------------------------------------- Russian troll tweets
RT_RE = re.compile(r"^RT @(\w{1,30}):", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<![\w@])@(\w{1,30})")


class RussianTrollsAdapter(GraphDatasetAdapter):
    """NBC News Russian troll tweets (2016 US election period) -> canonical graph tables.

    Users are keyed by lower-cased screen name (``u_<screen_name>``) because mentions and
    retweet prefixes only give screen names. Interactions:
      * RETWEETED: tweet author -> retweeted_status_id. The ORIGINAL post is a stub
        (``is_stub``) whose author is INFERRED from the "RT @name:" prefix.
      * REPLIED_TO: tweet -> in_reply_to_status_id (stub target, author unknown).
      * MENTIONED: tweet -> each mentioned screen name (the RT author is excluded to avoid
        double-counting the retweet).
    Account-level fact: every author in this dataset is an IRA-linked account; stored as
    ``source_bot_label='ira_linked'``. That is NOT a hate label: run the classifier on text.
    """

    def load(self) -> CanonicalDataset:
        m = self.mapping
        files, c = m["files"], m["tweet_columns"]
        tpath = _abs(files["tweets"])
        if not tpath.exists():
            raise DatasetError(f"{tpath} not found. See docs/ALTERNATIVE_DATASETS.md.")
        raw = pd.read_csv(tpath, dtype=str, keep_default_na=True)
        with open(tpath, "rb") as fh:
            if fh.read(40).startswith(b"version https://git-lfs"):
                raise DatasetError("tweets.csv is a Git-LFS pointer, not data. Download it from the GitHub web page.")
        raw = raw.dropna(subset=[c["tweet_id"]])
        raw["_sn"] = raw[c["screen_name"]].fillna("").str.strip().str.lower()
        raw = raw[raw["_sn"] != ""]
        raw["_ts"] = raw[c["created_str"]].map(normalize_timestamp)
        if c.get("created_at_ms") in raw.columns:
            missing = raw["_ts"].isna()
            ms = pd.to_numeric(raw.loc[missing, c["created_at_ms"]], errors="coerce")
            raw.loc[missing, "_ts"] = pd.to_datetime(ms, unit="ms", utc=True).map(
                lambda t: None if pd.isna(t) else t.isoformat())
        s = m.get("sampling", {}) or {}
        ts = pd.to_datetime(raw["_ts"], utc=True, errors="coerce")
        if s.get("date_from"):
            raw = raw[ts >= pd.Timestamp(s["date_from"], tz="UTC")]
            ts = ts.loc[raw.index]
        if s.get("date_to"):
            raw = raw[ts < pd.Timestamp(s["date_to"], tz="UTC")]
        if s.get("max_tweets") and len(raw) > int(s["max_tweets"]):
            raw = raw.sample(n=int(s["max_tweets"]), random_state=int(s.get("seed", 42)))
        uid = lambda sn: f"u_{sn.lower()}"
        posts, inter, stubs = [], [], {}
        for row in raw.to_dict(orient="records"):
            tid, sn, text = normalize_id(row[c["tweet_id"]]), row["_sn"], row.get(c["text"])
            text = None if pd.isna(text) else str(text)
            posts.append({"post_id": tid, "user_id": uid(sn), "text": text, "timestamp": row["_ts"],
                          "language": None, "source_labels": None, "conversation_id": None,
                          "dataset_source": "russian_trolls", "synthetic": False, "is_stub": False})
            rt_author = None
            if text:
                mrt = RT_RE.match(text)
                rt_author = mrt.group(1).lower() if mrt else None
            rt_id = normalize_id(row.get(c["retweeted_status_id"]))
            if rt_id:
                inter.append(self._inter("RETWEETED", row["_ts"], source_user_id=uid(sn), target_post_id=rt_id))
                stubs.setdefault(rt_id, uid(rt_author) if rt_author else None)
            rep_id = normalize_id(row.get(c["in_reply_to_status_id"]))
            if rep_id:
                inter.append(self._inter("REPLIED_TO", row["_ts"], source_user_id=uid(sn),
                                         source_post_id=tid, target_post_id=rep_id))
                stubs.setdefault(rep_id, None)
            for ment in self._mentions(row.get(c["mentions"]), text):
                if ment == rt_author or ment == sn:
                    continue
                inter.append(self._inter("MENTIONED", row["_ts"], source_user_id=uid(sn),
                                         source_post_id=tid, target_user_id=uid(ment)))
        real_ids = {p["post_id"] for p in posts}
        for pid, author in stubs.items():
            if pid not in real_ids:
                posts.append({"post_id": pid, "user_id": author, "text": None, "timestamp": None,
                              "language": None, "source_labels": None, "conversation_id": None,
                              "dataset_source": "russian_trolls", "synthetic": False, "is_stub": True})
        ira = sorted(set(raw["_sn"]))
        users = pd.DataFrame({"user_id": [uid(x) for x in ira], "username": ira,
                              "dataset_source": "russian_trolls", "synthetic": False,
                              "source_bot_label": "ira_linked", "synthetic_gt_community": None,
                              "user_hate_label": None})
        ds = CanonicalDataset(
            name="russian_trolls", users=ensure_columns(users, USER_COLUMNS),
            posts=ensure_columns(pd.DataFrame(posts), POST_COLUMNS),
            interactions=ensure_columns(pd.DataFrame(inter), INTERACTION_COLUMNS),
            capabilities=m.get("capabilities", {}),
            notes=["All authors are IRA-linked accounts (dataset definition); others appear as placeholders.",
                   "Retweeted-post authors are INFERRED from the 'RT @name:' prefix.",
                   "No content labels: run the classifier (label_source=model)."])
        logger.info("Russian trolls loaded: %s", ds.summary())
        return ds

    @staticmethod
    def _mentions(field: Any, text: str | None) -> list[str]:
        names: list[str] = []
        if isinstance(field, str) and field.strip() not in ("", "[]"):
            try:
                parsed = json.loads(field.replace("'", '"'))
                names = [str(x) for x in parsed] if isinstance(parsed, list) else []
            except ValueError:
                names = MENTION_RE.findall(field) or [x.strip(" []'\"") for x in field.split(",")]
        elif text:
            names = MENTION_RE.findall(text)
        out: list[str] = []
        for n in names:
            n = n.strip().lstrip("@").lower()
            if n and n not in out:
                out.append(n)
        return out

    @staticmethod
    def _inter(kind: str, ts: str | None, **ends: Any) -> dict[str, Any]:
        row = {col: None for col in INTERACTION_COLUMNS}
        row.update(ends)
        row.update(interaction_type=kind, timestamp=ts, dataset_source="russian_trolls", synthetic=False)
        return row


GRAPH_ADAPTERS = {"synthetic": SyntheticAdapter, "twibot22": TwiBot22Adapter,
                  "hateful_users": HatefulUsersAdapter, "russian_trolls": RussianTrollsAdapter}
CLASSIFICATION_ADAPTERS = {"hatexplain": HateXplainAdapter, "hasoc": HASOCAdapter}


def get_graph_adapter(name: str) -> GraphDatasetAdapter:
    if name not in GRAPH_ADAPTERS:
        raise DatasetError(f"Unknown graph dataset '{name}'. Options: {list(GRAPH_ADAPTERS)}")
    return GRAPH_ADAPTERS[name](load_dataset_mapping(name))


def get_classification_adapter(name: str, taxonomy: dict[str, Any]) -> ClassificationDatasetAdapter:
    if name not in CLASSIFICATION_ADAPTERS:
        raise DatasetError(f"Unknown classifier dataset '{name}'. Options: {list(CLASSIFICATION_ADAPTERS)}")
    return CLASSIFICATION_ADAPTERS[name](load_dataset_mapping(name), LabelMapper.from_taxonomy(taxonomy, name))
