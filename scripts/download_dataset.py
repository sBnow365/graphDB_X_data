"""Download or register a dataset, then verify basic integrity.

Only HateXplain is downloaded automatically (public GitHub, MIT license).
HASOC and TwiBot-22 require you to obtain files yourself (registration / access
application); use --from-path to copy your files into data/raw/<dataset>/.

    python scripts/download_dataset.py --dataset hatexplain
    python scripts/download_dataset.py --dataset hasoc --from-path "C:\\Downloads\\hasoc_2021_en.tsv"
    python scripts/download_dataset.py --dataset twibot22 --from-path "D:\\TwiBot-22"
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import PROJECT_ROOT, get_config
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import save_json, utc_now_iso

HATEXPLAIN_FILES = {  # official repository: https://github.com/hate-alert/HateXplain
    "dataset.json": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
    "post_id_divisions.json": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json",
}
RUSSIAN_TROLLS_FILES = {  # NBC News data mirrored on GitHub (cite NBC News)
    "tweets.csv": "https://raw.githubusercontent.com/loretoparisi/russian_trolls_tweets_dataset/master/tweets.csv",
    "users.csv": "https://raw.githubusercontent.com/loretoparisi/russian_trolls_tweets_dataset/master/users.csv",
}
MANUAL = {
    "hateful_users": "Download the ZIP from https://www.kaggle.com/datasets/manoelribeiro/hateful-users-on-twitter "
                     "(free Kaggle login), unzip it, then re-run with --from-path <unzipped folder>. "
                     "Needed files: users_neighborhood_anon.csv and users.edges.",
    "hasoc": "Obtain HASOC data from https://hasocfire.github.io/hasoc/ (dataset page of the chosen year; "
             "follow its terms / registration). Then re-run with --from-path <file>.",
    "twibot22": "Apply for TwiBot-22 access as described at https://github.com/LuoUndergradXJTU/TwiBot-22 "
                "(institutional email, advisor, use case). Then re-run with --from-path <folder>.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)


def verify(files: list[Path]) -> list[dict]:
    out = []
    for f in files:
        rec = {"file": str(f.relative_to(PROJECT_ROOT)), "exists": f.exists()}
        if f.exists():
            rec.update(size_bytes=f.stat().st_size, sha256=sha256(f))
            if f.suffix == ".json" and f.stat().st_size < 500_000_000:
                try:
                    json.loads(f.read_text(encoding="utf-8"))
                    rec["json_valid"] = True
                except ValueError:
                    rec["json_valid"] = False
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["hatexplain", "hasoc", "twibot22", "hateful_users", "russian_trolls"])
    ap.add_argument("--from-path", help="local file or folder to copy into data/raw/<dataset>/")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "download")
    dest = cfg.path("raw") / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    if args.from_path:
        src = Path(args.from_path)
        if not src.exists():
            raise SystemExit(f"Not found: {src}")
        items = [src] if src.is_file() else [p for p in src.iterdir() if p.is_file()]
        for item in items:
            target = dest / item.name
            if not target.exists() or args.force:
                shutil.copy2(item, target)  # copy: your original stays untouched
            files.append(target)
    elif args.dataset in ("hatexplain", "russian_trolls"):
        sources = HATEXPLAIN_FILES if args.dataset == "hatexplain" else RUSSIAN_TROLLS_FILES
        for name, url in sources.items():
            target = dest / name
            if target.exists() and not args.force:
                print(f"exists, skipping: {target}")
            else:
                print(f"downloading {url}")
                download(url, target)
            files.append(target)
            with target.open("rb") as fh:
                if fh.read(40).startswith(b"version https://git-lfs"):
                    raise SystemExit(f"{target} is a Git-LFS pointer, not the data. Download it via the "
                                     "GitHub web page ('Download raw file') and use --from-path.")
    else:
        raise SystemExit(MANUAL[args.dataset])
    report = {"dataset": args.dataset, "checked_at": utc_now_iso(), "files": verify(files)}
    save_json(report, dest / "_download_manifest.json")
    for f in report["files"]:
        print(f)
    print(f"\nManifest: {dest / '_download_manifest.json'}")
    print("Next: python scripts/inspect_dataset.py --path <file>  then  preprocess_dataset.py")


if __name__ == "__main__":
    main()
