"""Evaluate a Transformer classifier on the TEST split of HateXplain or HASOC.

    python scripts/evaluate_classifier.py --dataset hatexplain --split test
Writes outputs/metrics/classifier_<dataset>_<split>.json and a confusion-matrix PNG.
Metrics are computed ONLY from predictions actually produced in this run.
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from graph_enhanced_lm.classification.evaluation import evaluate_multi_label, evaluate_single_label
from graph_enhanced_lm.classification.factory import build_classifier
from graph_enhanced_lm.classification.label_mapping import primary_label
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.data.loaders import read_table
from graph_enhanced_lm.data.schemas import split_labels
from graph_enhanced_lm.pipeline import processed_dir
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import capture_environment, save_json, utc_now_iso


def plot_confusion(cm, labels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hatexplain", choices=["hatexplain", "hasoc"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-rows", type=int, help="evaluate on a subset (quick CPU check)")
    args = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "eval_classifier")
    df = read_table(processed_dir(cfg, args.dataset) / "classification.csv")
    df = df[df["split"] == args.split]
    df = df[~df["source_labels"].fillna("UNKNOWN").str.contains("UNKNOWN")]  # no-majority posts excluded
    if args.max_rows:
        df = df.head(args.max_rows)
    if df.empty:
        raise SystemExit("No rows to evaluate. Run preprocess_dataset.py first.")
    ccfg = {**cfg.classifier, "backend": "transformer"}
    clf = build_classifier(ccfg, cfg.label_taxonomy, args.dataset)
    preds = clf.predict(df["post_id"].tolist(), df["text"].tolist())
    prio = cfg.label_taxonomy["primary_label_priority"]
    true_sets = [split_labels(v) for v in df["source_labels"]]
    # Compare on the label axis the model actually predicts (HATE/OFFENSIVE/NORMAL)
    axis = ["HATE", "OFFENSIVE", "NORMAL"]
    y_true = [primary_label([l for l in s if l in axis], prio) for s in true_sets]
    y_pred = [p.primary_label for p in preds]
    res = evaluate_single_label(y_true, y_pred, labels=axis)
    res["multi_label"] = evaluate_multi_label(true_sets, [p.labels for p in preds], axis)
    res.update({"dataset": args.dataset, "split": args.split, "model": clf.name,
                "model_version": getattr(clf, "version", None), "evaluated_at": utc_now_iso(),
                "environment": capture_environment(),
                "note": "COMMUNITY_TARGETED is not predicted by this 3-class model; it is evaluated separately if a target model is added."})
    out = cfg.path("metrics") / f"classifier_{args.dataset}_{args.split}.json"
    save_json(res, out)
    plot_confusion(res["confusion_matrix"], axis, cfg.path("figures") / f"confusion_{args.dataset}_{args.split}.png")
    print(res["report_text"])
    print(f"macro-F1 = {res['macro_f1']:.4f}   saved {out}")
    pd.DataFrame([{"post_id": p.post_id, "true": t, "pred": p.primary_label, "conf": p.confidence}
                  for p, t in zip(preds, y_true)]).to_csv(
        cfg.path("metrics") / f"classifier_{args.dataset}_{args.split}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
