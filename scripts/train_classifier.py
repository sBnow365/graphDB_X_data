"""OPTIONAL: fine-tune a BERT-family classifier on HateXplain/HASOC (GPU recommended).

The baseline does NOT need this. Use it as an extra experiment and compare with the
pretrained checkpoint using evaluate_classifier.py.

    python scripts/train_classifier.py --dataset hatexplain
"""
import argparse

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import PROJECT_ROOT, get_config
from graph_enhanced_lm.data.loaders import read_table
from graph_enhanced_lm.data.schemas import split_labels
from graph_enhanced_lm.pipeline import processed_dir
from graph_enhanced_lm.utils.logging_config import setup_logging
from graph_enhanced_lm.utils.reproducibility import set_seed

AXIS = ["NORMAL", "OFFENSIVE", "HATE"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hatexplain", choices=["hatexplain", "hasoc"])
    args = ap.parse_args()
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                                  TrainingArguments)
    except ImportError:
        raise SystemExit("Install torch and requirements-classifier.txt first.")
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "train_classifier")
    ft = cfg.classifier["fine_tuning"]
    set_seed(int(cfg.project["random_seed"]))
    df = read_table(processed_dir(cfg, args.dataset) / "classification.csv")
    df["y"] = df["source_labels"].map(lambda v: next((AXIS.index(l) for l in ("HATE", "OFFENSIVE", "NORMAL")
                                                      if l in split_labels(v)), None))
    df = df.dropna(subset=["y", "text"])
    tok = AutoTokenizer.from_pretrained(ft["base_model"])

    class DS(torch.utils.data.Dataset):
        def __init__(self, part):
            self.enc = tok(part["text"].tolist(), truncation=True, padding=True,
                           max_length=int(cfg.classifier["transformer"]["max_length"]))
            self.y = part["y"].astype(int).tolist()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        ft["base_model"], num_labels=3, id2label=dict(enumerate([a.lower() for a in AXIS])),
        label2id={a.lower(): i for i, a in enumerate(AXIS)})
    out = PROJECT_ROOT / ft["output_dir"]
    targs = TrainingArguments(output_dir=str(out), num_train_epochs=float(ft["epochs"]),
                              learning_rate=float(ft["learning_rate"]),
                              per_device_train_batch_size=int(ft["batch_size"]),
                              eval_strategy="epoch", save_strategy="epoch",
                              load_best_model_at_end=True, seed=int(cfg.project["random_seed"]))
    trainer = Trainer(model=model, args=targs, train_dataset=DS(df[df["split"] == "train"]),
                      eval_dataset=DS(df[df["split"] == "val"]))
    trainer.train()
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    print(f"Saved to {out / 'final'}. To use it set classifier.yaml transformer.model_name to that path "
          "and add a mapping with keys normal/offensive/hate.")


if __name__ == "__main__":
    main()
