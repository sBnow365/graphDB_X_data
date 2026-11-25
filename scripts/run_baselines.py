"""Collect results of the three compared configurations into reports/experiment_results.md.

  A. Text classification only        <- outputs/metrics/classifier_*.json
  B. Graph analysis without the LLM  <- graph_algorithms_latest.json, community_ranking.csv
  C. Graph RAG / Text-to-Cypher      <- text_to_cypher_eval.json, graph_rag_eval.json

Only numbers present in those files are reported; missing experiments are shown as
"NOT RUN". Nothing is estimated or invented.

    python scripts/run_baselines.py
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import PROJECT_ROOT, get_config
from graph_enhanced_lm.utils.reproducibility import utc_now_iso


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    cfg = get_config()
    m = cfg.path("metrics")
    lines = ["# Experiment results (auto-generated)", "", f"Generated: {utc_now_iso()}",
             f"Graph dataset: **{cfg.dataset['active_graph_dataset']}**"
             + (" (SYNTHETIC - not real-world evidence)" if cfg.dataset["active_graph_dataset"] == "synthetic" else ""),
             "", "## A. Text classification only", ""]
    found = sorted(m.glob("classifier_*_*.json"))
    found = [f for f in found if not f.name.endswith("_predictions.json")]
    if not found:
        lines.append("NOT RUN - run scripts/evaluate_classifier.py")
    for f in found:
        r = load(f)
        lines.append(f"- `{f.name}`: model `{r.get('model')}`, n={r.get('n')}, accuracy={r.get('accuracy'):.4f}, "
                     f"macro-F1={r.get('macro_f1'):.4f}")
    lines += ["", "## B. Graph analysis without LLM", ""]
    g = load(m / "graph_algorithms_latest.json")
    if g:
        lv = g.get("louvain", {})
        lines.append(f"- Louvain: {lv.get('communityCount')} communities, modularity={lv.get('modularity')}")
        if g.get("synthetic_ground_truth_recovery"):
            lines.append(f"- Synthetic ground-truth recovery: {g['synthetic_ground_truth_recovery']}")
        lines.append(f"- Community sizes: {g.get('size_distribution', {}).get('sizes')}")
    else:
        lines.append("NOT RUN - run scripts/run_graph_algorithms.py")
    rank = m / "community_ranking.md"
    lines.append(f"- Community ranking: {'see outputs/metrics/community_ranking.md' if rank.exists() else 'NOT RUN'}")
    lines += ["", "## C. Graph RAG / Text-to-Cypher", ""]
    t = load(m / "text_to_cypher_eval.json")
    if t:
        lines.append(f"- Safety suite accuracy: {t['safety_suite']['accuracy']}")
        b = t.get("benchmark")
        if b:
            lines.append(f"- Benchmark ({b['n']} items, model {b.get('llm_model')}, "
                         f"{b.get('unverified_items')} unverified): valid={b['syntax_valid_rate']}, "
                         f"executed={b['execution_success_rate']}, exact match={b['exact_result_match_rate']}, "
                         f"mean F1={b['mean_result_f1']}, latency={b['mean_latency_seconds']}s")
        else:
            lines.append("- Benchmark: NOT RUN (only safety suite)")
    else:
        lines.append("NOT RUN - run scripts/evaluate_text_to_cypher.py")
    r = load(m / "graph_rag_eval.json")
    if r:
        lines.append(f"- Intent routing accuracy: {r['routing_accuracy']}")
        if r.get("automatic_grounding"):
            lines.append(f"- Automatic grounding stats: {r['automatic_grounding']}")
    else:
        lines.append("- Graph RAG: NOT RUN - run scripts/evaluate_graph_rag.py")
    lines += ["", "Do not claim graph enhancement improves results unless these numbers show it."]
    out = PROJECT_ROOT / "reports" / "experiment_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
