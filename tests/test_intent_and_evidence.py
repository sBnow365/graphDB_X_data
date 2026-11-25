import yaml

from graph_enhanced_lm.config import PROJECT_ROOT
from graph_enhanced_lm.retrieval.evidence_formatter import format_rows, merge_evidence
from graph_enhanced_lm.retrieval.intent_parser import parse_question


def test_ten_days_before_election(cfg):
    it = parse_question("Which users posted hate-labelled content during the ten days before the election?",
                        cfg.temporal["election_date"], cfg.temporal["timezone"])
    assert it.labels == ["HATE"] and it.intent == "users_posting_label_in_range"
    assert it.date_range.days == 10 and it.date_range.name == "pre_10"
    it2 = parse_question("hate posts 10 days before the election by which users", "2025-03-15", "Asia/Kolkata")
    assert it2.date_range.days == 10


def test_routing_benchmark(cfg):
    items = yaml.safe_load((PROJECT_ROOT / "benchmarks/graph_rag_benchmark.yaml").read_text(encoding="utf-8"))["items"]
    for it in items:
        intent = parse_question(it["question"], cfg.temporal["election_date"], cfg.temporal["timezone"])
        assert intent.intent == it["expected_intent"], (it["id"], intent.intent)
        assert set(it.get("expected_labels", [])) <= set(intent.labels), (it["id"], intent.labels)


def test_explicit_dates_and_community():
    it = parse_question("Explain the evidence for community 7 between 2025-03-01 and 2025-03-05", None)
    assert it.community_id == 7 and it.intent == "community_evidence"
    assert it.date_range.days == 5


def test_election_without_date_is_noted():
    it = parse_question("hate before the election", None)
    assert it.date_range is None and "No election date configured" in it.notes


def test_evidence_formatting_truncates_and_ids():
    rows = [{"post_id": f"p{i}", "text": "x" * 500, "confidence": 0.912345} for i in range(5)]
    ev = format_rows(rows, max_items=3, max_text_chars=20)
    assert ev.ids() == ["E1", "E2", "E3"] and ev.truncated
    assert "..." in ev.items[0].content and "0.9123" in ev.items[0].content
    assert ev.items[0].record_ids == {"post_id": "p0"}


def test_merge_renumbers():
    a, b = format_rows([{"user_id": "u1"}]), format_rows([{"user_id": "u2"}])
    assert merge_evidence(a, b).ids() == ["E1", "E2"]
