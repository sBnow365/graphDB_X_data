"""Graph RAG orchestrator.

question -> intent parsing -> template retrieval (or validated Text-to-Cypher)
         -> evidence formatting -> grounded LLM answer (with citation checks).
Works WITHOUT an LLM: you still get the retrieved evidence.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graph_enhanced_lm.llm.answer_generation import AnswerGenerator, GroundedAnswer
from graph_enhanced_lm.llm.text_to_cypher import TextToCypher
from graph_enhanced_lm.retrieval.evidence_formatter import Evidence, format_rows
from graph_enhanced_lm.retrieval.graph_retriever import GraphRetriever, RetrievalResult
from graph_enhanced_lm.retrieval.intent_parser import Intent, parse_question

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    question: str
    intent: Intent
    method: str
    cypher: str | None
    params: dict[str, Any]
    evidence: Evidence
    answer: GroundedAnswer | None
    errors: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "intent": self.intent.intent, "labels": self.intent.labels,
            "date_range": self.intent.date_range.as_params() if self.intent.date_range else None,
            "method": self.method, "cypher": self.cypher, "params": self.params,
            "evidence": self.evidence.to_records(), "evidence_notes": self.evidence.notes,
            "answer": self.answer.text if self.answer else None,
            "insufficient": self.answer.insufficient if self.answer else None,
            "unknown_citations": self.answer.unknown_citations if self.answer else [],
            "errors": self.errors, "latency_seconds": self.latency_seconds,
        }


class GraphRAG:
    def __init__(self, retriever: GraphRetriever, answerer: AnswerGenerator,
                 text_to_cypher: TextToCypher | None, election_date: str | None,
                 timezone: str = "UTC", max_evidence: int = 40, max_text_chars: int = 160,
                 synthetic: bool = False):
        self.retriever = retriever
        self.answerer = answerer
        self.t2c = text_to_cypher
        self.election_date = election_date
        self.timezone = timezone
        self.max_evidence = max_evidence
        self.max_text_chars = max_text_chars
        self.synthetic = synthetic

    def ask(self, question: str, generate_answer: bool = True) -> RAGResult:
        t0 = time.perf_counter()
        intent = parse_question(question, self.election_date, self.timezone)
        errors: list[str] = []
        retrieval: RetrievalResult | None = None
        try:
            retrieval = self.retriever.retrieve(intent)
        except Exception as exc:
            errors.append(f"Template retrieval failed: {exc}")
        if retrieval is None and self.t2c is not None:
            params = intent.params()
            params.setdefault("community_property", self.retriever.prop)
            t2c = self.t2c.run(question, params)
            if t2c.executed:
                retrieval = RetrievalResult(method="text_to_cypher", rows=t2c.rows,
                                            cypher=t2c.cypher, params=t2c.params)
            else:
                errors += t2c.errors or ["Text-to-Cypher produced no executable query"]
                retrieval = RetrievalResult(method="text_to_cypher_failed", cypher=t2c.cypher,
                                            params=t2c.params)
        if retrieval is None:
            retrieval = RetrievalResult(method="none", notes=[
                "Question not recognised and no LLM configured for Text-to-Cypher."])
        evidence = format_rows(retrieval.rows, "computed" if retrieval.computed else "record",
                               self.max_evidence, self.max_text_chars)
        evidence.source_query = retrieval.cypher
        evidence.notes += retrieval.notes + intent.notes
        if self.synthetic:
            evidence.notes.append("All records are SYNTHETIC test data, not real-world evidence.")
        if retrieval.method == "text_to_cypher_failed":
            reason = errors[-1] if errors else "no valid query was produced"
            answer = GroundedAnswer(
                text=("QUERY_FAILED: the language model could not produce a valid read-only Cypher query, "
                      f"so the graph was not queried. Last error: {reason}"),
                insufficient=True) if generate_answer else None
        else:
            answer = self.answerer.answer(question, evidence) if generate_answer else None
        return RAGResult(question, intent, retrieval.method, retrieval.cypher, retrieval.params,
                         evidence, answer, errors, round(time.perf_counter() - t0, 3))


def build_graph_rag(client: Any, llm: Any | None, cfg: Any) -> GraphRAG:
    """Wire everything from configuration (used by scripts/ask.py and run_demo.py)."""
    from graph_enhanced_lm.llm.query_validation import CypherValidator

    gds_cfg, llm_cfg = cfg.gds, cfg.llm
    retriever = GraphRetriever(client, gds_cfg["louvain"]["write_property"], cfg.temporal,
                               gds_cfg["community_analysis"]["priority_weights"],
                               int(gds_cfg["community_analysis"]["min_users"]))
    t2c = None
    if llm is not None:
        t2c_cfg = llm_cfg["text_to_cypher"]
        validator = CypherValidator(int(t2c_cfg["max_rows"]), t2c_cfg.get("allowed_procedures"))
        t2c = TextToCypher(llm, validator, client, int(t2c_cfg["max_retries"]),
                           float(t2c_cfg["query_timeout_seconds"]))
    return GraphRAG(retriever, AnswerGenerator(llm), t2c, cfg.temporal.get("election_date"),
                    cfg.temporal.get("timezone", "UTC"), int(llm_cfg["answer"]["max_evidence_items"]),
                    int(cfg.project["privacy"]["evidence_text_max_chars"]),
                    synthetic=cfg.dataset["active_graph_dataset"] == "synthetic")
