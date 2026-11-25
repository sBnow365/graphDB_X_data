"""Evidence-grounded answer generation with citation checking."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from graph_enhanced_lm.llm import prompts
from graph_enhanced_lm.llm.base import LLMClient
from graph_enhanced_lm.llm.qwen_client import strip_thinking
from graph_enhanced_lm.retrieval.evidence_formatter import Evidence
from graph_enhanced_lm.utils.errors import LLMError

CITE_RE = re.compile(r"\[E(\d+)\]")
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass
class GroundedAnswer:
    text: str
    insufficient: bool
    cited_ids: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    uncited: bool = False
    model: str = ""
    latency_seconds: float = 0.0
    llm_called: bool = False


class AnswerGenerator:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def answer(self, question: str, evidence: Evidence) -> GroundedAnswer:
        if evidence.is_empty():
            return GroundedAnswer(
                text=f"{INSUFFICIENT}: the graph query returned no records for this question.",
                insufficient=True)
        if self.llm is None:
            return GroundedAnswer(text="(No LLM configured.) Evidence only:\n" + evidence.to_prompt_text(),
                                  insufficient=False)
        messages = [{"role": "system", "content": prompts.ANSWER_SYSTEM},
                    {"role": "user", "content": prompts.ANSWER_USER.format(
                        question=question, evidence=evidence.to_prompt_text())}]
        try:
            resp = self.llm.chat(messages)
        except LLMError as exc:
            return GroundedAnswer(text=f"LLM unavailable: {exc}\n\nEvidence:\n{evidence.to_prompt_text()}",
                                  insufficient=False)
        text = strip_thinking(resp.text)
        cited = [f"E{n}" for n in CITE_RE.findall(text)]
        valid_ids = set(evidence.ids())
        unknown = sorted({c for c in cited if c not in valid_ids})
        insufficient = text.strip().upper().startswith(INSUFFICIENT)
        return GroundedAnswer(text=text, insufficient=insufficient, cited_ids=sorted(set(cited)),
                              unknown_citations=unknown, uncited=not cited and not insufficient,
                              model=resp.model, latency_seconds=resp.latency_seconds, llm_called=True)
