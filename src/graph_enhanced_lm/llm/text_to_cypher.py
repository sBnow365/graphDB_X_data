"""Natural language -> validated, read-only Cypher -> structured rows.

Pipeline: schema-aware prompt -> LLM -> extract Cypher -> static validation ->
EXPLAIN query-type check ('r' only) -> READ transaction with timeout + row cap.
On failure the error is fed back once for a corrected query; if that also fails a
controlled error is returned. Results are NEVER fabricated.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graph_enhanced_lm.graph.cypher_queries import TEMPLATES
from graph_enhanced_lm.graph.schema import schema_as_text
from graph_enhanced_lm.llm import prompts
from graph_enhanced_lm.llm.base import LLMClient
from graph_enhanced_lm.llm.qwen_client import strip_thinking
from graph_enhanced_lm.llm.query_validation import CypherValidator, extract_cypher
from graph_enhanced_lm.utils.errors import LLMError

logger = logging.getLogger(__name__)

FEW_SHOT = ["users_posting_label_in_range", "community_label_stats", "repeated_amplifiers"]


@dataclass
class TextToCypherResult:
    question: str
    cypher: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    executed: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    generation_latency: float = 0.0
    query_type: str | None = None

    @property
    def ok(self) -> bool:
        return self.valid and (self.executed or not self.errors)


class TextToCypher:
    def __init__(self, llm: LLMClient, validator: CypherValidator, client: Any | None = None,
                 max_retries: int = 1, query_timeout: float = 20):
        self.llm = llm
        self.validator = validator
        self.client = client          # None -> generate + validate only (dry run)
        self.max_retries = max_retries
        self.query_timeout = query_timeout

    def build_messages(self, question: str, params: dict[str, Any]) -> list[dict[str, str]]:
        examples = "\n\n".join(f"// {TEMPLATES[n].description}\n{TEMPLATES[n].cypher}" for n in FEW_SHOT)
        system = prompts.TEXT_TO_CYPHER_SYSTEM.format(
            schema=schema_as_text(), param_names=", ".join(f"${k}" for k in params))
        user = prompts.TEXT_TO_CYPHER_USER.format(
            question=question, examples=examples,
            param_values="\n".join(f"  ${k} = {v!r}" for k, v in params.items()))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def run(self, question: str, params: dict[str, Any] | None = None) -> TextToCypherResult:
        params = dict(params or {})
        params.setdefault("limit", self.validator.max_rows)
        result = TextToCypherResult(question=question, params=params)
        messages = self.build_messages(question, params)
        last_error = ""
        for attempt in range(self.max_retries + 1):
            result.attempts = attempt + 1
            if attempt:
                messages = messages + [{"role": "assistant", "content": f"```cypher\n{result.cypher}\n```"},
                                       {"role": "user", "content": prompts.RETRY_USER.format(
                                           error=last_error, cypher=result.cypher)}]
            t0 = time.perf_counter()
            try:
                text = strip_thinking(self.llm.chat(messages).text)
            except LLMError as exc:
                result.errors.append(f"LLM failure: {exc}")
                return result
            result.generation_latency += time.perf_counter() - t0
            result.cypher = extract_cypher(text)
            check = self.validator.validate(result.cypher, params)
            result.warnings = check.warnings
            if not check.ok:
                last_error = "; ".join(check.errors)
                result.errors.append(f"attempt {attempt + 1}: {last_error}")
                continue
            result.cypher = check.cypher
            result.valid = True
            if self.client is None:
                return result
            try:
                qtype = self.client.explain_query_type(result.cypher, params)
                result.query_type = qtype
                if qtype != "r":
                    result.valid = False
                    last_error = f"Neo4j EXPLAIN reports query type '{qtype}', only 'r' is allowed"
                    result.errors.append(f"attempt {attempt + 1}: {last_error}")
                    continue
                result.rows = self.client.run_read(result.cypher, params, timeout=self.query_timeout,
                                                   max_rows=self.validator.max_rows)
                result.executed = True
                result.errors = []
                return result
            except Exception as exc:  # Cypher syntax/semantic errors from Neo4j
                result.valid = False
                last_error = f"Neo4j error: {str(exc)[:400]}"
                result.errors.append(f"attempt {attempt + 1}: {last_error}")
        logger.info("Text-to-Cypher failed after %d attempt(s)", result.attempts)
        return result
