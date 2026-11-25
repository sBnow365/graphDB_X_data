"""Deterministic fake LLM for tests and offline demos. It does not use any model."""
from __future__ import annotations

from typing import Callable

from graph_enhanced_lm.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Returns the first response whose trigger substring appears in the last user message."""

    model = "mock-llm"

    def __init__(self, responses: dict[str, str] | None = None,
                 default: str | Callable[[str], str] = "INSUFFICIENT_EVIDENCE"):
        self.responses = responses or {}
        self.default = default
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for trigger, reply in self.responses.items():
            if trigger.lower() in last.lower():
                return LLMResponse(text=reply, model=self.model, latency_seconds=0.0)
        text = self.default(last) if callable(self.default) else self.default
        return LLMResponse(text=text, model=self.model, latency_seconds=0.0)
