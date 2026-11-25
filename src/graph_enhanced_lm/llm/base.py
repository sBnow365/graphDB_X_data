"""LLM client interface. Swap Qwen3-8B for another model by changing config only."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_seconds: float
    raw: dict | None = None


class LLMClient(ABC):
    model: str = "unknown"

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], temperature: float | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        """Send OpenAI-style messages [{'role': 'system'|'user'|'assistant', 'content': ...}]."""
