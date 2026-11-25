"""Factory for the configured LLM client + Qwen3-specific output cleaning."""
from __future__ import annotations

import re
from typing import Any

from graph_enhanced_lm.llm.base import LLMClient
from graph_enhanced_lm.utils.errors import LLMError

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` blocks (and a dangling opening tag)."""
    text = THINK_RE.sub("", text or "")
    if "<think>" in text.lower():          # unterminated block: keep what follows it
        text = text.split("</think>")[-1]
    return text.strip()


def build_llm_client(llm_cfg: dict[str, Any]) -> LLMClient:
    """Create the client named by ``llm_cfg['backend']`` (env overrides already applied)."""
    backend = llm_cfg.get("backend", "openai_compatible")
    if backend == "openai_compatible":
        from graph_enhanced_lm.llm.backends.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url=llm_cfg.get("base_url", ""), model=llm_cfg.get("model", ""),
            api_key=llm_cfg.get("api_key"), timeout=float(llm_cfg.get("timeout_seconds", 120)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 768)),
            disable_thinking=bool(llm_cfg.get("disable_thinking", True)),
            extra_body=llm_cfg.get("extra_body") or {})
    if backend == "hf_transformers":
        from graph_enhanced_lm.llm.backends.hf_transformers import HFTransformersClient

        return HFTransformersClient(
            model_id=llm_cfg.get("hf_model_id", "Qwen/Qwen3-8B"),
            load_in_4bit=bool(llm_cfg.get("load_in_4bit", True)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 768)),
            disable_thinking=bool(llm_cfg.get("disable_thinking", True)))
    if backend == "mock":
        from graph_enhanced_lm.llm.backends.mock import MockLLMClient

        return MockLLMClient()
    raise LLMError(f"Unknown LLM backend '{backend}'")
