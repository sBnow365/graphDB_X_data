# """Any server exposing POST /v1/chat/completions (Ollama, llama.cpp, vLLM, Colab tunnel).

# Only the ``requests`` package is needed; no vendor SDK and no paid API.
# """
# from __future__ import annotations

# import time
# from typing import Any

# import requests

# from graph_enhanced_lm.llm.base import LLMClient, LLMResponse
# from graph_enhanced_lm.utils.errors import LLMError


# class OpenAICompatibleClient(LLMClient):
#     def __init__(self, base_url: str, model: str, api_key: str | None = None,
#                  timeout: float = 120, temperature: float = 0.0, max_tokens: int = 768,
#                  disable_thinking: bool = True):
#         if not base_url:
#             raise LLMError("LLM_BASE_URL is empty. Set it in .env (see docs/QWEN_SETUP.md).")
#         self.base_url = base_url.rstrip("/")
#         self.model = model
#         self.api_key = api_key
#         self.timeout = timeout
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.disable_thinking = disable_thinking

#     def chat(self, messages, temperature=None, max_tokens=None):
#         msgs = [dict(m) for m in messages]
#         if self.disable_thinking and msgs and msgs[-1]["role"] == "user":
#             msgs[-1]["content"] += " /no_think"   # Qwen3 soft switch (harmless for other models)
#         payload: dict[str, Any] = {
#             "model": self.model, "messages": msgs,
#             "temperature": self.temperature if temperature is None else temperature,
#             "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
#         }
#         if self.disable_thinking:   # honoured by vLLM / llama.cpp; ignored by others
#             payload["chat_template_kwargs"] = {"enable_thinking": False}
#         headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
#         if self.api_key:
#             headers["Authorization"] = f"Bearer {self.api_key}"
#         t0 = time.perf_counter()
#         try:
#             resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
#                                  headers=headers, timeout=self.timeout)
#         except requests.RequestException as exc:
#             raise LLMError(f"Cannot reach LLM endpoint {self.base_url}: {exc}") from exc
#         if resp.status_code != 200:
#             raise LLMError(f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")
#         try:
#             data = resp.json()
#             text = data["choices"][0]["message"]["content"] or ""
#         except (ValueError, KeyError, IndexError) as exc:
#             raise LLMError(f"Unexpected LLM response format: {resp.text[:300]}") from exc
#         return LLMResponse(text=text, model=data.get("model", self.model),
#                            latency_seconds=round(time.perf_counter() - t0, 3), raw=data)

#     def health(self) -> list[str]:
#         """List model ids served by the endpoint (GET /v1/models)."""
#         headers = {"ngrok-skip-browser-warning": "1"}
#         if self.api_key:
#             headers["Authorization"] = f"Bearer {self.api_key}"
#         try:
#             r = requests.get(f"{self.base_url}/models", headers=headers, timeout=15)
#             r.raise_for_status()
#             return [m.get("id") for m in r.json().get("data", [])]
#         except (requests.RequestException, ValueError) as exc:
#             raise LLMError(f"LLM health check failed at {self.base_url}: {exc}") from exc


"""Any server exposing POST /v1/chat/completions (Ollama, llama.cpp, vLLM, Colab tunnel).

Only the ``requests`` package is needed; no vendor SDK and no paid API.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from graph_enhanced_lm.llm.base import LLMClient, LLMResponse
from graph_enhanced_lm.utils.errors import LLMError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 120, temperature: float = 0.0, max_tokens: int = 768,
                 disable_thinking: bool = True, extra_body: dict[str, Any] | None = None):
        if not base_url:
            raise LLMError("LLM_BASE_URL is empty. Set it in .env (see docs/QWEN_SETUP.md).")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self.extra_body = dict(extra_body or {})

    def chat(self, messages, temperature=None, max_tokens=None):
        msgs = [dict(m) for m in messages]
        if self.disable_thinking and msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] += " /no_think"   # Qwen3 soft switch (harmless for other models)
        payload: dict[str, Any] = {
            "model": self.model, "messages": msgs,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}   # vLLM / llama.cpp
            payload["think"] = False                                       # Ollama
        payload.update(self.extra_body)                                    # anything server-specific
        headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"Cannot reach LLM endpoint {self.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            text = message.get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected LLM response format: {resp.text[:300]}") from exc
        if not text.strip():
            # Thinking models may put everything in a separate reasoning field and run out of
            # tokens before writing the answer. Use the reasoning only if it contains the answer.
            reasoning = message.get("reasoning") or message.get("reasoning_content") or message.get("thinking") or ""
            if "```" in reasoning:
                text = reasoning[reasoning.index("```"):]
            else:
                raise LLMError(
                    f"The model returned an empty answer (finish_reason={choice.get('finish_reason')}, "
                    f"reasoning chars={len(reasoning)}). Thinking mode probably used the whole token budget: "
                    "raise llm.yaml max_tokens or make sure thinking is disabled.")
        return LLMResponse(text=text, model=data.get("model", self.model),
                           latency_seconds=round(time.perf_counter() - t0, 3), raw=data)

    def health(self) -> list[str]:
        """List model ids served by the endpoint (GET /v1/models)."""
        headers = {"ngrok-skip-browser-warning": "1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.get(f"{self.base_url}/models", headers=headers, timeout=15)
            r.raise_for_status()
            return [m.get("id") for m in r.json().get("data", [])]
        except (requests.RequestException, ValueError) as exc:
            raise LLMError(f"LLM health check failed at {self.base_url}: {exc}") from exc
