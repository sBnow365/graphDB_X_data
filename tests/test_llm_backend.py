import pytest

from graph_enhanced_lm.llm.backends import openai_compatible as oc
from graph_enhanced_lm.utils.errors import LLMError


class _Resp:
    def __init__(self, payload):
        self.status_code, self._p, self.text = 200, payload, str(payload)

    def json(self):
        return self._p


def _patch(monkeypatch, payload, seen):
    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(json)
        return _Resp(payload)
    monkeypatch.setattr(oc.requests, "post", fake_post)


def test_sends_think_false_and_extra_body(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"choices": [{"message": {"content": "```cypher\nMATCH (n) RETURN n\n```"}}]}, seen)
    c = oc.OpenAICompatibleClient("http://x/v1", "qwen3:8b", extra_body={"reasoning_effort": "none"})
    assert "MATCH" in c.chat([{"role": "user", "content": "q"}]).text
    assert seen["think"] is False and seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["reasoning_effort"] == "none"


def test_empty_content_uses_reasoning_code_block(monkeypatch):
    _patch(monkeypatch, {"choices": [{"finish_reason": "stop", "message": {
        "content": "", "reasoning": "thinking... ```cypher\nMATCH (u:User) RETURN count(u)\n```"}}]}, {})
    out = oc.OpenAICompatibleClient("http://x/v1", "m").chat([{"role": "user", "content": "q"}]).text
    assert out.startswith("```cypher")


def test_empty_content_clear_error(monkeypatch):
    _patch(monkeypatch, {"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning": "long"}}]}, {})
    with pytest.raises(LLMError, match="finish_reason=length"):
        oc.OpenAICompatibleClient("http://x/v1", "m").chat([{"role": "user", "content": "q"}])
