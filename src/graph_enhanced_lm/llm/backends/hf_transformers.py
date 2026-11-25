"""Run the model in-process with Hugging Face transformers (Colab / Linux GPU).

Qwen3's chat template accepts ``enable_thinking=False`` for non-thinking mode.
"""
from __future__ import annotations

import time

from graph_enhanced_lm.llm.base import LLMClient, LLMResponse
from graph_enhanced_lm.utils.errors import LLMError


class HFTransformersClient(LLMClient):
    def __init__(self, model_id: str, load_in_4bit: bool = True, temperature: float = 0.0,
                 max_tokens: int = 768, disable_thinking: bool = True):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise LLMError("Install torch and requirements-llm-local.txt for the hf_transformers backend") from exc
        self.torch = torch
        self.model = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def chat(self, messages, temperature=None, max_tokens=None):
        t0 = time.perf_counter()
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=not self.disable_thinking)
        except TypeError:  # templates without enable_thinking
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.lm.device)
        temp = self.temperature if temperature is None else temperature
        gen_kwargs = {"max_new_tokens": max_tokens or self.max_tokens}
        gen_kwargs.update({"do_sample": True, "temperature": temp} if temp > 0 else {"do_sample": False})
        with self.torch.no_grad():
            out = self.lm.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return LLMResponse(text=text, model=self.model,
                           latency_seconds=round(time.perf_counter() - t0, 3))
