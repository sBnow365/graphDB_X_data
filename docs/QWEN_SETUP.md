# Qwen3-8B Setup

## Facts checked (30 Oct 2026)
* Qwen3-8B: ~8.2B parameters, Apache-2.0 licence, 32,768-token native context (131,072 with
  YaRN), switchable thinking / non-thinking mode (`enable_thinking`), served by vLLM, SGLang,
  llama.cpp, Ollama, LM Studio. Source: Hugging Face model card `Qwen/Qwen3-8B` and community
  model listings. **Re-check the model card before your final report.**
* We do **not** train Qwen. It is used through prompting for Text-to-Cypher and answer writing.
* Thinking mode is disabled for these tasks (faster, cleaner output). The code also appends the
  documented `/no_think` soft switch and strips any `<think>…</think>` text.

## How the code talks to the model
`src/graph_enhanced_lm/llm/backends/openai_compatible.py` sends
`POST {LLM_BASE_URL}/chat/completions`. Any server that implements this works. Change only `.env`:
```
LLM_BACKEND=openai_compatible
LLM_BASE_URL=...      # see options below, must end with /v1
LLM_MODEL=...         # model name as the server knows it
```

## Option A (recommended for you): Google Colab GPU + tunnel
Memory: 8B weights need ~16 GB in 16-bit, more than the free T4 has, so a **4-bit** build is used
(Ollama's default `qwen3:8b` tag is a 4-bit quantisation **[verify on ollama.com/library/qwen3]**).

1. Open https://colab.research.google.com → *File → Upload notebook* → `notebooks/colab_qwen_server.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. The last cell prints a URL like `https://random-words.trycloudflare.com`.
4. In `.env`: `LLM_BASE_URL=https://random-words.trycloudflare.com/v1` and `LLM_MODEL=qwen3:8b`.
5. On Windows: `python scripts/check_llm.py` should print `reply: 'READY'`.

Notes:
* The URL changes every time the notebook restarts; update `.env` each session.
* Free Colab sessions disconnect after idle time / usage limits; keep the tab open.
* The quick-tunnel URL is public: anyone with it can use your model while it runs. Do not share it;
  stop the runtime when finished. Do not send private data through it.
* Read Colab's current terms of use for running servers/tunnels before relying on this
  **[verify]**. If not permitted, use Option B or C.

## Option B: local Ollama on Windows (no GPU needed, smaller model)
1. Install Ollama for Windows from https://ollama.com/download.
2. `ollama pull qwen3:4b` (or `qwen3:1.7b` on 8 GB RAM laptops). `qwen3:8b` on CPU works but is slow.
3. `.env`: `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen3:4b`.
Smaller models generate more invalid Cypher; the validator and retry make this safe, and the
benchmark measures it. Report which model you used.

## Option C: Hugging Face transformers inside Colab/Linux
`LLM_BACKEND=hf_transformers`, `pip install -r requirements-llm-local.txt` (bitsandbytes needs an
NVIDIA GPU, Linux/Colab). Model id from `configs/llm.yaml → hf_model_id`. Use this if you run the
whole pipeline inside Colab against a reachable Neo4j.

## Option D: another OpenAI-compatible endpoint
Any provider your institution permits. Put its key in `LLM_API_KEY` (never in code). Not required.

## Fallback models (same code, change LLM_MODEL)
| Model | When |
|---|---|
| Qwen3-8B (4-bit) | default, Colab T4 |
| Qwen3-4B | local CPU / weak GPU |
| Qwen3-1.7B | very limited RAM; expect lower Text-to-Cypher accuracy |

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Cannot reach LLM endpoint` | Colab stopped or URL changed; re-run notebook, update `.env` |
| HTTP 404 | `LLM_BASE_URL` must end with `/v1` |
| HTTP 404 "model not found" | `LLM_MODEL` must match `ollama list` in the notebook |
| very slow first answer | model is loading into GPU; second call is faster |
| answer contains `<think>` | handled automatically; if persistent, keep `disable_thinking: true` |
| Text-to-Cypher fails often | expected with small models; check `outputs/logs/ask_*.json`, improve few-shot examples in `prompts.py` |

## Optional later experiment: LoRA/QLoRA
Fine-tuning Qwen on (question, Cypher) pairs from your verified benchmark is a possible extension.
It is **not** needed for the baseline and must be evaluated on questions not used for training.
