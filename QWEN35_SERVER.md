# Qwen3.5 Server Notes

This project now keeps two model paths:

- Local primary: `QuantTrio/Qwen3.5-9B-AWQ` through vLLM.
- Remote fallback/check: `deepseek-chat` through the DeepSeek API.

The old generic OpenAI/OpenRouter/Claude style providers are not part of the intended setup.

The full-precision `Qwen/Qwen3.5-9B` loads on 2x 2080 Ti but leaves no KV-cache
budget even at 1024 context, so the practical local path is a Qwen3.5 9B AWQ
checkpoint.

Qwen3.5 is a multimodal architecture, but this project only uses text prompts.
The in-process vLLM client therefore starts it with `language_model_only=True`
by default so vLLM does not load/profile the vision encoder on 2080 Ti GPUs.
It also starts with `enable_thinking=False` by default because this benchmark
expects short entity-name outputs, not long reasoning traces.

## Do We Need Ollama?

No, not for the main server path. Qwen3.5 is officially published in Hugging Face format and can be served by vLLM, SGLang, KTransformers, or Transformers. vLLM is the best fit for this codebase because the project already uses vLLM directly.

Use Ollama only if the server cannot run vLLM and you have a compatible GGUF/Ollama package. That would be a separate low-throughput fallback, not the recommended experiment setup.

## Server Checklist

On the server, first check:

```bash
nvidia-smi
python -V
```

Practical rule of thumb:

- For this project's short prompts, start with `BPC_MAX_MODEL_LEN=32768` or lower.
- On 2x 2080 Ti, use a high `BPC_GPU_MEM` such as `0.90` or `0.92`; lower values
  can fail before any inference because the Qwen3.5 weights leave no KV-cache budget.
- If memory is still tight, reduce `BPC_MAX_MODEL_LEN` to `1024`.
- This run is intended for `CUDA_VISIBLE_DEVICES=0,1` with `BPC_TP=2`.

## Setup

```bash
cd /path/to/Bounded\ Path\ Context
bash scripts/setup_env.sh
conda activate bpc
```

## Start Local Qwen3.5 API

```bash
CUDA_VISIBLE_DEVICES=0,1 BPC_TP=2 BPC_MAX_MODEL_LEN=2048 bash scripts/serve_qwen35_vllm.sh
```

This exposes an OpenAI-compatible endpoint at:

```text
http://localhost:8000/v1
```

Quick smoke test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "QuantTrio/Qwen3.5-9B-AWQ",
    "messages": [{"role": "user", "content": "Return only: ok"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

## Run Experiments Directly

The experiment runner loads vLLM in-process by default:

```bash
BPC_MODEL=QuantTrio/Qwen3.5-9B-AWQ \
CUDA_VISIBLE_DEVICES=0,1 \
BPC_TP=2 \
BPC_LANGUAGE_MODEL_ONLY=1 \
BPC_ENABLE_THINKING=0 \
BPC_GPU_MEM=0.92 \
BPC_MAX_MODEL_LEN=2048 \
BPC_OUT=results/qwen35_smoke \
BPC_METHODS=bpc \
BPC_DATASETS=webqsp \
BPC_SUBSET=20 \
python experiments/emnlp_main_plus/run_env.py
```

If this OOMs, lower `BPC_MAX_MODEL_LEN` first, then lower batch/concurrency only if needed.
