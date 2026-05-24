#!/bin/bash
# Start a local OpenAI-compatible vLLM server for Qwen3.5.
# Usage:
#   bash scripts/serve_qwen35_vllm.sh
# Optional env:
#   BPC_MODEL=QuantTrio/Qwen3.5-9B-AWQ
#   BPC_PORT=8000
#   BPC_TP=1
#   BPC_MAX_MODEL_LEN=32768

set -euo pipefail

MODEL="${BPC_MODEL:-QuantTrio/Qwen3.5-9B-AWQ}"
PORT="${BPC_PORT:-8000}"
TP="${BPC_TP:-1}"
MAX_MODEL_LEN="${BPC_MAX_MODEL_LEN:-32768}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "model=$MODEL"
echo "port=$PORT"
echo "tensor_parallel_size=$TP"
echo "max_model_len=$MAX_MODEL_LEN"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found; vLLM needs CUDA GPUs for this path." >&2
fi

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --language-model-only \
    "$@"
