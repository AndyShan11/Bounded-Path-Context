#!/bin/bash
# Server-side env setup for Bounded Path Context
# Target: Rocky Linux 9 + 4x RTX 2080 Ti
# Robustness notes:
#   - Uses absolute $CONDA_PREFIX/bin paths (PATH/PYTHONPATH may be contaminated by user's /home/shanxh/python3.10 install)
#   - Unsets PYTHONPATH/PYTHONUSERBASE/PIP_TARGET/PIP_USER inside the script
#   - Removes any pre-existing bpc env for a clean slate

set -e

ENV_NAME="${BPC_ENV:-bpc}"
PYTHON_VERSION="3.11"

echo "[1/6] HF mirror config (huggingface.co blocked)..."
export HF_ENDPOINT="https://hf-mirror.com"
if ! grep -q "HF_ENDPOINT" "$HOME/.bashrc"; then
    echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> "$HOME/.bashrc"
fi

echo "[2/6] Unset contaminating env vars (PYTHONPATH/PYTHONUSERBASE/PIP_TARGET/PIP_USER)..."
unset PYTHONPATH PYTHONUSERBASE PIP_TARGET PIP_USER PIP_CONFIG_FILE
echo "  (will pass clean env to pip)"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[3/6] Recreating conda env $ENV_NAME (clean slate, python $PYTHON_VERSION)..."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
conda activate "$ENV_NAME"

# Absolute paths — robust against PATH/alias weirdness
BPC_PY="$CONDA_PREFIX/bin/python"
BPC_PIP="$CONDA_PREFIX/bin/pip"
echo "  CONDA_PREFIX = $CONDA_PREFIX"
echo "  BPC_PY       = $BPC_PY"
echo "  BPC_PIP      = $BPC_PIP"
"$BPC_PY" --version
"$BPC_PIP" --version

echo "[4/6] Installing current vLLM for Qwen3.5 (~6GB download)..."
"$BPC_PIP" install --upgrade --no-cache-dir pip uv
UV_BIN="$CONDA_PREFIX/bin/uv"
"$UV_BIN" pip install --python "$BPC_PY" --no-cache vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

echo "[5/6] Installing remaining requirements into bpc..."
"$BPC_PIP" install --no-cache-dir -r requirements.txt

echo "[6/6] Cloning baselines..."
mkdir -p baselines
cd baselines
[ -d ToG ] || git clone https://github.com/IDEA-FinAI/ToG.git
[ -d reasoning-on-graphs ] || git clone https://github.com/RManLuo/reasoning-on-graphs.git
cd ..

echo ""
echo "==== Sanity check ===="
"$BPC_PY" -c "import sys; print('python:', sys.executable); print('site:', sys.path[:3])"
"$BPC_PY" -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}')"
"$BPC_PY" -c "import vllm; print(f'vllm={vllm.__version__}')"

echo ""
echo "===== SETUP COMPLETE ====="
echo "BPC_PY = $BPC_PY"
