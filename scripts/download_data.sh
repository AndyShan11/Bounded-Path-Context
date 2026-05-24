#!/bin/bash
# Download datasets for Bounded Path Context experiments
# Robust: absolute python path via CONDA_PREFIX, unset contaminating env vars

set -e

# Clean env to avoid /home/shanxh/python3.10 site-packages contamination
unset PYTHONPATH PYTHONUSERBASE PIP_TARGET PIP_USER PIP_CONFIG_FILE

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${BPC_ENV:-bpc}"

BPC_PY="$CONDA_PREFIX/bin/python"
echo "Using HF_ENDPOINT = $HF_ENDPOINT"
echo "Using python      = $BPC_PY"

mkdir -p data
cd data

echo "==============================================="
echo "Bounded Path Context — Data download"
echo "==============================================="

# RoG (Luo et al. ICLR 2024) hosts pre-processed subgraphs + QA splits:
#   rmanluo/RoG-webqsp, rmanluo/RoG-cwq

echo "[1/3] Pulling WebQSP (RoG-aligned)..."
"$BPC_PY" -c "
from datasets import load_dataset
ds = load_dataset('rmanluo/RoG-webqsp')
ds.save_to_disk('webqsp')
print('WebQSP splits:', list(ds.keys()))
for split, d in ds.items():
    print(f'  {split}: {len(d)} examples; columns={d.column_names}')
"

echo "[2/3] Pulling CWQ (RoG-aligned)..."
"$BPC_PY" -c "
from datasets import load_dataset
ds = load_dataset('rmanluo/RoG-cwq')
ds.save_to_disk('cwq')
print('CWQ splits:', list(ds.keys()))
for split, d in ds.items():
    print(f'  {split}: {len(d)} examples; columns={d.column_names}')
"

echo "[3/3] GRADE difficulty labels..."
echo "  TODO: confirm release URL from GRADE paper (EMNLP 2025 Findings)."

cd ..
echo ""
echo "===== DATA DOWNLOAD COMPLETE ====="
