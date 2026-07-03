#!/usr/bin/env bash
# Create conda env `rank_demand` (python 3.10) and install dependencies.
# Idempotent: skips env creation if it already exists.
set -euo pipefail

ENV_NAME=rank_demand
PYTHON_VERSION=3.10
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_BASE=""
for c in "$HOME/anaconda3" "$HOME/miniconda" "$HOME/miniconda3"; do
    if [ -x "$c/bin/conda" ]; then CONDA_BASE="$c"; break; fi
done
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: no conda installation found" >&2
    exit 1
fi
echo "Using conda at $CONDA_BASE"

# conda may place envs outside $CONDA_BASE (envs_dirs config) -> resolve via conda itself
env_path() {
    "$CONDA_BASE/bin/conda" env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}'
}

ENV_PATH="$(env_path)"
if [ -z "$ENV_PATH" ]; then
    "$CONDA_BASE/bin/conda" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    ENV_PATH="$(env_path)"
else
    echo "Env $ENV_NAME already exists at $ENV_PATH, skipping creation."
fi

PY="$ENV_PATH/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: python not found at $PY" >&2
    exit 1
fi

# Torch with CUDA 12.1 wheels (works on Turing sm_75 and newer)
"$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install --quiet -r "$REPO_DIR/requirements.txt"

echo "=== Environment check ==="
"$PY" - <<'EOF'
import torch, transformers
print(f"python ok, torch {torch.__version__}, transformers {transformers.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB, sm_{p.major}{p.minor}")
    cc = torch.cuda.get_device_capability(0)
    if cc < (8, 0):
        print("NOTE: compute capability < 8.0 -> no native bf16 / no flash-attn; "
              "pipeline will default to fp16 + sdpa/eager.")
else:
    print("WARNING: CUDA not available")
EOF
echo "Setup done. Run with: $PY"
