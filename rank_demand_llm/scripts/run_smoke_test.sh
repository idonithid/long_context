#!/usr/bin/env bash
# Smoke test: 2 task families, 5 examples each, 1024 tokens, on one GPU.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/initzan/anaconda3/envs/rank_demand/bin/python}"
# default: first idle GPU (<1GB used), unless caller pins CUDA_VISIBLE_DEVICES
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 1000 && !f {print $1; f=1}')
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPU $CUDA_VISIBLE_DEVICES"

cd "$REPO_DIR"
"$PY" scripts/prepare_ruler_data.py --config configs/ruler_qwen2p5_7b.yaml \
    --mode smoke --tasks niah_single_1 vt --lengths 1024 --num_samples 5

"$PY" -m rank_demand.eval_ruler --config configs/ruler_qwen2p5_7b.yaml \
    --mode smoke --tasks niah_single_1 vt --lengths 1024 --num_samples 5 \
    --output_dir results/smoke_run

"$PY" scripts/run_ntk_features.py --config configs/ruler_qwen2p5_7b.yaml \
    --mode smoke --tasks niah_single_1 vt --lengths 1024 --num_samples 5 \
    --output_dir results/smoke_run

"$PY" -m rank_demand.analyze --results results/smoke_run --report reports
echo "Smoke test done. Report: $REPO_DIR/reports/summary.md"
