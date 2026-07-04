#!/usr/bin/env bash
# Main first run: 5 task families x 50 samples x {1024,2048,4096,8192}.
# Long job -> runs inside a named tmux session with a tee'd log.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/initzan/anaconda3/envs/rank_demand/bin/python}"
# awk must not early-exit: SIGPIPE to nvidia-smi + pipefail would kill the script
GPU="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 && !f {print $1; f=1}')}"
echo "Using GPU $GPU"
SESSION=rank_demand_main
LOG="$REPO_DIR/results/main_run/main_run.log"
mkdir -p "$REPO_DIR/results/main_run"

CMD="cd $REPO_DIR && CUDA_VISIBLE_DEVICES=$GPU $PY scripts/prepare_ruler_data.py \
  --config configs/ruler_qwen2p5_7b.yaml --mode main && \
CUDA_VISIBLE_DEVICES=$GPU $PY -m rank_demand.eval_ruler \
  --config configs/ruler_qwen2p5_7b.yaml --mode main --output_dir results/main_run && \
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_ntk_features.py \
  --config configs/ruler_qwen2p5_7b.yaml --mode main --output_dir results/main_run \
  --max_ntk_tokens 4096 && \
$PY -m rank_demand.analyze --results results/main_run --report reports/main"

if [ -n "${NO_TMUX:-}" ]; then
    eval "$CMD" 2>&1 | tee "$LOG"
else
    tmux new-session -d -s "$SESSION" "($CMD) 2>&1 | tee $LOG"
    echo "Launched tmux session '$SESSION' (GPU $GPU). Follow with:"
    echo "  tmux attach -t $SESSION   # or: tail -f $LOG"
fi
