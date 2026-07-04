#!/usr/bin/env bash
# Main run: 5 task families x 50 samples x {1024,2048,4096,8192}.
# Long job -> runs inside a named tmux session with a tee'd log.
# Usage: run_main_ruler.sh [config.yaml] [run_name]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/initzan/anaconda3/envs/rank_demand/bin/python}"
CONFIG="${1:-configs/ruler_qwen2p5_7b.yaml}"
RUN_NAME="${2:-main_run}"
# awk must not early-exit: SIGPIPE to nvidia-smi + pipefail would kill the script
GPU="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 && !f {print $1; f=1}')}"
echo "Using GPU $GPU"
SESSION="rank_demand_${RUN_NAME}"
LOG="$REPO_DIR/results/$RUN_NAME/$RUN_NAME.log"
mkdir -p "$REPO_DIR/results/$RUN_NAME"

CMD="cd $REPO_DIR && CUDA_VISIBLE_DEVICES=$GPU $PY scripts/prepare_ruler_data.py \
  --config $CONFIG --mode main && \
CUDA_VISIBLE_DEVICES=$GPU $PY -m rank_demand.eval_ruler \
  --config $CONFIG --mode main --output_dir results/$RUN_NAME && \
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_ntk_features.py \
  --config $CONFIG --mode main --output_dir results/$RUN_NAME \
  --max_ntk_tokens 8192 && \
$PY -m rank_demand.analyze --results results/$RUN_NAME --report reports/$RUN_NAME"

if [ -n "${NO_TMUX:-}" ]; then
    eval "$CMD" 2>&1 | tee "$LOG"
else
    tmux new-session -d -s "$SESSION" "($CMD) 2>&1 | tee $LOG"
    echo "Launched tmux session '$SESSION' (GPU $GPU). Follow with:"
    echo "  tmux attach -t $SESSION   # or: tail -f $LOG"
fi
