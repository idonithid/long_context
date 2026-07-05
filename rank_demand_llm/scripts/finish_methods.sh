#!/usr/bin/env bash
# Laptop-close-safe finisher: waits for the remaining interference runs
# (LoRA robustness + Llama replication), then builds the combined method
# report. Runs inside tmux on the box; needs no live client session.
set -u
REPO=/home/initzan/long_context/rank_demand_llm
PY=/home/initzan/anaconda3/envs/rank_demand/bin/python
cd "$REPO"

LORA=results/interference_lora/summary.json
LLAMA=results/interference_llama/summary.json
DEADLINE=$(( $(date +%s) + 24*3600 ))

echo "$(date) waiting for: $LORA + $LLAMA"
while true; do
    lora_done=0; llama_done=0
    [ -f "$LORA" ] && lora_done=1
    [ -f "$LLAMA" ] && llama_done=1
    if [ $lora_done = 1 ] && [ $llama_done = 1 ]; then break; fi
    # detect dead sessions with no result: still finish with what exists
    alive=$(tmux ls 2>/dev/null | grep -cE "lora_scale|llama_interference" || true)
    if [ "$alive" = 0 ]; then
        echo "$(date) WARNING: worker sessions gone (lora=$lora_done llama=$llama_done); finishing with available results"
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "$(date) TIMEOUT after 24h; finishing with available results"
        break
    fi
    sleep 300
done

echo "$(date) building combined method report"
"$PY" scripts/analyze_methods.py \
    --selfgrad results/main_run/selfgrad.jsonl \
    --selfgrad_llama results/llama_run/selfgrad.jsonl \
    --interference results/interference \
    --interference_extra results/interference_hetero results/interference_lora results/interference_llama \
    --out reports/methods.md > reports/methods_analysis.log 2>&1 \
    || "$PY" scripts/analyze_methods.py \
        --selfgrad results/main_run/selfgrad.jsonl \
        --selfgrad_llama results/llama_run/selfgrad.jsonl \
        --interference results/interference \
        --out reports/methods.md >> reports/methods_analysis.log 2>&1

echo "$(date) DONE" | tee results/FINISH_STATUS.txt
{
    echo "finished: $(date)"
    echo "reports/methods.md ready"
    for f in results/interference results/interference_hetero results/interference_lora results/interference_llama; do
        if [ -f "$f/summary.json" ]; then
            echo "== $f =="; cat "$f/summary.json"
        else
            echo "== $f == MISSING"
        fi
    done
} >> results/FINISH_STATUS.txt
