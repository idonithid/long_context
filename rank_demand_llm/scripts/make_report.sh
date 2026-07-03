#!/usr/bin/env bash
# Regenerate analysis + report from an existing results dir.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/initzan/anaconda3/envs/rank_demand/bin/python}"
RESULTS="${1:-$REPO_DIR/results/smoke_run}"
REPORT="${2:-$REPO_DIR/reports}"
cd "$REPO_DIR"
"$PY" -m rank_demand.analyze --results "$RESULTS" --report "$REPORT"
