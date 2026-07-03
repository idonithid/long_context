#!/usr/bin/env bash
# Clone NVIDIA/RULER into external/RULER and fetch its raw data dependencies:
#   - Paul Graham essays (haystack for essay-type NIAH tasks)
#   - SQuAD dev json (qa_1)
# Idempotent: skips anything already present.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULER_DIR="$REPO_DIR/external/RULER"
JSON_DIR="$RULER_DIR/scripts/data/synthetic/json"
PY="${PY:-/home/initzan/anaconda3/envs/rank_demand/bin/python}"

if [ ! -d "$RULER_DIR/.git" ]; then
    git clone --depth 1 https://github.com/NVIDIA/RULER.git "$RULER_DIR"
else
    echo "RULER already cloned at $RULER_DIR"
fi

# Paul Graham essays (used by essay-haystack niah tasks, e.g. niah_multikey_1)
if [ ! -f "$JSON_DIR/PaulGrahamEssays.json" ]; then
    echo "Downloading Paul Graham essays (one-time, a few minutes)..."
    (cd "$JSON_DIR" && "$PY" download_paulgraham_essay.py) \
        || echo "WARNING: essay download failed; essay-haystack tasks will not generate."
else
    echo "PaulGrahamEssays.json present"
fi

# SQuAD (qa_1)
if [ ! -f "$JSON_DIR/squad.json" ]; then
    echo "Downloading SQuAD dev set..."
    wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O "$JSON_DIR/squad.json" \
        || echo "WARNING: squad download failed; qa_1 will not generate."
else
    echo "squad.json present"
fi

echo "download_ruler.sh done."
