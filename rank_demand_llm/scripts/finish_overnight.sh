#!/usr/bin/env bash
# Disconnect-safe finisher for the overnight method experiments:
#   ntk_caps      -> results/interference_caps8/summary.json
#   ntk_horizon   -> results/interference_h{1,9,27,81}/summary.json
#   ntk_rehearsal -> results/rehearsal/summary.json
# Waits (12h deadline), then builds reports/overnight.md.
set -u
REPO=/home/initzan/long_context/rank_demand_llm
PY=/home/initzan/anaconda3/envs/rank_demand/bin/python
cd "$REPO"

WANT="results/interference_caps8/summary.json results/interference_h81/summary.json results/rehearsal/summary.json"
DEADLINE=$(( $(date +%s) + 12*3600 ))
echo "$(date) waiting for: $WANT"
while true; do
    missing=0
    for f in $WANT; do [ -f "$f" ] || missing=1; done
    [ "$missing" = 0 ] && break
    alive=$(tmux ls 2>/dev/null | grep -cE "ntk_caps|ntk_horizon|ntk_rehearsal" || true)
    if [ "$alive" = 0 ]; then
        echo "$(date) WARNING: worker sessions gone; finishing with available results"
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "$(date) TIMEOUT after 12h; finishing with available results"
        break
    fi
    sleep 300
done

echo "$(date) building overnight report"
"$PY" scripts/analyze_methods.py \
    --interference results/interference \
    --interference_extra results/interference_hetero results/interference_lora \
        results/interference_llama results/interference_caps8 \
        results/interference_h1 results/interference_h9 \
        results/interference_h27 results/interference_h81 \
    --out reports/methods_overnight.md > reports/overnight_analysis.log 2>&1

"$PY" - <<'EOF' >> reports/overnight_analysis.log 2>&1
import json
from pathlib import Path

rep = ["# Overnight experiments\n"]

rep.append("## 1. Extended capability set (8 caps: +safety/+xquad_zh/+arc_c)\n")
p = Path("results/interference_caps8/summary.json")
rep.append("```json\n" + p.read_text() + "\n```\n" if p.exists() else "MISSING\n")

rep.append("## 2. Horizon scaling — does first-order prediction decay?\n")
rep.append("| epochs | spearman all | off-diag spearman |")
rep.append("|---|---|---|")
for e, d in [(1, "results/interference_h1"), (3, "results/interference_hetero"),
             (9, "results/interference_h9"), (27, "results/interference_h27"),
             (81, "results/interference_h81")]:
    q = Path(d) / "summary.json"
    if q.exists():
        s = json.loads(q.read_text())
        rep.append(f"| {e} | {s['spearman_kernel_vs_dce']} | {s['off_diag_spearman']} |")
    else:
        rep.append(f"| {e} | MISSING | |")
rep.append("")

rep.append("## 3. Rehearsal targeting (kernel-selected vs random vs none)\n")
p = Path("results/rehearsal/summary.json")
if p.exists():
    s = json.loads(p.read_text())
    s.pop("per_sample_spearman_none", None) and None
    rep.append("```json\n" + json.dumps(json.loads(p.read_text()), indent=2) + "\n```\n")
else:
    rep.append("MISSING\n")

Path("reports/overnight.md").write_text("\n".join(rep))
print("report -> reports/overnight.md")
EOF

echo "$(date) OVERNIGHT DONE" | tee results/OVERNIGHT_STATUS.txt
for f in results/interference_caps8 results/interference_h1 results/interference_h9 \
         results/interference_h27 results/interference_h81 results/rehearsal; do
    if [ -f "$f/summary.json" ]; then
        echo "== $f ==" >> results/OVERNIGHT_STATUS.txt
        cat "$f/summary.json" >> results/OVERNIGHT_STATUS.txt
    else
        echo "== $f == MISSING" >> results/OVERNIGHT_STATUS.txt
    fi
done
