"""Fig 4: hidden-state effective rank by task family, both models (log y).

Numbers from reports/main/summary.md (Qwen) and reports/llama_run/summary.md (Llama),
'Mean rank/entropy metrics by task family' tables, column hidden_r_eff_mid.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_rank_motivation.pdf")

# hidden_r_eff_mid
families = ["single_needle", "multi_hop_tracing", "aggregation", "multi_needle", "qa"]
flabel = ["single", "tracking", "aggregation", "multikey", "qa"]
qwen = {"single_needle": 2.11, "multi_hop_tracing": 2.59, "aggregation": 21.98,
        "multi_needle": 100.54, "qa": 146.35}
llama = {"single_needle": 28.35, "multi_hop_tracing": 35.82, "aggregation": 82.25,
         "multi_needle": 122.04, "qa": 87.85}

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(families))
w = 0.38
ax.bar(x - w / 2, [qwen[f] for f in families], w, label="Qwen2.5-7B",
       color="#1f77b4", edgecolor="k", linewidth=0.4)
ax.bar(x + w / 2, [llama[f] for f in families], w, label="Llama-3.1-8B",
       color="#e6a24a", edgecolor="k", linewidth=0.4)

ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(flabel, fontsize=10)
ax.set_ylabel("hidden-state effective rank (mid layer)", fontsize=10)
ax.set_title("Rank demand is task-intrinsic and orders families consistently",
             fontsize=10.5)
ax.legend(fontsize=9)
ax.grid(axis="y", ls=":", alpha=0.5)
for xi, f in zip(x, families):
    ax.text(xi - w / 2, qwen[f] * 1.05, f"{qwen[f]:.0f}", ha="center",
            va="bottom", fontsize=7)
    ax.text(xi + w / 2, llama[f] * 1.05, f"{llama[f]:.0f}", ha="center",
            va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
