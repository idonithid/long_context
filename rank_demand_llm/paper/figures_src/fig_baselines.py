"""Fig 3: off-diagonal Spearman per predictor x setting (baseline shootout).

Numbers copied from reports/methods.md JSON (off_diag entries), which match
results/interference*/summary.json. Non-significant bars (p>0.05) are hatched.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_baselines.pdf")

settings = ["RULER", "Hetero", "LoRA", "Llama"]
predictors = ["ntk_kernel", "ntk_cos", "grad_magnitude", "embed_cos"]
pretty = ["ntk_kernel (ours)", "ntk_cos", "grad_magnitude", "embed_cos"]

# (rho, p) off-diagonal; None = no data (RULER embeddings missing)
DATA = {
    "RULER":  {"ntk_kernel": (0.6797, 0.001), "ntk_cos": (0.6677, 0.0013),
               "grad_magnitude": (0.2526, 0.2826), "embed_cos": None},
    "Hetero": {"ntk_kernel": (0.8737, 0.0), "ntk_cos": (0.7910, 0.0),
               "grad_magnitude": (0.8045, 0.0), "embed_cos": (-0.6241, 0.0033)},
    "LoRA":   {"ntk_kernel": (0.6180, 0.0037), "ntk_cos": (0.5504, 0.0119),
               "grad_magnitude": (0.2827, 0.2272), "embed_cos": (-0.2977, 0.2023)},
    "Llama":  {"ntk_kernel": (0.6917, 0.0007), "ntk_cos": (0.5910, 0.0061),
               "grad_magnitude": (0.0932, 0.6958), "embed_cos": (0.0241, 0.9198)},
}

colors = ["#1f77b4", "#6baed6", "#e6a24a", "#9e5db0"]
fig, ax = plt.subplots(figsize=(9.6, 4.2))
x = np.arange(len(settings))
w = 0.2

for k, pred in enumerate(predictors):
    vals, hatches, alphas = [], [], []
    for s in settings:
        cell = DATA[s][pred]
        if cell is None:
            vals.append(np.nan); hatches.append(""); alphas.append(1.0)
        else:
            rho, p = cell
            vals.append(rho)
            hatches.append("////" if p > 0.05 else "")
            alphas.append(0.45 if p > 0.05 else 1.0)
    pos = x + (k - 1.5) * w
    for xi, v, h, a in zip(pos, vals, hatches, alphas):
        if np.isnan(v):
            ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=6.5,
                    rotation=90, color="0.5")
            continue
        ax.bar(xi, v, w, color=colors[k], alpha=a, hatch=h,
               edgecolor="k", linewidth=0.4)

# legend proxies
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], ec="k", lw=0.4)
           for k in range(len(predictors))]
handles.append(plt.Rectangle((0, 0), 1, 1, fc="0.7", ec="k", lw=0.4, hatch="////"))
lbls = pretty + ["not significant (p>0.05)"]
ax.legend(handles, lbls, fontsize=8.5, ncol=2, loc="upper right", framealpha=0.9)

ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=10)
ax.set_ylabel(r"off-diagonal Spearman $\rho$ vs measured $\Delta$CE", fontsize=10)
ax.set_ylim(-0.75, 1.02)
ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
