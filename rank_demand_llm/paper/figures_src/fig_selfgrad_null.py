"""Fig 5: self-gradient null. AUC of grad-norm vs free confidence baselines.

Numbers from reports/methods.md JSON, exp1_selfgrad (Qwen2.5-7B, Llama-3.1-8B).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_selfgrad_null.pdf")

feats = ["self_grad_norm", "self_answer_ce", "min_token_logprob",
         "mean_token_entropy", "max_token_entropy"]
flabel = ["grad norm\n(proposed)", "answer CE", "min logprob",
          "mean entropy", "max entropy"]
# (auc, lo, hi)
qwen = {
    "self_grad_norm": (0.7757, 0.7383, 0.8133),
    "self_answer_ce": (0.8990, 0.8699, 0.9283),
    "min_token_logprob": (0.8726, 0.8389, 0.9046),
    "mean_token_entropy": (0.9020, 0.8748, 0.9288),
    "max_token_entropy": (0.8740, 0.8380, 0.9076),
}
llama = {
    "self_grad_norm": (0.7579, 0.7055, 0.8103),
    "self_answer_ce": (0.8912, 0.8517, 0.9251),
    "min_token_logprob": (0.8462, 0.7900, 0.8924),
    "mean_token_entropy": (0.9055, 0.8726, 0.9328),
    "max_token_entropy": (0.8918, 0.8531, 0.9251),
}
dcv = {"Qwen2.5-7B": -0.0016, "Llama-3.1-8B": -0.0001}

fig, ax = plt.subplots(figsize=(8.4, 4.0))
x = np.arange(len(feats))
w = 0.38
for off, (name, d, c) in enumerate([("Qwen2.5-7B", qwen, "#1f77b4"),
                                     ("Llama-3.1-8B", llama, "#e6a24a")]):
    vals = [d[f][0] for f in feats]
    err = [[d[f][0] - d[f][1] for f in feats], [d[f][2] - d[f][0] for f in feats]]
    pos = x + (off - 0.5) * w
    bars = ax.bar(pos, vals, w, yerr=err, capsize=2.5, color=c,
                  edgecolor="k", linewidth=0.4, label=name,
                  error_kw=dict(lw=0.8))
    bars[0].set_alpha(0.55)
    bars[0].set_hatch("////")

ax.set_xticks(x); ax.set_xticklabels(flabel, fontsize=8.5)
ax.set_ylabel("AUC (predict per-sample correctness)", fontsize=10)
ax.set_ylim(0.5, 0.98)
ax.axhline(0.5, color="k", lw=0.5, ls=":")
ax.set_title("Self-gradient norm is confidence-in-disguise: "
             r"CV $\Delta$AUC over baselines $\approx$ "
             f"{dcv['Qwen2.5-7B']:+.4f} / {dcv['Llama-3.1-8B']:+.4f}",
             fontsize=9.5)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
