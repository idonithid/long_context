"""Fig 2: measured Delta CE vs predicted -K heatmaps (heterogeneous setting)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_matrices, LABELS

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_heatmaps_hetero.pdf")

setting = "interference_hetero"
order, dce, ker = load_matrices(setting)
labels = [LABELS[t] for t in order]
n = len(order)

M = np.array([[dce[i][j] for j in order] for i in order])       # measured
P = np.array([[-ker[i][j] for j in order] for i in order])      # predicted -K

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))


def draw(ax, A, title, cmap, unit):
    vmax = np.max(np.abs(A))
    im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("eval capability", fontsize=9)
    ax.set_ylabel("train capability", fontsize=9)
    ax.set_title(title, fontsize=11)
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)


draw(axes[0], M, r"measured $\Delta$CE", "RdBu_r", r"$\Delta$CE")
draw(axes[1], P, r"predicted $-K[i,j]$", "RdBu_r", r"$-K$")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
