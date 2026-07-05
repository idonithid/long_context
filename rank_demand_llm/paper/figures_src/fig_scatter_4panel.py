"""Fig 1: predicted (-K) vs measured Delta CE scatter, 4 panels."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import load_matrices, load_summary, SETTINGS

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_scatter_4panel.pdf")

fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4))

for ax, (setting, title) in zip(axes, SETTINGS):
    order, dce, ker = load_matrices(setting)
    xs, ys, offdiag = [], [], []
    for i in order:
        for j in order:
            xs.append(-ker[i][j])   # predicted Delta CE proportional to -K
            ys.append(dce[i][j])     # measured Delta CE
            offdiag.append(i != j)
    xs = np.array(xs); ys = np.array(ys); offdiag = np.array(offdiag)

    ax.scatter(xs[~offdiag], ys[~offdiag], s=42, c="#b0b0b0", marker="o",
               edgecolors="k", linewidths=0.4, label="diagonal (self)", zorder=2)
    ax.scatter(xs[offdiag], ys[offdiag], s=42, c="#1f77b4", marker="D",
               edgecolors="k", linewidths=0.4, label="off-diagonal", zorder=3)

    summ = load_summary(setting)
    rho_off, p_off = summ["off_diag_spearman"]
    rho_all, _ = summ["spearman_kernel_vs_dce"]
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"predicted $-K[i,j]$", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel(r"measured $\Delta$CE", fontsize=10)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ptxt = "<.001" if p_off < 0.001 else f"={p_off:.3f}"
    ax.text(0.04, 0.96,
            f"off-diag $\\rho$={rho_off:.2f} (p{ptxt})\nall $\\rho$={rho_all:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
