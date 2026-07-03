"""Diagnostic plots (matplotlib, static PNG).

Style: validated categorical palette with FIXED slot order per task family
(color follows entity, never rank), recessive grid, thin marks, one axis per
panel, direct labels where room allows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger("rank_demand.plots")

# validated categorical palette (light mode), fixed order
_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
            "#e34948", "#e87ba4", "#eb6834"]
# fixed slot per family — never re-assigned when a family is filtered out
FAMILY_ORDER = ["single_needle", "multi_needle", "multi_hop_tracing",
                "aggregation", "qa"]
FAMILY_COLOR = {f: _PALETTE[i] for i, f in enumerate(FAMILY_ORDER)}
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": _INK,
    "text.color": _INK, "xtick.color": _MUTED, "ytick.color": _MUTED,
    "axes.grid": True, "grid.color": _GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "figure.dpi": 130,
})


def _fam_color(family: str) -> str:
    return FAMILY_COLOR.get(family, _MUTED)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot -> %s", path)


def accuracy_vs_length(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam in FAMILY_ORDER:
        sub = df[df.task_family == fam]
        if sub.empty:
            continue
        g = sub.groupby("target_context_length")["correct"].mean()
        ax.plot(g.index, g.values, "-o", color=_fam_color(fam), lw=2,
                ms=6, label=fam)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy vs context length by task family")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out)


def rank_vs_layer(df_layer: pd.DataFrame, value_col: str, split_col: str,
                  title: str, ylabel: str, out: Path,
                  split_colors: dict | None = None):
    """Generic per-layer line plot; df_layer has columns [layer, value_col, split_col]."""
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = (FAMILY_ORDER if split_col == "task_family"
              else sorted(df_layer[split_col].dropna().unique()))
    for gval in groups:
        sub = df_layer[df_layer[split_col] == gval]
        if sub.empty:
            continue
        g = sub.groupby("layer")[value_col].agg(["mean", "sem"])
        color = (split_colors or {}).get(gval) or _fam_color(str(gval))
        ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=color, lw=2,
                    marker="o", ms=5, capsize=2, label=str(gval))
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out)


def entropy_vs_rank_scatter(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for fam in FAMILY_ORDER:
        sub = df[df.task_family == fam]
        if sub.empty:
            continue
        ax.scatter(sub["attn_entropy_mean"], sub["attn_block_r_eff_mean"],
                   s=28, color=_fam_color(fam), alpha=0.75, edgecolors=_SURFACE,
                   linewidths=0.8, label=fam)
    ax.set_xlabel("attention entropy (mean over layers/heads)")
    ax.set_ylabel("attention block effective rank (mean)")
    ax.set_title("Attention entropy vs block effective rank")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out)


def correctness_vs_rank_scatter(df: pd.DataFrame, rank_col: str, out: Path,
                                xlabel: str):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rng = np.random.default_rng(0)
    ok = df[df.correct == True]   # noqa: E712
    bad = df[df.correct == False]  # noqa: E712
    for sub, y0, color, label in [(bad, 0, "#d03b3b", "incorrect"),
                                  (ok, 1, "#0ca30c", "correct")]:
        if sub.empty:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(sub[rank_col], y0 + jitter, s=26, color=color, alpha=0.7,
                   edgecolors=_SURFACE, linewidths=0.8, label=label)
    ax.set_yticks([0, 1], ["incorrect", "correct"])
    ax.set_xlabel(xlabel)
    ax.set_title(f"Correctness vs {xlabel}")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _save(fig, out)


def ntk_gram_heatmap(G: np.ndarray, labels: list[str], out: Path):
    """Cosine-normalized NTK Gram, samples ordered/annotated by task family.
    Sequential single-hue colormap (magnitude encoding)."""
    from matplotlib.colors import LinearSegmentedColormap

    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = G / np.outer(d, d)
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(C, cmap=cmap, vmin=-0.1, vmax=1.0)
    # family boundaries
    bounds = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color=_INK, lw=0.8)
        ax.axvline(b - 0.5, color=_INK, lw=0.8)
    # family tick labels at group centers
    centers, names = [], []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            centers.append((start + i - 1) / 2)
            names.append(labels[start])
            start = i
    ax.set_xticks(centers, names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(centers, names, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="NTK cosine similarity")
    ax.set_title("Empirical NTK Gram (sketched), grouped by task family")
    _save(fig, out)


def evidence_survival_curves(curves: dict, out: Path):
    """curves: {layer: {"ks": [...], "ev_mean": [...], "rand_mean": [...], "n": int}}"""
    layers = sorted(curves, key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(2.6 * len(layers), 3.2),
                             sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, li in zip(axes, layers):
        c = curves[li]
        ax.plot(c["ks"], c["ev_mean"], "-o", color="#2a78d6", lw=2, ms=4,
                label="evidence")
        if c.get("rand_mean") is not None:
            ax.plot(c["ks"], c["rand_mean"], "-o", color="#898781", lw=2, ms=4,
                    label="random")
        ax.set_xscale("log", base=2)
        ax.set_title(f"layer {li}", fontsize=9)
        ax.set_xlabel("k (top eigvecs)")
    axes[0].set_ylabel("survival $\\|H V_k V_k^T\\|_F^2 / \\|H\\|_F^2$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Evidence survival in top-k covariance subspace", y=1.02)
    _save(fig, out)
