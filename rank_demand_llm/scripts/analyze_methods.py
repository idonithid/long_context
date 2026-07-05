#!/usr/bin/env python
"""Analysis for the two method experiments.

Exp 1 (selfgrad): does the self-answer gradient norm predict correctness
beyond cheap confidence baselines (mean/min token logprob, entropy)?
  - pooled AUC per signal (+ 95% bootstrap CI)
  - CV logistic: baselines-only vs baselines+grad -> delta AUC
  - within-(family,length) AUC (controls composition effects)

Exp 2 (interference): predicted sketch-kernel vs measured dCE matrices.

  python scripts/analyze_methods.py --selfgrad results/main_run/selfgrad.jsonl \
      --selfgrad_llama results/llama_run/selfgrad.jsonl \
      --interference results/interference --out reports/methods.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand.utils import read_jsonl, setup_logging  # noqa: E402

SIGNALS = {
    "self_grad_norm": ("grad_norm", +1),        # higher grad -> predict incorrect
    "self_answer_ce": ("answer_ce", +1),        # == -mean logprob
    "min_token_logprob": ("min_token_logprob", -1),
    "mean_token_entropy": ("mean_token_entropy", +1),
    "max_token_entropy": ("max_token_entropy", +1),
}


def auc_ci(y, score, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(auc), 4), [round(float(lo), 4), round(float(hi), 4)]


def analyze_selfgrad(path: Path, label: str) -> dict:
    rows = [r for r in read_jsonl(path) if "grad_norm" in r
            and r.get("correct") is not None]
    d = pd.DataFrame(rows)
    out = {"model": label, "n": len(d),
           "n_incorrect": int((~d.correct.astype(bool)).sum())}
    if out["n_incorrect"] < 10:
        out["skip_reason"] = "too few incorrect samples"
        return out
    y = (~d["correct"].astype(bool)).astype(int).values  # predict INCORRECT
    out["auc"] = {}
    for name, (col, sign) in SIGNALS.items():
        if col not in d or d[col].isna().any():
            continue
        auc, ci = auc_ci(y, sign * d[col].values)
        out["auc"][name] = {"auc": auc, "ci95": ci}

    # incremental value: CV logistic, baselines vs baselines+grad
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_cols = ["answer_ce", "min_token_logprob", "mean_token_entropy",
                 "max_token_entropy"]
    base_cols = [c for c in base_cols if c in d]
    Xb = d[base_cols].values
    Xg = np.column_stack([Xb, np.log1p(d["grad_norm"].values)])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res_cv = {}
    for tag, X in [("baselines", Xb), ("baselines+grad", Xg)]:
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))
        prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        res_cv[tag] = round(float(roc_auc_score(y, prob)), 4)
    res_cv["delta"] = round(res_cv["baselines+grad"] - res_cv["baselines"], 4)
    out["cv_logistic"] = res_cv

    # within-cell AUC (family x length cells with both classes)
    from sklearn.metrics import roc_auc_score as ras
    cells = []
    for (fam, ln), g in d.groupby(["task_family", "target_context_length"]):
        yy = (~g.correct.astype(bool)).astype(int).values
        if yy.min() == yy.max() or len(g) < 10:
            continue
        cells.append({
            "cell": f"{fam}@{ln}", "n": len(g),
            "auc_grad": round(float(ras(yy, g["grad_norm"])), 3),
            "auc_ce": round(float(ras(yy, g["answer_ce"])), 3),
        })
    out["within_cell"] = cells
    return out


def analyze_interference(idir: Path) -> dict:
    summ = json.loads((idir / "summary.json").read_text())
    measured = json.loads((idir / "measured.json").read_text())
    kernel = json.loads((idir / "predicted_kernel.json").read_text())
    fams = list(measured.keys())
    out = {"summary": summ}
    out["measured_dce"] = {ti: {tj: measured[ti]["post"][tj]["dce"]
                                for tj in fams} for ti in fams}
    out["measured_dacc"] = {ti: {tj: measured[ti]["post"][tj]["dacc"]
                                 for tj in fams} for ti in fams}
    out["predicted_kernel"] = {k: v["mean_kernel"] for k, v in kernel.items()}
    out["baselines"] = baseline_comparison(idir, measured, kernel, fams)
    return out


def baseline_comparison(idir: Path, measured, kernel, fams) -> dict:
    """Predictor shoot-out on the same measured dCE matrix.

    Predictors (all computed pre-training):
      ntk_kernel      mean <g_i, g_j> (the method)
      ntk_cos         mean cosine (direction only)
      grad_magnitude  |g_i|*|g_j| (magnitude only — is direction needed?)
      embed_cos       mean cosine of last-hidden-state prompt embeddings
    """
    from scipy.stats import spearmanr

    ids_by = {}
    sk_path, emb_path = idir / "sketches.npz", idir / "embeddings.npz"
    if not sk_path.exists():
        return {"skip_reason": "no sketches.npz"}
    z = np.load(sk_path)
    for key in z.files:
        tag, sid = key.split("__", 1)
        # sample_id usually starts with the capability name (qa_1_1024_7);
        # tolerate ids that use a shortened stem (mbpp_eval_0 vs mbpp_code)
        stem = sid.split(f"_{tag}_")[0] if f"_{tag}_" in sid else sid
        for t in sorted(fams, key=len, reverse=True):
            if sid.startswith(t) or t.startswith(stem):
                ids_by.setdefault((tag, t), []).append(key)
                break
    emb = np.load(emb_path) if emb_path.exists() else None

    preds: dict[str, dict] = {"ntk_kernel": {}, "ntk_cos": {},
                              "grad_magnitude": {}, "embed_cos": {}}
    for ti in fams:
        Si = np.stack([z[k] for k in ids_by[("train", ti)]])
        ni = np.linalg.norm(Si, axis=1, keepdims=True)
        for tj in fams:
            Sj = np.stack([z[k] for k in ids_by[("eval", tj)]])
            nj = np.linalg.norm(Sj, axis=1, keepdims=True)
            G = Si @ Sj.T
            preds["ntk_kernel"][(ti, tj)] = G.mean()
            preds["ntk_cos"][(ti, tj)] = (G / (ni @ nj.T + 1e-30)).mean()
            preds["grad_magnitude"][(ti, tj)] = (ni @ nj.T).mean()
            if emb is not None:
                Ei = np.stack([emb[k] for k in ids_by[("train", ti)] if k in emb.files])
                Ej = np.stack([emb[k] for k in ids_by[("eval", tj)] if k in emb.files])
                Ei /= np.linalg.norm(Ei, axis=1, keepdims=True) + 1e-30
                Ej /= np.linalg.norm(Ej, axis=1, keepdims=True) + 1e-30
                preds["embed_cos"][(ti, tj)] = (Ei @ Ej.T).mean()

    pairs_all = [(ti, tj) for ti in fams for tj in fams]
    pairs_off = [(ti, tj) for ti in fams for tj in fams if ti != tj]
    meas = {p: measured[p[0]]["post"][p[1]]["dce"] for p in pairs_all}
    res = {}
    for name, P in preds.items():
        if not P:
            res[name] = {"skip_reason": "no data (embeddings missing?)"}
            continue
        for tag, pairs in [("all", pairs_all), ("off_diag", pairs_off)]:
            rho, pval = spearmanr([-P[p] for p in pairs],
                                  [meas[p] for p in pairs])
            res.setdefault(name, {})[tag] = [round(float(rho), 4),
                                             round(float(pval), 4)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfgrad", default=None)
    ap.add_argument("--selfgrad_llama", default=None)
    ap.add_argument("--interference", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports/methods.md"))
    args = ap.parse_args()
    setup_logging()

    report = {"exp1_selfgrad": [], "exp2_interference": None}
    for p, label in [(args.selfgrad, "Qwen2.5-7B"),
                     (args.selfgrad_llama, "Llama-3.1-8B")]:
        if p and Path(p).exists():
            report["exp1_selfgrad"].append(analyze_selfgrad(Path(p), label))
    if args.interference and Path(args.interference, "summary.json").exists():
        report["exp2_interference"] = analyze_interference(Path(args.interference))

    print(json.dumps(report, indent=2, default=str))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Method experiments\n\n```json\n"
                   + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
