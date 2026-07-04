"""Analysis: summary tables, plots, logistic-regression ablations, report.

Usage:
  python -m rank_demand.analyze --results results/smoke_run --report reports
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand import plots  # noqa: E402
from rank_demand.utils import read_jsonl, setup_logging  # noqa: E402

logger = logging.getLogger("rank_demand.analyze")

MID = "mid"  # feature source: layer at 50% depth


def _mid_layer(layers: list[str]) -> str:
    ls = sorted(int(x) for x in layers)
    return str(ls[len(ls) // 2]) if ls else None


def extract_features(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten result rows -> (per-sample df, per-sample-per-layer df)."""
    recs, layer_recs = [], []
    for r in rows:
        ms = r.get("metrics_summary") or {}
        hidden = ms.get("hidden", {})
        attn = ms.get("attention", {})
        rec = {
            "sample_id": r["sample_id"], "task": r.get("task"),
            "task_family": r.get("task_family"),
            "target_context_length": r.get("target_context_length"),
            "actual_input_tokens": r.get("actual_input_tokens"),
            "correct": r.get("correct"),
            "match_fraction": r.get("match_fraction"),
            "parse_status": r.get("parse_status"),
            "evidence_rel_position": r.get("evidence_rel_position"),
            "attention_collected": ms.get("attention_collected"),
            "n_skips": len(r.get("skips") or []),
        }
        # hidden features
        h_effs, h_tn = {}, {}
        for li, m in hidden.items():
            if "r_eff" in m:
                h_effs[li] = m["r_eff"]
                h_tn[li] = m.get("r_eff_tokennorm")
                layer_recs.append({
                    "sample_id": r["sample_id"], "task_family": r.get("task_family"),
                    "correct": r.get("correct"), "layer": int(li),
                    "hidden_r_eff": m["r_eff"],
                    "hidden_r_eff_tokennorm": m.get("r_eff_tokennorm"),
                    "hidden_stable_rank": m.get("stable_rank"),
                })
        mid = _mid_layer(list(h_effs))
        rec["hidden_r_eff_mid"] = h_effs.get(mid)
        rec["hidden_r_eff_tokennorm_mid"] = h_tn.get(mid)
        rec["hidden_r_eff_mean"] = float(np.mean(list(h_effs.values()))) if h_effs else None
        # attention features (mean over layers & heads)
        ents, sranks, breffs, mmass = [], [], [], []
        for li, per_head in attn.items():
            l_ent, l_br = [], []
            for h, m in per_head.items():
                if "entropy" in m:
                    ents.append(m["entropy"]); l_ent.append(m["entropy"])
                    sranks.append(m["stable_rank"])
                    breffs.append(m["block_r_eff"]); l_br.append(m["block_r_eff"])
                    mmass.append(m["max_mass"])
            if l_ent:
                layer_recs.append({
                    "sample_id": r["sample_id"], "task_family": r.get("task_family"),
                    "correct": r.get("correct"), "layer": int(li),
                    "attn_entropy": float(np.mean(l_ent)),
                    "attn_block_r_eff": float(np.mean(l_br)),
                })
        rec["attn_entropy_mean"] = float(np.mean(ents)) if ents else None
        rec["attn_stable_rank_mean"] = float(np.mean(sranks)) if sranks else None
        rec["attn_block_r_eff_mean"] = float(np.mean(breffs)) if breffs else None
        rec["attn_max_mass_mean"] = float(np.mean(mmass)) if mmass else None
        # evidence survival at k=32, mid layer
        es = ms.get("evidence_survival", {})
        if es:
            li = _mid_layer([k for k, v in es.items() if "ks" in v])
            if li and "ks" in es.get(li, {}):
                ks = es[li]["ks"]
                k_idx = min(range(len(ks)), key=lambda i: abs(ks[i] - 32))
                rec["evidence_survival_k32_mid"] = es[li]["evidence_survival"][k_idx]
                if es[li].get("random_survival"):
                    rec["evidence_survival_gap_k32_mid"] = (
                        es[li]["evidence_survival"][k_idx]
                        - es[li]["random_survival"][k_idx])
        recs.append(rec)
    return pd.DataFrame(recs), pd.DataFrame(layer_recs)


def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    scored = df[df.correct.notna()].copy()
    scored["correct"] = scored["correct"].astype(float)
    out["accuracy"] = scored.pivot_table(
        index="task_family", columns="target_context_length",
        values="correct", aggfunc="mean")
    agg = {"hidden_r_eff_mid": "mean", "hidden_r_eff_tokennorm_mid": "mean",
           "attn_block_r_eff_mean": "mean",
           "attn_entropy_mean": "mean", "attn_stable_rank_mean": "mean"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out["ranks_by_family"] = df.groupby("task_family").agg(agg).round(2)
    out["parse_status"] = df.parse_status.value_counts().to_frame("count")
    return out


def run_regressions(df: pd.DataFrame) -> dict:
    """Logistic ablations: base(len+family) -> +entropy -> +entropy+rank.
    In-sample AUC (N too small for CV at smoke scale; noted in report)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=["correct", "attn_entropy_mean",
                          "hidden_r_eff_mid", "attn_block_r_eff_mean"]).copy()
    d["correct"] = d["correct"].astype(int)
    n = len(d)
    res: dict = {"n": n}
    if n < 20:
        res["skip_reason"] = f"only {n} usable samples; need >=20 for regression"
        return res
    y = d["correct"].values
    if y.min() == y.max():
        res["skip_reason"] = "degenerate outcome (all correct or all incorrect)"
        return res

    fam = pd.get_dummies(d["task_family"], prefix="fam", drop_first=True)
    cont_all = {
        "context_length": np.log2(d["target_context_length"].values),
        "entropy": d["attn_entropy_mean"].values,
        "hidden_rank": d["hidden_r_eff_mid"].values,
        "hidden_rank_tokennorm": d["hidden_r_eff_tokennorm_mid"].values
        if d["hidden_r_eff_tokennorm_mid"].notna().all() else None,
        "attention_rank": d["attn_block_r_eff_mean"].values,
    }
    cont_all = {k: v for k, v in cont_all.items() if v is not None}
    rank_cols = [c for c in ("hidden_rank", "hidden_rank_tokennorm",
                             "attention_rank") if c in cont_all]
    models = {
        "base: length+family": ["context_length"],
        "+entropy": ["context_length", "entropy"],
        "+entropy+rank": ["context_length", "entropy"] + rank_cols,
    }
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    res["models"] = {}
    do_cv = n >= 60 and min((y == 0).sum(), (y == 1).sum()) >= 10
    for name, cols in models.items():
        X_cont = np.column_stack([cont_all[c] for c in cols])
        X_scaled = StandardScaler().fit_transform(X_cont)
        X = np.column_stack([X_scaled, fam.values.astype(float)])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, y)
        prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, prob)
        coef_names = cols + list(fam.columns)
        entry = {
            "auc_in_sample": round(float(auc), 4),
            "coefficients": {c: round(float(w), 4)
                             for c, w in zip(coef_names, clf.coef_[0])},
        }
        if do_cv:
            # scaler inside the pipeline -> no leakage across folds
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0))
            Xr = np.column_stack([X_cont, fam.values.astype(float)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            prob_cv = cross_val_predict(pipe, Xr, y, cv=cv,
                                        method="predict_proba")[:, 1]
            entry["auc_cv5"] = round(float(roc_auc_score(y, prob_cv)), 4)
            # simple bootstrap CI on CV AUC
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(500):
                idx = rng.integers(0, n, n)
                if y[idx].min() != y[idx].max():
                    boots.append(roc_auc_score(y[idx], prob_cv[idx]))
            if boots:
                entry["auc_cv5_ci95"] = [round(float(np.percentile(boots, q)), 4)
                                         for q in (2.5, 97.5)]
        res["models"][name] = entry
    key = "auc_cv5" if do_cv else "auc_in_sample"
    aucs = {k: v[key] for k, v in res["models"].items()}
    res["auc_metric_used"] = key
    res["delta_auc_rank_beyond_entropy"] = round(
        aucs["+entropy+rank"] - aucs["+entropy"], 4)
    return res


def analyze_ntk(results_dir: Path, df: pd.DataFrame, fig_dir: Path) -> dict:
    """Join eNTK per-prompt features with correctness; Gram structure by family.
    Returns summary dict (empty if no NTK outputs present)."""
    ntk_path = results_dir / "ntk_features.jsonl"
    npz_path = results_dir / "ntk_sketches.npz"
    if not ntk_path.exists():
        return {}
    from rank_demand.ntk_features import gram_from_sketches

    ntk_rows = read_jsonl(ntk_path)
    nd = pd.DataFrame([r for r in ntk_rows if "grad_norm" in r])
    out: dict = {"n_ntk": len(nd),
                 "n_skipped": len(ntk_rows) - len(nd)}
    if nd.empty:
        out["skip_reason"] = "all NTK samples skipped"
        return out
    merged = nd.merge(df[["sample_id", "correct"]], on="sample_id", how="left")
    out["self_kernel_by_family"] = (
        merged.groupby("task_family")["self_kernel"].agg(["mean", "std"])
        .round(4).to_dict("index"))
    out["answer_ce_by_family"] = (
        merged.groupby("task_family")["answer_ce"].mean().round(4).to_dict())
    sc = merged.dropna(subset=["correct"])
    if len(sc) and sc.correct.nunique() > 1:
        out["grad_norm_correct"] = round(
            float(sc[sc.correct == True]["grad_norm"].mean()), 4)   # noqa: E712
        out["grad_norm_incorrect"] = round(
            float(sc[sc.correct == False]["grad_norm"].mean()), 4)  # noqa: E712
    if npz_path.exists():
        z = np.load(npz_path)
        ids = [r["sample_id"] for r in ntk_rows if r["sample_id"] in z.files]
        fam = {r["sample_id"]: r["task_family"] for r in ntk_rows}
        order = sorted(ids, key=lambda i: fam[i])
        if len(order) >= 4:
            S = np.stack([z[i] for i in order])
            g = gram_from_sketches(S)
            out["gram_effective_rank"] = round(g["effective_rank"], 2)
            out["gram_top1_share"] = round(g["top1_share"], 4)
            # within- vs across-family mean cosine
            d = np.sqrt(np.clip(np.diag(g["gram"]), 1e-30, None))
            C = g["gram"] / np.outer(d, d)
            fams = np.array([fam[i] for i in order])
            same = fams[:, None] == fams[None, :]
            off = ~np.eye(len(order), dtype=bool)
            out["mean_cos_within_family"] = round(float(C[same & off].mean()), 4)
            out["mean_cos_across_family"] = round(float(C[~same].mean()), 4)
            try:
                from rank_demand import plots
                plots.ntk_gram_heatmap(g["gram"], list(fams),
                                       fig_dir / "ntk_gram.png")
                out["gram_plot"] = "figures/ntk_gram.png"
            except Exception as e:
                logger.warning("ntk gram plot failed: %s", e)
    return out


def analyze_truncation(rows: list[dict], fig_dir: Path) -> dict:
    """Offline low-rank intervention proxy: ||AV - A_kV||/||AV|| by family/k,
    and its association with correctness. Returns {} if no truncation data."""
    recs = []
    for r in rows:
        tr = (r.get("metrics_summary") or {}).get("truncation") or {}
        for li, per_head in tr.items():
            for h, m in per_head.items():
                if "ks" in m:
                    for k, e in zip(m["ks"], m["rel_error"]):
                        recs.append({"sample_id": r["sample_id"],
                                     "task_family": r["task_family"],
                                     "correct": r.get("correct"),
                                     "layer": int(li), "head": int(h),
                                     "k": int(k), "rel_error": float(e)})
    if not recs:
        return {}
    td = pd.DataFrame(recs)
    out: dict = {"n_samples": td.sample_id.nunique()}
    out["rel_error_by_family_k"] = (
        td.groupby(["task_family", "k"])["rel_error"].mean().round(4)
        .unstack("k").to_dict("index"))
    # per-sample scalar: mean rel_error at k=16 across layers/heads
    s16 = (td[td.k == 16].groupby("sample_id")
           .agg(rel_error_k16=("rel_error", "mean"),
                correct=("correct", "first"),
                task_family=("task_family", "first")).reset_index())
    sc = s16.dropna(subset=["correct"])
    if len(sc) and sc.correct.nunique() > 1:
        a = sc[sc.correct == True]["rel_error_k16"]   # noqa: E712
        b = sc[sc.correct == False]["rel_error_k16"]  # noqa: E712
        out["rel_error_k16_correct"] = round(float(a.mean()), 4)
        out["rel_error_k16_incorrect"] = round(float(b.mean()), 4)
        try:
            from scipy.stats import mannwhitneyu
            out["mannwhitney_p"] = round(float(
                mannwhitneyu(a, b, alternative="two-sided").pvalue), 4)
        except Exception:
            pass
    else:
        out["correctness_split"] = "degenerate (all correct or all wrong)"
    # plot: error vs k per family (mid-depth layers only for readability)
    try:
        from rank_demand.plots import FAMILY_ORDER, _fam_color, _save
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for fam in FAMILY_ORDER:
            sub = td[td.task_family == fam]
            if sub.empty:
                continue
            g = sub.groupby("k")["rel_error"].mean()
            ax.plot(g.index, g.values, "-o", lw=2, ms=5,
                    color=_fam_color(fam), label=fam)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (rank of truncated attention)")
        ax.set_ylabel(r"$\|AV - A_k V\| / \|AV\|$")
        ax.set_title("Attention low-rank reconstruction error by task family")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, fig_dir / "truncation_error.png")
        out["plot"] = "figures/truncation_error.png"
    except Exception as e:
        logger.warning("truncation plot failed: %s", e)
    return out


def survival_curves_for_plot(rows: list[dict]) -> dict:
    """Average evidence-survival curves across samples per layer."""
    acc: dict[str, dict] = {}
    for r in rows:
        es = (r.get("metrics_summary") or {}).get("evidence_survival", {})
        for li, c in es.items():
            if "ks" not in c:
                continue
            a = acc.setdefault(li, {"ks": c["ks"], "ev": [], "rand": []})
            if c["ks"] == a["ks"]:
                a["ev"].append(c["evidence_survival"])
                if c.get("random_survival"):
                    a["rand"].append(c["random_survival"])
    out = {}
    for li, a in acc.items():
        if not a["ev"]:
            continue
        out[li] = {
            "ks": a["ks"],
            "ev_mean": np.mean(a["ev"], axis=0).tolist(),
            "rand_mean": (np.mean(a["rand"], axis=0).tolist()
                          if a["rand"] else None),
            "n": len(a["ev"]),
        }
    return out


def write_report(results_dir: Path, report_dir: Path, df, df_layer, tables,
                 reg, fig_paths, rows, ntk: dict | None = None,
                 trunc: dict | None = None):
    meta = {}
    meta_p = results_dir / "run_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
    n_oom = sum(1 for r in rows for s in (r.get("skips") or [])
                if "oom" in str(s.get("skip_reason", "")).lower())
    n_err = sum(1 for r in rows
                if str(r.get("parse_status", "")).startswith("error"))

    # verdicts
    verdict = []
    if reg.get("models"):
        d_auc = reg["delta_auc_rank_beyond_entropy"]
        verdict.append(f"A. Rank vs entropy: adding rank on top of entropy changes "
                       f"in-sample AUC by {d_auc:+.3f} "
                       f"({'rank adds signal' if d_auc > 0.02 else 'rank adds little beyond entropy at this N'}).")
    else:
        verdict.append(f"A. Regression skipped: {reg.get('skip_reason')} — "
                       "rank-vs-entropy question needs the main run.")
    if "ranks_by_family" in tables and not tables["ranks_by_family"].empty:
        t = tables["ranks_by_family"]
        if "hidden_r_eff_mid" in t and t["hidden_r_eff_mid"].notna().any():
            verdict.append(f"B. Highest hidden effective rank (mid layer): "
                           f"**{t['hidden_r_eff_mid'].idxmax()}** "
                           f"({t['hidden_r_eff_mid'].max():.1f}); lowest: "
                           f"{t['hidden_r_eff_mid'].idxmin()} "
                           f"({t['hidden_r_eff_mid'].min():.1f}).")
    sc = df.dropna(subset=["correct"])
    if len(sc) and sc.correct.nunique() > 1 and sc["hidden_r_eff_mid"].notna().any():
        m_ok = sc[sc.correct == True]["hidden_r_eff_mid"].mean()   # noqa: E712
        m_bad = sc[sc.correct == False]["hidden_r_eff_mid"].mean()  # noqa: E712
        verdict.append(f"C. Hidden r_eff (mid): correct={m_ok:.1f} vs "
                       f"incorrect={m_bad:.1f} (descriptive; N small).")
    else:
        verdict.append("C. Correct/incorrect rank split not computable "
                       "(degenerate correctness or missing metrics).")
    if "evidence_survival_gap_k32_mid" in df and df["evidence_survival_gap_k32_mid"].notna().any():
        g = df["evidence_survival_gap_k32_mid"].mean()
        verdict.append(f"D. Evidence survival gap (evidence - random, k=32, mid "
                       f"layer): {g:+.3f} on average "
                       f"({'evidence tokens preferentially survive' if g > 0 else 'no preferential survival'}).")
    else:
        verdict.append("D. No evidence-survival data (no recoverable evidence positions).")

    lines = [
        "# Rank Demand — run report\n",
        f"- command: `{meta.get('command', 'n/a')}`",
        f"- model: `{meta.get('model_id', 'n/a')}`",
        f"- mode: {meta.get('mode')} | tasks: {list((meta.get('tasks') or {}).keys())} "
        f"| lengths: {meta.get('lengths')}",
        f"- samples: {len(rows)} rows "
        f"({meta.get('num_samples_per_task_length')} per task x length)",
        f"- hardware: {json.dumps(meta.get('gpu', {}))}",
        f"- failures: {n_err} hard errors, {n_oom} OOM-skipped metric blocks, "
        f"parse statuses: {tables['parse_status'].to_dict()['count']}",
        "",
        "## Accuracy by task family x context length\n",
        tables["accuracy"].to_markdown() if not tables["accuracy"].empty else "_no scored samples_",
        "",
        "## Mean rank/entropy metrics by task family\n",
        tables["ranks_by_family"].to_markdown(),
        "",
        "## Regression (correct ~ predictors, standardized, one-hot family)\n",
        "```json",
        json.dumps(reg, indent=2),
        "```",
        "_AUC is in-sample; treat as descriptive until the main run has enough N for CV._",
        "",
        "## Plots\n",
    ]
    for p in fig_paths:
        lines.append(f"![{p.stem}]({p.relative_to(report_dir)})")
    if ntk:
        lines += ["", "## Empirical NTK per-prompt features\n",
                  "```json", json.dumps(
                      {k: v for k, v in ntk.items() if k != "gram_plot"},
                      indent=2, default=str), "```", ""]
    if trunc:
        lines += ["", "## Low-rank attention truncation (offline diagnostic)\n",
                  "```json", json.dumps(
                      {k: v for k, v in trunc.items() if k != "plot"},
                      indent=2, default=str), "```", ""]
    lines += ["", "## Preliminary verdict\n"]
    lines += [f"- {v}" for v in verdict]
    lines += ["- E. Continuation call: see console summary / next-command printout.", ""]

    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "summary.md"
    out.write_text("\n".join(lines))
    logger.info("report -> %s", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run output dir with results.jsonl")
    ap.add_argument("--report", default=str(REPO_ROOT / "reports"))
    args = ap.parse_args()
    setup_logging()

    results_dir = Path(args.results)
    report_dir = Path(args.report)
    rows = read_jsonl(results_dir / "results.jsonl")
    df, df_layer = extract_features(rows)
    logger.info("loaded %d rows (%d with metrics)", len(df),
                df["hidden_r_eff_mid"].notna().sum())

    tables = summary_tables(df)
    for name, t in tables.items():
        print(f"\n=== {name} ===\n{t}")

    reg = run_regressions(df)
    print(f"\n=== regression ===\n{json.dumps(reg, indent=2)}")

    fig_dir = report_dir / "figures"
    figs = []

    def _try(fn, *a, **kw):
        try:
            fn(*a, **kw)
            path = kw.get("out") or next(
                x for x in reversed(a) if isinstance(x, Path))
            figs.append(path)
        except Exception as e:
            logger.warning("plot %s failed: %s", fn.__name__, e)

    scored = df.dropna(subset=["correct"])
    _try(plots.accuracy_vs_length, scored, fig_dir / "accuracy_vs_length.png")
    if not df_layer.empty and "hidden_r_eff" in df_layer:
        dl = df_layer.dropna(subset=["hidden_r_eff"])
        _try(plots.rank_vs_layer, dl, "hidden_r_eff", "correct",
             "Hidden effective rank vs layer (correct vs incorrect)",
             "hidden r_eff", fig_dir / "hidden_rank_vs_layer.png",
             split_colors={True: "#0ca30c", False: "#d03b3b"})
    if not df_layer.empty and "attn_block_r_eff" in df_layer:
        da = df_layer.dropna(subset=["attn_block_r_eff"])
        _try(plots.rank_vs_layer, da, "attn_block_r_eff", "task_family",
             "Attention block effective rank vs layer by task family",
             "block r_eff", fig_dir / "attn_rank_vs_layer.png")
    if df["attn_entropy_mean"].notna().any():
        _try(plots.entropy_vs_rank_scatter,
             df.dropna(subset=["attn_entropy_mean", "attn_block_r_eff_mean"]),
             fig_dir / "entropy_vs_rank.png")
        _try(plots.correctness_vs_rank_scatter,
             scored.dropna(subset=["attn_block_r_eff_mean"]),
             "attn_block_r_eff_mean", fig_dir / "correct_vs_rank.png",
             "attention block effective rank (mean)")
    curves = survival_curves_for_plot(rows)
    if curves:
        _try(plots.evidence_survival_curves, curves,
             fig_dir / "evidence_survival.png")

    ntk = analyze_ntk(results_dir, df, fig_dir)
    if ntk:
        print(f"\n=== eNTK features ===\n{json.dumps(ntk, indent=2, default=str)}")
        if ntk.get("gram_plot"):
            figs.append(fig_dir / "ntk_gram.png")

    trunc = analyze_truncation(rows, fig_dir)
    if trunc:
        print(f"\n=== truncation ===\n{json.dumps(trunc, indent=2, default=str)}")
        if trunc.get("plot"):
            figs.append(fig_dir / "truncation_error.png")

    out = write_report(results_dir, report_dir, df, df_layer, tables, reg,
                       figs, rows, ntk, trunc)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
