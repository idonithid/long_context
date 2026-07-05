#!/usr/bin/env python
"""Rehearsal targeting: does the sketch kernel tell you WHICH data to rehearse?

Protocol (hetero capabilities, subset trainer, CE outcome):
  For each attacker capability A:
    rehearsal pool = train samples of the other capabilities (their sketches
    already exist from the hetero interference run).
    predicted damage of pool sample s' = mean_{s in A_train} <sk(s), sk(s')>
    (first-order: dCE(s') ~ -eta * kernel, so most-negative kernel = most hurt)
    Conditions, same budget:
      none    fine-tune on A_train only
      random  A_train + k random pool samples
      kernel  A_train + k most-negatively-predicted pool samples
    Measure per-sample CE on every capability's eval set before/after.

  Success criterion: victim (non-A) mean dCE  kernel < random < none.
  Side result: per-sample Spearman between predicted kernel and measured
  per-eval-sample dCE under the `none` condition (finer grain than the
  capability-level matrices).

Reuses sketches from results/interference_hetero/sketches.npz (same data,
seed, and CountSketch projection).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_interference import encode, finetune_subset  # noqa: E402

from rank_demand.config import load_config  # noqa: E402
from rank_demand.hooks import select_layers  # noqa: E402
from rank_demand.model_loader import load_model_and_tokenizer  # noqa: E402
from rank_demand.ntk_features import select_ntk_params  # noqa: E402
from rank_demand.utils import (print_gpu_info, read_jsonl, set_seed,  # noqa: E402
                               setup_logging)

CAPS = ["gsm8k", "mbpp_code", "qa_1", "niah_single_1", "cwe"]


@torch.no_grad()
def eval_ce_per_sample(model, tokenizer, samples, device):
    out = {}
    for s in samples:
        ids, ans = encode(tokenizer, s, device)
        full = torch.cat([ids, ans], dim=1)
        labels = full.clone()
        labels[:, : ids.shape[1]] = -100
        out[s["sample_id"]] = float(
            model(input_ids=full, labels=labels, use_cache=False).loss)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/ruler_qwen2p5_7b.yaml"))
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--k_rehearse", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--data_dir", default="results/data_hetero")
    ap.add_argument("--sketches", default="results/interference_hetero/sketches.npz")
    ap.add_argument("--output_dir", default="results/rehearsal")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    print_gpu_info()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ddir = REPO_ROOT / args.data_dir
    train_sets, eval_sets = {}, {}
    for cap in CAPS:
        train_sets[cap] = read_jsonl(ddir / "train" / "main" / f"{cap}_{args.length}.jsonl")[: args.n_train]
        eval_sets[cap] = read_jsonl(ddir / "eval" / "main" / f"{cap}_{args.length}.jsonl")[: args.n_eval]

    z = np.load(REPO_ROOT / args.sketches)
    sk = {k: z[k] for k in z.files}

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"], dtype=cfg["dtype"],
        attn_implementation=cfg["attn_implementation_generate"])
    device = next(model.parameters()).device
    layers = select_layers(model.config.num_hidden_layers,
                           cfg["metrics"]["layer_fractions"])
    params = select_ntk_params(model, layers)
    snapshot = {n: p.detach().clone() for n, p in params.items()}

    # --- PRE per-sample CE ---
    pre_path = out_dir / "pre_per_sample.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
    else:
        pre = {cap: eval_ce_per_sample(model, tokenizer, eval_sets[cap], device)
               for cap in CAPS}
        pre_path.write_text(json.dumps(pre, indent=2))
    print("PRE ce:", {c: round(float(np.mean(list(v.values()))), 3)
                      for c, v in pre.items()})

    meas_path = out_dir / "measured_rehearsal.json"
    measured = json.loads(meas_path.read_text()) if meas_path.exists() else {}

    for A in CAPS:
        A_sk = np.stack([sk[f"train__{s['sample_id']}"] for s in train_sets[A]])
        pool = [(cap, s) for cap in CAPS if cap != A for s in train_sets[cap]]
        # predicted per-pool-sample kernel with the attacker train set
        pred_pool = {}
        for cap, s in pool:
            v = sk[f"train__{s['sample_id']}"]
            pred_pool[s["sample_id"]] = float((A_sk @ v).mean())
        ranked = sorted(pool, key=lambda cs: pred_pool[cs[1]["sample_id"]])
        sel_kernel = [s for _, s in ranked[: args.k_rehearse]]  # most negative
        rng = random.Random(0)
        sel_random = [s for _, s in rng.sample(pool, args.k_rehearse)]

        # per-eval-sample predicted kernel (for the fine-grained correlation)
        pred_eval = {}
        for cap in CAPS:
            for s in eval_sets[cap]:
                v = sk[f"eval__{s['sample_id']}"]
                pred_eval[s["sample_id"]] = float((A_sk @ v).mean())

        measured.setdefault(A, {"pred_eval": pred_eval,
                                "rehearsed_kernel": [s["sample_id"] for s in sel_kernel],
                                "rehearsed_random": [s["sample_id"] for s in sel_random]})
        for cond, extra in [("none", []), ("random", sel_random),
                            ("kernel", sel_kernel)]:
            if cond in measured[A].get("post", {}):
                continue
            t0 = time.time()
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(snapshot[n])
            mix = list(train_sets[A]) + list(extra)
            random.Random(0).shuffle(mix)
            steps = finetune_subset(model, tokenizer, params, mix, device,
                                    lr=args.lr, epochs=args.epochs)
            post = {cap: eval_ce_per_sample(model, tokenizer, eval_sets[cap],
                                            device) for cap in CAPS}
            dce = {cap: {sid: round(post[cap][sid] - pre[cap][sid], 5)
                         for sid in post[cap]} for cap in CAPS}
            victim = float(np.mean([np.mean(list(dce[c].values()))
                                    for c in CAPS if c != A]))
            measured[A].setdefault("post", {})[cond] = {
                "steps": steps, "minutes": round((time.time() - t0) / 60, 1),
                "victim_mean_dce": round(victim, 5),
                "self_mean_dce": round(float(np.mean(list(dce[A].values()))), 5),
                "dce_per_cap": {c: round(float(np.mean(list(dce[c].values()))), 5)
                                for c in CAPS},
                "dce_per_sample": dce,
            }
            meas_path.write_text(json.dumps(measured, indent=2))
            print(f"A={A} cond={cond}: victim dce={victim:+.4f} "
                  f"self dce={measured[A]['post'][cond]['self_mean_dce']:+.4f}")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(snapshot[n])

    # --- summary ---
    from scipy.stats import spearmanr, wilcoxon
    rows = {c: {} for c in ["none", "random", "kernel"]}
    per_sample_rho = {}
    for A in CAPS:
        for cond in rows:
            rows[cond][A] = measured[A]["post"][cond]["victim_mean_dce"]
        # per-sample prediction check under `none`
        pk, pm = [], []
        for cap in CAPS:
            if cap == A:
                continue
            d = measured[A]["post"]["none"]["dce_per_sample"][cap]
            for sid, v in d.items():
                pk.append(-measured[A]["pred_eval"][sid])
                pm.append(v)
        rho, p = spearmanr(pk, pm)
        per_sample_rho[A] = [round(float(rho), 4), round(float(p), 6)]

    none_v = [rows["none"][A] for A in CAPS]
    rand_v = [rows["random"][A] for A in CAPS]
    kern_v = [rows["kernel"][A] for A in CAPS]
    summary = {
        "victim_mean_dce_by_condition": {c: {A: rows[c][A] for A in CAPS}
                                         for c in rows},
        "mean_over_attackers": {c: round(float(np.mean(list(rows[c].values()))), 5)
                                for c in rows},
        "protection_kernel_vs_none": round(float(np.mean(none_v) - np.mean(kern_v)), 5),
        "protection_random_vs_none": round(float(np.mean(none_v) - np.mean(rand_v)), 5),
        "wilcoxon_kernel_vs_random_pvalue": round(float(
            wilcoxon(kern_v, rand_v).pvalue), 4) if len(CAPS) >= 5 else None,
        "per_sample_spearman_none": per_sample_rho,
        "k_rehearse": args.k_rehearse, "epochs": args.epochs, "lr": args.lr,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
