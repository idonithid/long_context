#!/usr/bin/env python
"""Method experiment 2: sketched-eNTK capability fingerprint -> fine-tuning
interference prediction.

Protocol (families = capabilities):
 1. Fresh TRAIN samples per family (seed 999, disjoint from eval).
 2. PRE: gold-answer CE + accuracy on an eval set per family; eNTK sketches
    for train and eval samples (same CountSketch seed -> inner products
    approximate true gradient kernels).
 3. For each family i: fine-tune ONLY the NTK parameter subset (q/v of the
    selected layers — the same parameters the sketches are computed on, so
    first-order NTK theory applies exactly), then re-measure CE/accuracy on
    every family j. Restore weights.
 4. Prediction target: measured dCE[i,j]. Predictor: mean train-i x eval-j
    sketch kernel K[i,j] (first-order GD step: dLoss_j ~ -eta * K[i,j]).

Output: results/interference/{measured_dce.json, predicted_kernel.json,
train/eval sketches, per-family logs}.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand.config import load_config  # noqa: E402
from rank_demand.eval_ruler import score_prediction  # noqa: E402
from rank_demand.generation import build_prompt, generate_answer  # noqa: E402
from rank_demand.hooks import select_layers  # noqa: E402
from rank_demand.model_loader import load_model_and_tokenizer  # noqa: E402
from rank_demand.ntk_features import NTKExtractor, select_ntk_params  # noqa: E402
from rank_demand.utils import (print_gpu_info, read_jsonl, set_seed,  # noqa: E402
                               setup_logging)


def encode(tokenizer, s, device):
    prompt = build_prompt(tokenizer, s["prompt_text"], s.get("answer_prefix", ""))
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    ans_text = " " + ", ".join(map(str, s["expected_answer"]))
    ans = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    return ids, ans


@torch.no_grad()
def eval_ce_acc(model, tokenizer, samples, device, gen_max_new=48):
    """Gold-answer teacher-forced CE + greedy-decoding accuracy per sample."""
    ces, accs = [], []
    for s in samples:
        ids, ans = encode(tokenizer, s, device)
        full = torch.cat([ids, ans], dim=1)
        labels = full.clone()
        labels[:, : ids.shape[1]] = -100
        ce = float(model(input_ids=full, labels=labels, use_cache=False).loss)
        ces.append(ce)
        g = generate_answer(model, tokenizer, ids, max_new_tokens=gen_max_new)
        sc = score_prediction(s["ruler_task"], g["prediction_text"],
                              s["expected_answer"])
        accs.append(float(sc["correct"]))
    return float(np.mean(ces)), float(np.mean(accs))


def finetune_subset(model, tokenizer, params, samples, device,
                    lr=2e-5, epochs=3, accum=8):
    """Adam on the NTK parameter subset only; answer-token CE."""
    opt = torch.optim.Adam([p for p in params.values()], lr=lr)
    model.train()
    step = 0
    for ep in range(epochs):
        for bi, s in enumerate(samples):
            ids, ans = encode(tokenizer, s, device)
            full = torch.cat([ids, ans], dim=1)
            labels = full.clone()
            labels[:, : ids.shape[1]] = -100
            loss = model(input_ids=full, labels=labels, use_cache=False).loss
            (loss / accum).backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
    opt.zero_grad(set_to_none=True)
    model.eval()
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/ruler_qwen2p5_7b.yaml"))
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_data_dir", default="results/data_ft")
    ap.add_argument("--output_dir", default="results/interference")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    print_gpu_info()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    families = cfg["tasks"]  # task -> family
    eval_dir = REPO_ROOT / cfg["data_dir"] / "main"
    train_dir = REPO_ROOT / args.train_data_dir / "main"
    eval_sets, train_sets = {}, {}
    for task in families:
        eval_sets[task] = read_jsonl(eval_dir / f"{task}_{args.length}.jsonl")[: args.n_eval]
        train_sets[task] = read_jsonl(train_dir / f"{task}_{args.length}.jsonl")[: args.n_train]
        assert train_sets[task], f"no train data for {task}"

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"], dtype=cfg["dtype"],
        attn_implementation=cfg["attn_implementation_generate"])
    device = next(model.parameters()).device
    layers = select_layers(model.config.num_hidden_layers,
                           cfg["metrics"]["layer_fractions"])

    # --- sketches for train + eval samples (same sketch seed) ---
    sk_path = out_dir / "sketches.npz"
    if sk_path.exists():
        z = np.load(sk_path)
        sketches = {k: z[k] for k in z.files}
    else:
        sketches = {}
        with NTKExtractor(model, layers, seed=cfg["seed"]) as ntk:
            for task in families:
                for tag, samples in [("train", train_sets[task]),
                                     ("eval", eval_sets[task])]:
                    for s in tqdm(samples, desc=f"sketch {task} {tag}"):
                        ids, ans = encode(tokenizer, s, device)
                        f = ntk.features(ids, ans)
                        sketches[f"{tag}__{s['sample_id']}"] = f["sketch"]
        np.savez_compressed(sk_path, **sketches)
    print(f"sketches: {len(sketches)} -> {sk_path}")

    params = select_ntk_params(model, layers)

    # --- PRE eval ---
    pre_path = out_dir / "pre_eval.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
    else:
        pre = {}
        for task in families:
            ce, acc = eval_ce_acc(model, tokenizer, eval_sets[task], device)
            pre[task] = {"ce": ce, "acc": acc}
            print(f"PRE {task}: ce={ce:.4f} acc={acc:.2f}")
        pre_path.write_text(json.dumps(pre, indent=2))

    # --- interference loop ---
    measured = {}
    meas_path = out_dir / "measured.json"
    if meas_path.exists():
        measured = json.loads(meas_path.read_text())
    snapshot = {n: p.detach().clone() for n, p in params.items()}
    for train_task in families:
        if train_task in measured:
            continue
        t0 = time.time()
        steps = finetune_subset(model, tokenizer, params,
                                train_sets[train_task], device,
                                lr=args.lr, epochs=args.epochs)
        post = {}
        for task in families:
            ce, acc = eval_ce_acc(model, tokenizer, eval_sets[task], device)
            post[task] = {"dce": round(ce - pre[task]["ce"], 5),
                          "dacc": round(acc - pre[task]["acc"], 4),
                          "ce": ce, "acc": acc}
        measured[train_task] = {"steps": steps, "post": post,
                                "minutes": round((time.time() - t0) / 60, 1)}
        # restore
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(snapshot[n])
        meas_path.write_text(json.dumps(measured, indent=2))
        print(f"train on {train_task}: "
              + " ".join(f"{t}:dce={post[t]['dce']:+.3f}" for t in families))

    # --- predicted kernel matrix ---
    K = {}
    for ti in families:
        Si = np.stack([sketches[f"train__{s['sample_id']}"] for s in train_sets[ti]])
        for tj in families:
            Sj = np.stack([sketches[f"eval__{s['sample_id']}"] for s in eval_sets[tj]])
            # mean kernel and mean cosine
            G = Si @ Sj.T
            ni = np.linalg.norm(Si, axis=1, keepdims=True)
            nj = np.linalg.norm(Sj, axis=1, keepdims=True)
            K[f"{ti}->{tj}"] = {"mean_kernel": float(G.mean()),
                                "mean_cos": float((G / (ni @ nj.T + 1e-30)).mean())}
    (out_dir / "predicted_kernel.json").write_text(json.dumps(K, indent=2))

    # --- correlation ---
    from scipy.stats import pearsonr, spearmanr
    pred, meas, pred_cos = [], [], []
    for ti in families:
        for tj in families:
            pred.append(-K[f"{ti}->{tj}"]["mean_kernel"])  # -K: higher kernel -> CE drop
            pred_cos.append(-K[f"{ti}->{tj}"]["mean_cos"])
            meas.append(measured[ti]["post"][tj]["dce"])
    summary = {
        "spearman_kernel_vs_dce": [round(x, 4) for x in spearmanr(pred, meas)],
        "pearson_kernel_vs_dce": [round(x, 4) for x in pearsonr(pred, meas)],
        "spearman_cos_vs_dce": [round(x, 4) for x in spearmanr(pred_cos, meas)],
        "off_diag_spearman": None,
    }
    od_p = [p for (p, ti, tj) in zip(pred, *zip(*[(i, j) for i in families for j in families]))
            if ti != tj]
    od_m = [m for (m, ti, tj) in zip(meas, *zip(*[(i, j) for i in families for j in families]))
            if ti != tj]
    if od_p:
        summary["off_diag_spearman"] = [round(x, 4) for x in spearmanr(od_p, od_m)]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
