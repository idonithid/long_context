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
def eval_ce_acc(model, tokenizer, samples, device, gen_max_new=48,
                ce_only=False):
    """Gold-answer teacher-forced CE (+ greedy accuracy unless ce_only)."""
    ces, accs = [], []
    for s in samples:
        ids, ans = encode(tokenizer, s, device)
        full = torch.cat([ids, ans], dim=1)
        labels = full.clone()
        labels[:, : ids.shape[1]] = -100
        ce = float(model(input_ids=full, labels=labels, use_cache=False).loss)
        ces.append(ce)
        if not ce_only:
            g = generate_answer(model, tokenizer, ids, max_new_tokens=gen_max_new)
            sc = score_prediction(s["ruler_task"], g["prediction_text"],
                                  s["expected_answer"])
            accs.append(float(sc["correct"]))
    return float(np.mean(ces)), (float(np.mean(accs)) if accs else None)


LOSS_SCALE = 1024.0


def finetune_subset(model, tokenizer, params, samples, device,
                    lr=2e-5, epochs=3, accum=8):
    """Adam on fp32 MASTER copies of the NTK parameter subset.

    Training fp16 weights directly (fp16 grads, fp16 Adam states, no loss
    scaling) destroys the model within a few steps — grads underflow/overflow
    and the update noise floor exceeds lr*grad. Standard mixed-precision
    recipe instead: scaled backward, fp32 master update, copy back to fp16.
    """
    masters = {n: p.detach().float().clone().requires_grad_(False)
               for n, p in params.items()}
    opt = torch.optim.Adam(list(masters.values()), lr=lr)
    model.train()
    step, skipped = 0, 0

    def apply_update():
        nonlocal step, skipped
        grads = {}
        finite = True
        for n, p in params.items():
            if p.grad is None:
                finite = False
                break
            g = p.grad.detach().float() / (LOSS_SCALE * accum)
            if not torch.isfinite(g).all():
                finite = False
                break
            grads[n] = g
        if finite:
            gn = torch.sqrt(sum((g ** 2).sum() for g in grads.values()))
            clip = min(1.0, 1.0 / (float(gn) + 1e-12))
            for n, m in masters.items():
                m.grad = grads[n] * clip
            opt.step()
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(masters[n].to(p.dtype))
            step += 1
        else:
            skipped += 1
        for p in params.values():
            p.grad = None
        for m in masters.values():
            m.grad = None

    for ep in range(epochs):
        for bi, s in enumerate(samples):
            ids, ans = encode(tokenizer, s, device)
            full = torch.cat([ids, ans], dim=1)
            labels = full.clone()
            labels[:, : ids.shape[1]] = -100
            loss = model(input_ids=full, labels=labels, use_cache=False).loss
            if torch.isfinite(loss):
                (loss * LOSS_SCALE / accum).backward()
            if (bi + 1) % accum == 0:
                apply_update()
    if any(p.grad is not None for p in params.values()):
        apply_update()
    model.eval()
    if skipped:
        print(f"  ({skipped} non-finite update(s) skipped)")
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/ruler_qwen2p5_7b.yaml"))
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 2e-5 subset, 2e-4 lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_data_dir", default="results/data_ft")
    ap.add_argument("--eval_data_dir", default=None,
                    help="override eval dir (default: cfg data_dir/main)")
    ap.add_argument("--capabilities", nargs="*", default=None,
                    help="capability/task names (default: cfg tasks)")
    ap.add_argument("--output_dir", default="results/interference")
    ap.add_argument("--trainer", choices=["subset", "lora"], default="subset",
                    help="subset: Adam on the sketched q/v params; "
                         "lora: rank-r adapters on q/v of ALL layers "
                         "(trained params != sketched params -> robustness)")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    ap.add_argument("--ce_only", action="store_true",
                    help="skip greedy-decoding accuracy (CE-only outcome)")
    ap.add_argument("--embeddings", action="store_true", default=True,
                    help="save mean last-hidden-state embeddings (baselines)")
    args = ap.parse_args()
    lr = args.lr or (2e-4 if args.trainer == "lora" else 2e-5)

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    print_gpu_info()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    families = args.capabilities or list(cfg["tasks"])
    eval_dir = Path(args.eval_data_dir).resolve() if args.eval_data_dir \
        else REPO_ROOT / cfg["data_dir"] / "main"
    train_dir = (REPO_ROOT / args.train_data_dir / "main"
                 if not Path(args.train_data_dir).is_absolute()
                 else Path(args.train_data_dir) / "main")
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

    # --- embeddings (baseline predictor: representation similarity) ---
    emb_path = out_dir / "embeddings.npz"
    if args.embeddings and not emb_path.exists():
        embs = {}
        with torch.no_grad():
            for task in families:
                for tag, samples in [("train", train_sets[task]),
                                     ("eval", eval_sets[task])]:
                    for s in tqdm(samples, desc=f"embed {task} {tag}"):
                        ids, _ = encode(tokenizer, s, device)
                        out = model(input_ids=ids, output_hidden_states=True,
                                    use_cache=False)
                        h = out.hidden_states[-1][0].float().mean(dim=0)
                        embs[f"{tag}__{s['sample_id']}"] = h.cpu().numpy()
        np.savez_compressed(emb_path, **embs)
        print(f"embeddings -> {emb_path}")

    if args.trainer == "lora":
        from rank_demand.lora import inject_lora, remove_lora
        params = inject_lora(model, layers=None, r=args.lora_rank,
                             alpha=args.lora_alpha)
    else:
        params = select_ntk_params(model, layers)

    # --- PRE eval ---
    pre_path = out_dir / "pre_eval.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
    else:
        pre = {}
        for task in families:
            ce, acc = eval_ce_acc(model, tokenizer, eval_sets[task], device,
                                  ce_only=args.ce_only)
            pre[task] = {"ce": ce, "acc": acc}
            print(f"PRE {task}: ce={ce:.4f} acc={acc}")
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
                                lr=lr, epochs=args.epochs)
        post = {}
        for task in families:
            ce, acc = eval_ce_acc(model, tokenizer, eval_sets[task], device,
                                  ce_only=args.ce_only)
            post[task] = {"dce": round(ce - pre[task]["ce"], 5),
                          "dacc": (round(acc - pre[task]["acc"], 4)
                                   if acc is not None else None),
                          "ce": ce, "acc": acc}
        measured[train_task] = {"steps": steps, "post": post,
                                "trainer": args.trainer, "lr": lr,
                                "minutes": round((time.time() - t0) / 60, 1)}
        # restore params/adapters to the exact pre-training state (for LoRA
        # this restores the same A-init for every family -> comparable runs)
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(snapshot[n])
        meas_path.write_text(json.dumps(measured, indent=2))
        print(f"train on {train_task}: "
              + " ".join(f"{t}:dce={post[t]['dce']:+.3f}" for t in families))
    if args.trainer == "lora":
        remove_lora(model)

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
