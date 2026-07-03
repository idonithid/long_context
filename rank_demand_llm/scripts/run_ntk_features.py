#!/usr/bin/env python
"""Empirical-NTK per-prompt features over prepared RULER samples.

Standalone pass (own model load; run after/independently of eval_ruler):
  python scripts/run_ntk_features.py --config configs/ruler_qwen2p5_7b.yaml \
      --mode smoke --tasks niah_single_1 vt --lengths 1024 --num_samples 5 \
      --output_dir results/smoke_run

Writes {output_dir}/ntk_features.jsonl (+ ntk_sketches.npz) keyed by sample_id
so analysis can join with results.jsonl.
"""
from __future__ import annotations

import argparse
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
from rank_demand.generation import build_prompt  # noqa: E402
from rank_demand.hooks import select_layers  # noqa: E402
from rank_demand.model_loader import load_model_and_tokenizer  # noqa: E402
from rank_demand.ntk_features import NTKExtractor  # noqa: E402
from rank_demand.utils import (append_jsonl, clear_gpu, oom_guard,  # noqa: E402
                               print_gpu_info, read_jsonl, set_seed,
                               setup_logging)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/ruler_qwen2p5_7b.yaml"))
    ap.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--sketch_dim", type=int, default=4096)
    ap.add_argument("--max_ntk_tokens", type=int, default=4096,
                    help="skip samples longer than this (backward memory)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    print_gpu_info()

    mode_cfg = cfg[args.mode]
    lengths = args.lengths or mode_cfg["context_lengths"]
    num_samples = args.num_samples or mode_cfg["num_samples"]
    tasks = {t: f for t, f in cfg["tasks"].items()
             if args.tasks is None or t in args.tasks}
    data_dir = Path(args.data_dir or REPO_ROOT / cfg["data_dir"]) / args.mode
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "ntk_features.jsonl"

    samples = []
    for task, family in tasks.items():
        for length in sorted(lengths):
            p = data_dir / f"{task}_{length}.jsonl"
            if p.exists():
                samples.extend(read_jsonl(p)[:num_samples])
    samples.sort(key=lambda s: s["target_context_length"])

    done = set()
    if out_jsonl.exists():
        done = {r["sample_id"] for r in read_jsonl(out_jsonl)}

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"], dtype=cfg["dtype"],
        attn_implementation=cfg["attn_implementation_generate"])  # sdpa: no attn weights needed
    device = next(model.parameters()).device
    layers = select_layers(model.config.num_hidden_layers,
                           cfg["metrics"]["layer_fractions"])

    sketches: dict[str, np.ndarray] = {}
    npz_path = out_dir / "ntk_sketches.npz"
    if npz_path.exists():
        old = np.load(npz_path)
        sketches = {k: old[k] for k in old.files}

    with NTKExtractor(model, layers, sketch_dim=args.sketch_dim,
                      seed=cfg["seed"]) as ntk:
        for s in tqdm(samples, desc="ntk"):
            if s["sample_id"] in done:
                continue
            row = {"sample_id": s["sample_id"], "task": s["ruler_task"],
                   "task_family": s["task_family"],
                   "target_context_length": s["target_context_length"],
                   "ntk_layers": layers}
            skips: list = []
            prompt = build_prompt(tokenizer, s["prompt_text"],
                                  s.get("answer_prefix", ""))
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            answer_text = " " + ", ".join(map(str, s["expected_answer"]))
            ans = tokenizer(answer_text, return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
            row["prompt_tokens"] = int(input_ids.shape[1])
            row["answer_tokens"] = int(ans.shape[1])
            if input_ids.shape[1] > args.max_ntk_tokens:
                row["skip_reason"] = (f"T={input_ids.shape[1]} > "
                                      f"max_ntk_tokens={args.max_ntk_tokens}")
                append_jsonl(out_jsonl, row)
                continue
            t0 = time.time()
            with oom_guard(f"ntk {s['sample_id']}", skips, broad=True) as st:
                feats = ntk.features(input_ids, ans)
                sketches[s["sample_id"]] = feats.pop("sketch")
                row.update(feats)
                row["ntk_time_sec"] = round(time.time() - t0, 2)
            if st["skipped"]:
                row["skip_reason"] = st["skip_reason"]
                clear_gpu()
            append_jsonl(out_jsonl, row)
            np.savez_compressed(npz_path, **sketches)

    print(f"NTK features -> {out_jsonl}\nSketches -> {npz_path}")


if __name__ == "__main__":
    main()
