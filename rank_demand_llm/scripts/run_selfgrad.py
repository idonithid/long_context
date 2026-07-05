#!/usr/bin/env python
"""Method experiment 1: test-time eNTK signal for correctness prediction.

For every sample of a completed run, teacher-force the model's OWN generated
answer (no gold labels -> deployable at test time) and compute:
  - self_grad_norm / self_kernel   (the method)
  - self answer CE == -mean token logprob, min token logprob,
    mean/max predictive entropy   (the cheap baselines)
Ground truth `correct` comes from the completed run's results.jsonl.

  python scripts/run_selfgrad.py --config configs/ruler_qwen2p5_7b.yaml \
      --results results/main_run --output results/main_run/selfgrad.jsonl
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True, help="completed run dir")
    ap.add_argument("--output", default=None)
    ap.add_argument("--mode", default="main")
    ap.add_argument("--max_tokens", type=int, default=8192)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    print_gpu_info()

    results_dir = Path(args.results)
    out_path = Path(args.output or results_dir / "selfgrad.jsonl")
    rows = read_jsonl(results_dir / "results.jsonl")
    data_dir = REPO_ROOT / cfg["data_dir"] / args.mode
    prompts = {}
    for task in cfg["tasks"]:
        for length in cfg[args.mode]["context_lengths"]:
            p = data_dir / f"{task}_{length}.jsonl"
            if p.exists():
                for s in read_jsonl(p):
                    prompts[s["sample_id"]] = s

    done = {r["sample_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = [r for r in rows if r["sample_id"] not in done]
    rows.sort(key=lambda r: r["target_context_length"])

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"], dtype=cfg["dtype"],
        attn_implementation=cfg["attn_implementation_generate"])
    device = next(model.parameters()).device
    layers = select_layers(model.config.num_hidden_layers,
                           cfg["metrics"]["layer_fractions"])

    with NTKExtractor(model, layers, seed=cfg["seed"]) as ntk:
        for r in tqdm(rows, desc="selfgrad"):
            s = prompts.get(r["sample_id"])
            row = {"sample_id": r["sample_id"], "task": r["task"],
                   "task_family": r["task_family"],
                   "target_context_length": r["target_context_length"],
                   "correct": r.get("correct"),
                   "match_fraction": r.get("match_fraction")}
            pred = (r.get("prediction_text") or "").rstrip()
            if s is None or not pred or r.get("correct") is None:
                row["skip_reason"] = "missing prompt/prediction/label"
                append_jsonl(out_path, row)
                continue
            prompt = build_prompt(tokenizer, s["prompt_text"],
                                  s.get("answer_prefix", ""))
            ids = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
            ans = tokenizer(pred, return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
            row["prompt_tokens"] = int(ids.shape[1])
            row["answer_tokens"] = int(ans.shape[1])
            if ids.shape[1] > args.max_tokens or ans.shape[1] == 0:
                row["skip_reason"] = f"T={ids.shape[1]} > cap or empty answer"
                append_jsonl(out_path, row)
                continue
            skips = []
            t0 = time.time()
            with oom_guard(f"selfgrad {r['sample_id']}", skips, broad=True) as st:
                f = ntk.features(ids, ans, return_token_stats=True)
                f.pop("sketch")   # not needed for this experiment
                f.pop("group_norms")
                row.update(f)
                row["time_sec"] = round(time.time() - t0, 2)
            if st["skipped"]:
                row["skip_reason"] = st["skip_reason"]
                clear_gpu()
            append_jsonl(out_path, row)

    print(f"selfgrad -> {out_path}")


if __name__ == "__main__":
    main()
