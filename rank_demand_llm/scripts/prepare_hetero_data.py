#!/usr/bin/env python
"""Heterogeneous capability sets for the interference experiment.

Capabilities:
  gsm8k       math word problems (answer = full CoT solution, CE target)
  mbpp_code   python function synthesis (answer = reference code)
  qa_1 / niah_single_1 / cwe   copied from the existing RULER sets @1024

Writes:
  results/data_hetero/train/main/{cap}_1024.jsonl   (n_train per capability)
  results/data_hetero/eval/main/{cap}_1024.jsonl    (n_eval per capability)
(The `_1024` suffix only satisfies the loader's naming; gsm8k/mbpp prompts
are naturally short.)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand.utils import read_jsonl, set_seed, setup_logging  # noqa: E402


def rows_gsm8k(n_train, n_eval, seed):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    ev = ds["test"].shuffle(seed=seed).select(range(n_eval))

    def conv(split, tag):
        out = []
        for i, ex in enumerate(split):
            out.append({
                "sample_id": f"gsm8k_{tag}_{i}",
                "ruler_task": "gsm8k", "task_family": "math",
                "target_context_length": 1024,
                "prompt_text": ("Solve the following math problem step by "
                                "step, then give the final answer after "
                                "'####'.\n\n" + ex["question"]),
                "answer_prefix": "",
                "expected_answer": [ex["answer"]],
                "evidence_position_status": "unavailable",
            })
        return out
    return conv(train, "train"), conv(ev, "eval")


def rows_mbpp(n_train, n_eval, seed):
    from datasets import load_dataset
    ds = load_dataset("mbpp", "full")
    pool = ds["train"].shuffle(seed=seed)
    train = pool.select(range(n_train))
    ev = ds["test"].shuffle(seed=seed).select(range(n_eval))

    def conv(split, tag):
        out = []
        for i, ex in enumerate(split):
            tests = "\n".join(ex["test_list"][:2])
            out.append({
                "sample_id": f"mbpp_{tag}_{i}",
                "ruler_task": "mbpp_code", "task_family": "code",
                "target_context_length": 1024,
                "prompt_text": (f"{ex['text']}\n\nYour code should satisfy "
                                f"these tests:\n{tests}\n\nWrite the Python "
                                "function."),
                "answer_prefix": "",
                "expected_answer": [ex["code"]],
                "evidence_position_status": "unavailable",
            })
        return out
    return conv(train, "train"), conv(ev, "eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--ruler_eval_dir", default="results/data/main")
    ap.add_argument("--ruler_train_dir", default="results/data_ft/main")
    ap.add_argument("--out", default="results/data_hetero")
    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    out = REPO_ROOT / args.out
    tdir = out / "train" / "main"
    edir = out / "eval" / "main"
    tdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)

    def dump(rows, path):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{len(rows):4d} -> {path}")

    for name, fn in [("gsm8k", rows_gsm8k), ("mbpp_code", rows_mbpp)]:
        tr, ev = fn(args.n_train, args.n_eval, args.seed)
        dump(tr, tdir / f"{name}_1024.jsonl")
        dump(ev, edir / f"{name}_1024.jsonl")

    # RULER capabilities: reuse existing eval + train splits
    for cap in ["qa_1", "niah_single_1", "cwe"]:
        src_e = REPO_ROOT / args.ruler_eval_dir / f"{cap}_1024.jsonl"
        src_t = REPO_ROOT / args.ruler_train_dir / f"{cap}_1024.jsonl"
        dump(read_jsonl(src_e)[: args.n_eval], edir / f"{cap}_1024.jsonl")
        dump(read_jsonl(src_t)[: args.n_train], tdir / f"{cap}_1024.jsonl")

    print("capabilities:", ["gsm8k", "mbpp_code", "qa_1", "niah_single_1", "cwe"])


if __name__ == "__main__":
    main()
