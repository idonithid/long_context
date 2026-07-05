#!/usr/bin/env python
"""Extended capability set for the interference experiment (overnight run).

Adds three new capability axes to the 5 heterogeneous ones:
  safety     hh-rlhf harmless-base single-turn (prompt -> safe assistant reply)
  xquad_zh   Chinese SQuAD (multilingual QA)
  arc_c      ARC-Challenge multiple choice (reasoning)

Copies the existing hetero capabilities (gsm8k, mbpp_code, qa_1,
niah_single_1, cwe) from results/data_hetero, writes everything to
results/data_hetero2/{train,eval}/main/{cap}_1024.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand.utils import set_seed, setup_logging  # noqa: E402


def base_row(cap, family, tag, i, prompt, answer):
    return {
        "sample_id": f"{cap}_{tag}_{i}",
        "ruler_task": cap, "task_family": family,
        "target_context_length": 1024,
        "prompt_text": prompt,
        "answer_prefix": "",
        "expected_answer": [answer],
        "evidence_position_status": "unavailable",
    }


def rows_safety(n_train, n_eval, seed):
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
    pool = ds["train"].shuffle(seed=seed)
    rows = []
    for ex in pool:
        t = ex["chosen"]
        if t.count("\n\nHuman:") != 1 or "\n\nAssistant:" not in t:
            continue
        human, _, rest = t.partition("\n\nAssistant:")
        prompt = human.replace("\n\nHuman:", "").strip()
        answer = rest.strip()
        if not (20 <= len(answer) <= 600) or not (10 <= len(prompt) <= 800):
            continue
        rows.append((prompt, answer))
        if len(rows) >= n_train + n_eval:
            break
    tr = [base_row("safety", "safety", "train", i, p, a)
          for i, (p, a) in enumerate(rows[:n_train])]
    ev = [base_row("safety", "safety", "eval", i, p, a)
          for i, (p, a) in enumerate(rows[n_train:n_train + n_eval])]
    return tr, ev


def rows_xquad(n_train, n_eval, seed):
    from datasets import load_dataset
    ds = load_dataset("xquad", "xquad.zh")["validation"].shuffle(seed=seed)

    def conv(split, tag, off=0):
        out = []
        for i, ex in enumerate(split):
            prompt = ("Read the passage and answer the question.\n\n"
                      f"Passage: {ex['context']}\n\nQuestion: {ex['question']}")
            out.append(base_row("xquad_zh", "multilingual", tag, i, prompt,
                                ex["answers"]["text"][0]))
        return out
    return (conv(ds.select(range(n_train)), "train"),
            conv(ds.select(range(n_train, n_train + n_eval)), "eval"))


def rows_arc(n_train, n_eval, seed):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Challenge")
    tr_pool = ds["train"].shuffle(seed=seed).select(range(n_train))
    ev_pool = ds["test"].shuffle(seed=seed).select(range(n_eval))

    def conv(split, tag):
        out = []
        for i, ex in enumerate(split):
            labels, texts = ex["choices"]["label"], ex["choices"]["text"]
            opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            prompt = ("Answer the following multiple-choice question with the "
                      "letter of the correct option followed by its text.\n\n"
                      f"{ex['question']}\n\nOptions:\n{opts}")
            ans_text = texts[labels.index(ex["answerKey"])]
            out.append(base_row("arc_c", "reasoning", tag, i, prompt,
                                f"{ex['answerKey']}. {ans_text}"))
        return out
    return conv(tr_pool, "train"), conv(ev_pool, "eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--hetero_dir", default="results/data_hetero")
    ap.add_argument("--out", default="results/data_hetero2")
    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    out = REPO_ROOT / args.out
    tdir, edir = out / "train" / "main", out / "eval" / "main"
    tdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)

    def dump(rows, path):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{len(rows):4d} -> {path}")

    for name, fn in [("safety", rows_safety), ("xquad_zh", rows_xquad),
                     ("arc_c", rows_arc)]:
        tr, ev = fn(args.n_train, args.n_eval, args.seed)
        assert len(tr) == args.n_train and len(ev) == args.n_eval, name
        dump(tr, tdir / f"{name}_1024.jsonl")
        dump(ev, edir / f"{name}_1024.jsonl")

    hetero = REPO_ROOT / args.hetero_dir
    for cap in ["gsm8k", "mbpp_code", "qa_1", "niah_single_1", "cwe"]:
        for sub, dst in [("train", tdir), ("eval", edir)]:
            src = hetero / sub / "main" / f"{cap}_1024.jsonl"
            shutil.copy(src, dst / src.name)
            print(f"copy -> {dst / src.name}")

    print("capabilities:", ["gsm8k", "mbpp_code", "qa_1", "niah_single_1",
                            "cwe", "safety", "xquad_zh", "arc_c"])


if __name__ == "__main__":
    main()
