#!/usr/bin/env python
"""Generate RULER examples via the official generator, convert to our schema.

Usage:
  python scripts/prepare_ruler_data.py --config configs/ruler_qwen2p5_7b.yaml --mode smoke
  python scripts/prepare_ruler_data.py --config ... --mode main
  python scripts/prepare_ruler_data.py --config ... --lengths 1024 2048 --num_samples 5 --tasks niah_single_1 vt

Output: {data_dir}/{mode}/{task}_{length}.jsonl in our sample schema.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand.config import load_config  # noqa: E402
from rank_demand.data_ruler import convert_ruler_row  # noqa: E402
from rank_demand.utils import read_jsonl, setup_logging  # noqa: E402

logger = logging.getLogger("rank_demand.prepare")

RULER_DATA = REPO_ROOT / "external" / "RULER" / "scripts" / "data"


def run_ruler_prepare(task: str, length: int, num_samples: int, seed: int,
                      tokenizer_path: str, raw_dir: Path) -> Path:
    """Invoke RULER's official prepare.py for one (task, length)."""
    save_dir = raw_dir / f"len{length}"
    out_file = save_dir / task / "validation.jsonl"
    if out_file.exists() and len(out_file.read_text().splitlines()) >= num_samples:
        logger.info("raw exists, skip: %s", out_file)
        return out_file
    cmd = [
        sys.executable, str(RULER_DATA / "prepare.py"),
        "--save_dir", str(save_dir),
        "--benchmark", "synthetic",
        "--task", task,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", "hf",
        "--max_seq_length", str(length),
        "--model_template_type", "base",
        "--num_samples", str(num_samples),
        "--random_seed", str(seed),
    ]
    logger.info("RULER prepare: task=%s len=%d n=%d", task, length, num_samples)
    # RULER's prepare.py shells out to bare `python`; make sure that resolves
    # to THIS interpreter's env, not the system python.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    res = subprocess.run(cmd, cwd=str(RULER_DATA), capture_output=True,
                         text=True, env=env)
    if res.returncode != 0 or not out_file.exists():
        logger.error("RULER prepare failed for %s@%d\nstdout:\n%s\nstderr:\n%s",
                     task, length, res.stdout[-3000:], res.stderr[-3000:])
        raise RuntimeError(f"RULER prepare failed: {task}@{length}")
    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/ruler_qwen2p5_7b.yaml"))
    ap.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of task names (default: all in config)")
    ap.add_argument("--lengths", nargs="*", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (e.g. disjoint train split)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    mode_cfg = cfg[args.mode]
    lengths = args.lengths or mode_cfg["context_lengths"]
    num_samples = args.num_samples or mode_cfg["num_samples"]
    tasks = {t: f for t, f in cfg["tasks"].items()
             if args.tasks is None or t in args.tasks}
    # absolutize: the RULER subprocess runs with cwd=external/RULER/scripts/data
    data_dir = Path(args.data_dir or REPO_ROOT / cfg["data_dir"]).resolve() / args.mode
    raw_dir = data_dir / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    ev_stats: dict[str, dict] = {}
    for task, family in tasks.items():
        for length in lengths:
            try:
                raw = run_ruler_prepare(task, length, num_samples, cfg["seed"],
                                        cfg["model_id"], raw_dir)
            except Exception as e:
                logger.error("SKIP %s@%d: %s", task, length, e)
                failures.append({"task": task, "length": length, "error": str(e)})
                continue
            rows = read_jsonl(raw)[:num_samples]
            out_path = data_dir / f"{task}_{length}.jsonl"
            statuses: dict[str, int] = {}
            with open(out_path, "w") as f:
                for i, row in enumerate(rows):
                    sample = convert_ruler_row(row, task, family, length, i)
                    statuses[sample["evidence_position_status"]] = \
                        statuses.get(sample["evidence_position_status"], 0) + 1
                    f.write(json.dumps(sample) + "\n")
            ev_stats[f"{task}@{length}"] = statuses
            logger.info("wrote %d samples -> %s  evidence: %s",
                        len(rows), out_path, statuses)

    manifest = {
        "mode": args.mode, "lengths": lengths, "num_samples": num_samples,
        "tasks": tasks, "failures": failures, "evidence_status": ev_stats,
        "tokenizer": cfg["model_id"], "seed": cfg["seed"],
    }
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest -> %s", data_dir / "manifest.json")
    if failures:
        logger.warning("%d task/length combos FAILED: %s", len(failures),
                       [(x["task"], x["length"]) for x in failures])


if __name__ == "__main__":
    main()
