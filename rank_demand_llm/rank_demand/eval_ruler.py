"""Main experiment runner: inference + rank metrics on prepared RULER data.

Usage:
  python -m rank_demand.eval_ruler --config configs/ruler_qwen2p5_7b.yaml \
      --mode smoke --tasks niah_single_1 vt --lengths 1024 --num_samples 5 \
      --output_dir results/smoke_run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_demand import rank_metrics as rm  # noqa: E402
from rank_demand.config import load_config  # noqa: E402
from rank_demand.data_ruler import evidence_token_positions  # noqa: E402
from rank_demand.generation import build_prompt, generate_answer  # noqa: E402
from rank_demand.hooks import measurement_forward, select_heads, select_layers  # noqa: E402
from rank_demand.model_loader import load_model_and_tokenizer  # noqa: E402
from rank_demand.utils import (append_jsonl, clear_gpu, oom_guard,  # noqa: E402
                               print_gpu_info, read_jsonl, set_seed,
                               setup_logging)

logger = logging.getLogger("rank_demand.eval")

# RULER official per-task metric type (external/RULER/scripts/eval/synthetic/constants.py)
METRIC_TYPE = {
    "niah": "all", "variable_tracking": "all", "common_words_extraction": "all",
    "freq_words_extraction": "all", "qa": "part",
}
TASK_TO_BASE = {
    "niah_single_1": "niah", "niah_single_2": "niah", "niah_single_3": "niah",
    "niah_multikey_1": "niah", "niah_multikey_2": "niah", "niah_multikey_3": "niah",
    "niah_multivalue": "niah", "niah_multiquery": "niah",
    "vt": "variable_tracking", "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction", "qa_1": "qa", "qa_2": "qa",
}


def score_prediction(task: str, prediction: str, refs: list[str]) -> dict:
    """RULER string-match scoring for a single sample.

    'all' tasks: fraction of references present (correct iff all present).
    'part' tasks (qa): correct iff any reference present.
    """
    base = TASK_TO_BASE.get(task, "niah")
    mtype = METRIC_TYPE[base]
    pred_l = prediction.lower()
    hits = [1.0 if r.lower() in pred_l else 0.0 for r in refs]
    if mtype == "all":
        frac = sum(hits) / max(len(hits), 1)
        correct = frac == 1.0
    else:
        frac = max(hits) if hits else 0.0
        correct = frac == 1.0
    parse_status = "ok" if prediction.strip() else "empty_prediction"
    return {"correct": bool(correct), "match_fraction": frac,
            "metric_type": mtype, "parse_status": parse_status,
            "parsed_answer": prediction.strip()[:500]}


def compute_sample_metrics(model, tokenizer, input_ids, sample, mcfg,
                           run_truncation: bool, skips: list) -> tuple[dict, dict]:
    """Measurement pass + all rank metrics. Returns (row_summary, npz_payload)."""
    T = input_ids.shape[1]
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    layers = select_layers(n_layers, mcfg["layer_fractions"])

    attn_cap = mcfg["exact_attention_max_tokens"]
    if mcfg.get("allow_exact_attention_4096"):
        attn_cap = max(attn_cap, 4096)
    collect_attention = T <= attn_cap and not mcfg.get("hidden_only", False)
    heads = select_heads(n_heads, T, mcfg["all_heads_max_tokens"],
                         mcfg["sampled_heads"]) if collect_attention else []
    qk_layers = layers if (mcfg.get("score_matrix") and collect_attention) else None

    summary: dict = {
        "measured_layers": layers,
        "measured_heads": heads,
        "attention_collected": collect_attention,
        "hidden": {}, "attention": {}, "score": {}, "evidence_survival": {},
    }
    if not collect_attention:
        summary["attention_skip_reason"] = (
            f"T={T} > exact_attention_max_tokens={attn_cap}"
            if not mcfg.get("hidden_only") else "hidden_only mode")
    npz: dict[str, np.ndarray] = {}

    hidden_states = attentions = capture = None
    with oom_guard(f"measurement forward T={T}", skips) as st:
        hidden_states, attentions, capture = measurement_forward(
            model, input_ids, collect_attention, qk_layers)
    if st["skipped"] and collect_attention:
        # retry without attention (hidden states only)
        summary["attention_collected"] = False
        summary["attention_skip_reason"] = st["skip_reason"]
        with oom_guard(f"measurement forward (hidden only) T={T}", skips) as st2:
            hidden_states, attentions, capture = measurement_forward(
                model, input_ids, False, None)
        if st2["skipped"]:
            summary["measurement_skip_reason"] = st2["skip_reason"]
            return summary, npz
    elif st["skipped"]:
        summary["measurement_skip_reason"] = st["skip_reason"]
        return summary, npz

    # evidence token positions in the final prompt
    ev_tok: list[int] = []
    if sample["evidence_position_status"] != "unavailable" and sample["evidence_text"]:
        try:
            full_prompt = sample["_full_prompt"]
            ev_tok = evidence_token_positions(tokenizer, full_prompt,
                                              sample["evidence_text"])
        except Exception as e:
            skips.append({"what": "evidence_token_positions",
                          "skip_reason": str(e)[:200]})
    summary["num_evidence_tokens"] = len(ev_tok)

    # --- A. hidden-state metrics + D. evidence survival ---
    for li in layers:
        H = hidden_states[li + 1][0]  # (T, d); +1 skips embedding layer output
        with oom_guard(f"hidden metrics L{li}", skips, broad=True) as st:
            hm = rm.hidden_state_metrics(
                H, center=mcfg["center_hidden"],
                max_tokens=mcfg["hidden_subsample_max_tokens"])
            npz[f"hidden_spectrum_L{li}"] = hm.pop("spectrum")
            summary["hidden"][str(li)] = hm
        if st["skipped"]:
            summary["hidden"][str(li)] = {"skip_reason": st["skip_reason"]}
            continue
        if ev_tok:
            with oom_guard(f"evidence survival L{li}", skips, broad=True) as st:
                es = rm.evidence_survival(
                    H, ev_tok, ks=mcfg["evidence_ks"],
                    center=mcfg["center_hidden"],
                    rng=np.random.default_rng(hash(sample["sample_id"]) % 2**32))
                summary["evidence_survival"][str(li)] = es
            if st["skipped"]:
                summary["evidence_survival"][str(li)] = \
                    {"skip_reason": st["skip_reason"]}

    # --- B. attention metrics ---
    if summary["attention_collected"] and attentions is not None:
        for li in layers:
            A_layer = attentions[li][0]  # (n_heads, T, T)
            per_head = {}
            for h in heads:
                with oom_guard(f"attn metrics L{li}H{h}", skips, broad=True) as st:
                    am = rm.attention_matrix_metrics(
                        A_layer[h], block_size=mcfg["block_size"],
                        exact_rank_max_tokens=mcfg["exact_rank_max_tokens"])
                    npz[f"attn_block_L{li}H{h}"] = am.pop("block_matrix")
                    per_head[str(h)] = am
                if st["skipped"]:
                    per_head[str(h)] = {"skip_reason": st["skip_reason"]}
            summary["attention"][str(li)] = per_head

            # optional low-rank truncation diagnostic (offline AV reconstruction)
            if run_truncation and T <= 1024:
                v_layer = None
                try:
                    # reconstruct V from v_proj weights: V = X W_v^T (GQA heads)
                    attn_mod = model.model.layers[li].self_attn
                    X = hidden_states[li][0].to(attn_mod.v_proj.weight.dtype)
                    V_all = attn_mod.v_proj(X)  # (T, n_kv*d_head)
                    n_kv = model.config.num_key_value_heads
                    d_head = V_all.shape[-1] // n_kv
                    v_layer = V_all.view(T, n_kv, d_head)
                except Exception as e:
                    skips.append({"what": f"truncation V L{li}",
                                  "skip_reason": str(e)[:200]})
                if v_layer is not None:
                    trunc = {}
                    group = n_heads // model.config.num_key_value_heads
                    for h in heads[:4]:  # cap cost: first few heads
                        with oom_guard(f"truncation L{li}H{h}", skips, broad=True) as st:
                            tr = rm.attention_truncation_error(
                                A_layer[h], v_layer[:, h // group],
                                ks=mcfg["truncation_ks"])
                            trunc[str(h)] = tr
                        if st["skipped"]:
                            trunc[str(h)] = {"skip_reason": st["skip_reason"]}
                    summary.setdefault("truncation", {})[str(li)] = trunc

    # --- C. pre-softmax score diagnostics ---
    if capture is not None:
        score_heads = heads[:3] if heads else []
        for li in layers:
            per_head = {}
            for h in score_heads:
                with oom_guard(f"score matrix L{li}H{h}", skips, broad=True) as st:
                    S = capture.score_matrix(li, h)
                    if S is not None:
                        per_head[str(h)] = rm.score_matrix_metrics(
                            S, block_size=mcfg["block_size"])
                        del S
                if st["skipped"]:
                    per_head[str(h)] = {"skip_reason": st["skip_reason"]}
            if per_head:
                summary["score"][str(li)] = per_head
        capture.clear()

    del hidden_states, attentions
    clear_gpu()
    return summary, npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/ruler_qwen2p5_7b.yaml"))
    ap.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--exact_attention_max_tokens", type=int, default=None)
    ap.add_argument("--allow_exact_attention_4096", action="store_true")
    ap.add_argument("--hidden_only", action="store_true",
                    help="skip all attention-weight metrics")
    ap.add_argument("--run_attention_truncation", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config, {
        "model_id": args.model_id, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    })
    mcfg = dict(cfg["metrics"])
    if args.exact_attention_max_tokens:
        mcfg["exact_attention_max_tokens"] = args.exact_attention_max_tokens
    if args.allow_exact_attention_4096:
        mcfg["allow_exact_attention_4096"] = True
    mcfg["hidden_only"] = args.hidden_only

    set_seed(cfg["seed"])
    gpu_info = print_gpu_info()

    mode_cfg = cfg[args.mode]
    lengths = args.lengths or mode_cfg["context_lengths"]
    num_samples = args.num_samples or mode_cfg["num_samples"]
    tasks = {t: f for t, f in cfg["tasks"].items()
             if args.tasks is None or t in args.tasks}

    data_dir = Path(args.data_dir or REPO_ROOT / cfg["data_dir"]) / args.mode
    out_dir = Path(args.output_dir or REPO_ROOT / cfg["output_dir"])
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # load samples (ascending length: fail late on big contexts)
    samples = []
    missing = []
    for task, family in tasks.items():
        for length in sorted(lengths):
            p = data_dir / f"{task}_{length}.jsonl"
            if not p.exists():
                missing.append(str(p))
                continue
            rows = read_jsonl(p)[:num_samples]
            samples.extend(rows)
    if missing:
        logger.warning("missing data files (run prepare_ruler_data.py?): %s", missing)
    if not samples:
        logger.error("no samples found under %s — aborting", data_dir)
        sys.exit(1)
    samples.sort(key=lambda s: s["target_context_length"])
    logger.info("running %d samples | tasks=%s | lengths=%s",
                len(samples), list(tasks), sorted(lengths))

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"], dtype=cfg["dtype"], load_4bit=args.load_4bit or cfg["load_4bit"],
        attn_implementation=cfg["attn_implementation_measure"])
    device = next(model.parameters()).device

    run_meta = {
        "command": " ".join(sys.argv), "model_id": cfg["model_id"],
        "mode": args.mode, "tasks": tasks, "lengths": sorted(lengths),
        "num_samples_per_task_length": num_samples, "seed": cfg["seed"],
        "gpu": gpu_info, "metrics_config": mcfg,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    done_ids = set()
    if results_path.exists():
        done_ids = {r["sample_id"] for r in read_jsonl(results_path)}
        logger.info("resuming: %d samples already done", len(done_ids))

    n_fail = 0
    for sample in tqdm(samples, desc="samples"):
        if sample["sample_id"] in done_ids:
            continue
        skips: list[dict] = []
        row = {
            "sample_id": sample["sample_id"], "model_id": cfg["model_id"],
            "task": sample["ruler_task"], "task_family": sample["task_family"],
            "target_context_length": sample["target_context_length"],
            "expected_answer": sample["expected_answer"],
            "evidence_position_status": sample["evidence_position_status"],
        }
        try:
            full_prompt = build_prompt(tokenizer, sample["prompt_text"],
                                       sample.get("answer_prefix", ""))
            sample["_full_prompt"] = full_prompt
            enc = tokenizer(full_prompt, return_tensors="pt",
                            add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            T = input_ids.shape[1]
            row["actual_input_tokens"] = int(T)
            # evidence relative position (position control for regression)
            if sample.get("evidence_positions"):
                mid = np.mean([(a + b) / 2 for a, b in sample["evidence_positions"]])
                row["evidence_rel_position"] = round(float(mid) / max(len(full_prompt), 1), 4)
            else:
                row["evidence_rel_position"] = None

            metrics_summary, npz = compute_sample_metrics(
                model, tokenizer, input_ids, sample, mcfg,
                args.run_attention_truncation, skips)

            gen = None
            with oom_guard(f"generation {sample['sample_id']}", skips) as st:
                gen = generate_answer(
                    model, tokenizer, input_ids,
                    max_new_tokens=cfg["max_new_tokens"],
                    generate_attn_impl=cfg["attn_implementation_generate"])
            if gen is None:
                row.update({"prediction_text": None, "correct": None,
                            "parse_status": "generation_oom"})
            else:
                row.update(gen)
                row.update(score_prediction(sample["ruler_task"],
                                            gen["prediction_text"],
                                            sample["expected_answer"]))

            # persist NPZ metrics
            mpath = metrics_dir / f"{sample['sample_id']}.npz"
            np.savez_compressed(
                mpath, **npz,
                summary=np.array(json.dumps(metrics_summary, default=str)))
            row["metrics_file"] = str(mpath.relative_to(out_dir))
            row["metrics_summary"] = metrics_summary
            row["skips"] = skips
        except Exception as e:
            logger.exception("sample %s failed", sample["sample_id"])
            row.update({"prediction_text": None, "correct": None,
                        "parse_status": f"error: {str(e)[:200]}", "skips": skips})
            n_fail += 1
            clear_gpu()
        append_jsonl(results_path, row)

    logger.info("done. results -> %s (%d hard failures)", results_path, n_fail)


if __name__ == "__main__":
    main()
