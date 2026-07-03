"""Config loading: YAML file + CLI overrides -> plain namespace-ish dict."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "seed": 1234,
    "dtype": "auto",              # auto | bf16 | fp16 | fp32
    "load_4bit": False,
    "attn_implementation_measure": "eager",
    "attn_implementation_generate": "sdpa",
    "max_new_tokens": 64,
    "data_dir": "results/data",
    "output_dir": "results/run",
    # task name -> family (verified against external/RULER/scripts/synthetic.yaml)
    "tasks": {
        "niah_single_1": "single_needle",
        "niah_multikey_1": "multi_needle",
        "vt": "multi_hop_tracing",
        "cwe": "aggregation",
        "qa_1": "qa",
    },
    "smoke": {"context_lengths": [1024, 2048], "num_samples": 5},
    "main": {"context_lengths": [1024, 2048, 4096, 8192], "num_samples": 50},
    "metrics": {
        "layer_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "center_hidden": True,
        "hidden_subsample_max_tokens": 8192,
        "exact_attention_max_tokens": 2048,   # full A only below this
        "allow_exact_attention_4096": False,  # explicit opt-in
        "exact_rank_max_tokens": 1024,        # exact effective rank of full A
        "block_size": 64,
        "all_heads_max_tokens": 1024,         # all heads below this, else sampled
        "sampled_heads": [0, "middle", "last"],
        "evidence_ks": [8, 16, 32, 64, 128, 256],
        "score_matrix": True,                 # pre-softmax QK^T diagnostics
        "store_attention": False,
        "truncation_ks": [4, 8, 16, 32, 64, 128],
    },
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg
