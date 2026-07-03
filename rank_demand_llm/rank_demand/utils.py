"""Shared utilities: seeding, JSONL IO, GPU info, OOM guards."""
from __future__ import annotations

import gc
import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger("rank_demand")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict) -> None:
    """Append one row and flush immediately (crash-safe incremental results)."""
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def print_gpu_info() -> dict:
    """Print GPU name and VRAM at start; return info dict."""
    info: dict[str, Any] = {"cuda": False}
    try:
        import torch

        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            p = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            info = {
                "cuda": True,
                "device_index": i,
                "name": p.name,
                "total_gb": round(total_b / 1e9, 1),
                "free_gb": round(free_b / 1e9, 1),
                "compute_capability": f"{p.major}.{p.minor}",
            }
            logger.info(
                "GPU: %s | %.1f GB total, %.1f GB free | sm_%d%d",
                p.name, total_b / 1e9, free_b / 1e9, p.major, p.minor,
            )
        else:
            logger.warning("CUDA not available — running on CPU will be very slow.")
    except ImportError:
        logger.warning("torch not importable")
    return info


def max_gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    return None


def clear_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def oom_guard(what: str, skips: list[dict] | None = None,
              broad: bool = False) -> Iterator[dict]:
    """Catch CUDA OOM (and any RuntimeError mentioning memory), record skip,
    continue. With broad=True, catch ANY exception (for per-metric blocks
    where a numerical failure — e.g. cuSOLVER non-convergence — must not kill
    the whole sample). Skips are always recorded, never silent.

    Usage:
        with oom_guard("attention metrics L14", skips) as status:
            ...
        if status["skipped"]: ...
    """
    import torch

    status = {"skipped": False, "skip_reason": None}
    try:
        yield status
    except torch.cuda.OutOfMemoryError as e:
        status["skipped"] = True
        status["skip_reason"] = f"cuda_oom: {str(e)[:200]}"
        logger.warning("OOM during %s — clearing cache and continuing", what)
        clear_gpu()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status["skipped"] = True
            status["skip_reason"] = f"oom_runtime: {str(e)[:200]}"
            logger.warning("OOM (RuntimeError) during %s — continuing", what)
            clear_gpu()
        elif broad:
            status["skipped"] = True
            status["skip_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning("error during %s (%s) — continuing", what, e)
        else:
            raise
    except Exception as e:
        if not broad:
            raise
        status["skipped"] = True
        status["skip_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        logger.warning("error during %s (%s) — continuing", what, e)
    if status["skipped"] and skips is not None:
        skips.append({"what": what, "skip_reason": status["skip_reason"]})
