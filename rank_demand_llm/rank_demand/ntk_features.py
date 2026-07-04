"""Empirical NTK features per data point (prompt).

For each sample we teacher-force the expected answer after the prompt and
compute the gradient of the mean answer-token cross-entropy w.r.t. a SMALL
parameter subset (default: q_proj/v_proj of the selected layers — the LoRA
target intuition: these carry most task-adaptation signal at a tractable
size, ~9M params for 5 layers of Qwen2.5-7B).

Per-sample features:
  - grad_norm            ||g||_2 over the subset (self-kernel sqrt)
  - self_kernel          k(x,x) = ||g||^2
  - per-param-group norms (layer-resolved)
Cross-sample features (computed in analysis from stored gradient sketches):
  - NTK Gram G_ij = <g_i, g_j> (on a random-projection sketch to keep memory
    bounded), its spectrum / effective rank, block structure by task family,
    and alignment with correctness.

Numerics: model weights are fp16 on this hardware; we scale the loss by
LOSS_SCALE before backward to avoid fp16 gradient underflow, then divide the
(fp32-cast) gradients by the same factor.

Memory: gradient checkpointing is enabled during the NTK pass; only the
selected params have requires_grad=True.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger("rank_demand.ntk")

LOSS_SCALE = 1024.0


def select_ntk_params(model, layer_indices: list[int],
                      kinds=("q_proj", "v_proj")) -> dict[str, torch.nn.Parameter]:
    """Pick the weight matrices of the given projections in selected layers."""
    chosen: dict[str, torch.nn.Parameter] = {}
    for li in layer_indices:
        attn = model.model.layers[li].self_attn
        for kind in kinds:
            mod = getattr(attn, kind, None)
            if mod is not None:
                chosen[f"L{li}.{kind}"] = mod.weight
    return chosen


class NTKExtractor:
    """Computes per-sample gradient features on a fixed parameter subset."""

    def __init__(self, model, layer_indices: list[int], sketch_dim: int = 4096,
                 seed: int = 0, kinds=("q_proj", "v_proj")):
        self.model = model
        self.params = select_ntk_params(model, layer_indices, kinds)
        n_total = sum(p.numel() for p in self.params.values())
        logger.info("NTK param subset: %d tensors, %.1fM params",
                    len(self.params), n_total / 1e6)
        self.sketch_dim = sketch_dim
        self.seed = seed
        self._was_training = model.training

    def __enter__(self):
        self._req = {n: p.requires_grad for n, p in
                     self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.params.values():
            p.requires_grad_(True)
        try:
            # use_reentrant=False is required: reentrant checkpointing needs
            # requires_grad inputs, but our embeddings are frozen.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception as e:
            logger.warning("gradient checkpointing unavailable: %s", e)
        # HF checkpointing is gated on module.training — in eval mode it is a
        # silent no-op (caused backward OOM at 4096 tokens). Qwen2.5 has
        # attention_dropout=0.0, so train mode stays deterministic here.
        self.model.train()
        return self

    def __exit__(self, *exc):
        for n, p in self.model.named_parameters():
            p.requires_grad_(self._req.get(n, False))
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:
            pass
        self.model.eval()
        return False

    def _sketch(self, g: torch.Tensor, name: str) -> np.ndarray:
        """Fixed random-sign projection (SRHT-lite) to sketch_dim; same seed
        per param tensor across samples -> inner products are preserved in
        expectation, so the Gram over sketches approximates the true Gram."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + (hash(name) % 2**31))
        n = g.numel()
        idx = torch.randint(0, self.sketch_dim, (n,), generator=gen)
        sign = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).float()
        out = torch.zeros(self.sketch_dim)
        out.index_add_(0, idx, g.flatten().cpu().float() * sign)
        return (out / np.sqrt(1.0)).numpy()

    @torch.enable_grad()
    def features(self, input_ids: torch.Tensor,
                 answer_ids: torch.Tensor) -> dict:
        """One backward pass. input_ids: (1, T_prompt); answer_ids: (1, T_ans).

        Returns dict: grad_norm, self_kernel, per-group norms, answer_ce,
        sketch (np.ndarray of sketch_dim, fp32).
        """
        model = self.model
        model.zero_grad(set_to_none=True)
        full = torch.cat([input_ids, answer_ids], dim=1)
        labels = full.clone()
        labels[:, : input_ids.shape[1]] = -100  # loss on answer tokens only
        out = model(input_ids=full, labels=labels, use_cache=False)
        loss = out.loss
        (loss * LOSS_SCALE).backward()

        norms = {}
        sq = 0.0
        sketch = np.zeros(self.sketch_dim, dtype=np.float64)
        nonfinite = 0
        for name, p in self.params.items():
            if p.grad is None:
                norms[name] = None
                continue
            g = (p.grad.detach().float() / LOSS_SCALE)
            if not torch.isfinite(g).all():
                nonfinite += int((~torch.isfinite(g)).sum())
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            norms[name] = float(g.norm())
            sq += norms[name] ** 2
            sketch += self._sketch(g, name)
        model.zero_grad(set_to_none=True)
        return {
            "answer_ce": float(loss.detach()),
            "grad_norm": float(np.sqrt(sq)),
            "self_kernel": sq,
            "group_norms": norms,
            "nonfinite_grad_elems": nonfinite,
            "sketch": sketch.astype(np.float32),
        }


def gram_from_sketches(sketches: np.ndarray) -> dict:
    """NTK Gram (approx) from sketched gradients, plus spectrum features.
    sketches: (N, sketch_dim)."""
    G = sketches @ sketches.T
    eigs = np.linalg.eigvalsh(G)
    eigs = np.clip(eigs, 0, None)[::-1]
    tot = eigs.sum()
    p = eigs / tot if tot > 0 else eigs
    ent = -(p * np.log(p + 1e-12)).sum()
    return {
        "gram": G,
        "eigs": eigs,
        "effective_rank": float(np.exp(ent)),
        "top1_share": float(eigs[0] / tot) if tot > 0 else None,
    }
