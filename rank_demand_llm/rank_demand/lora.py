"""Minimal LoRA for the robustness experiment (no peft dependency).

Wraps selected nn.Linear modules with y = Wx + (alpha/r) * B(Ax).
A: (r, in), Kaiming-init; B: (out, r), zero-init -> identity at start.
Adapters live in the base dtype (fp16 here); training uses fp32 master
copies + loss scaling (see finetune_lora), mirroring the subset trainer —
direct fp16 Adam destroys the model.
"""
from __future__ import annotations

import logging
import math

import torch
from torch import nn

logger = logging.getLogger("rank_demand.lora")

LOSS_SCALE = 1024.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        dt, dev = base.weight.dtype, base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, dtype=dt, device=dev))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        return y + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(model, target_names=("q_proj", "v_proj"), layers=None,
                r: int = 16, alpha: float = 32.0) -> dict[str, nn.Parameter]:
    """Wrap target projections in (all or selected) decoder layers.
    Returns dict of LoRA parameters. layers=None -> all layers."""
    params: dict[str, nn.Parameter] = {}
    decoder = model.model.layers
    idxs = range(len(decoder)) if layers is None else layers
    for li in idxs:
        attn = decoder[li].self_attn
        for name in target_names:
            base = getattr(attn, name, None)
            if base is None or isinstance(base, LoRALinear):
                continue
            wrapped = LoRALinear(base, r=r, alpha=alpha)
            setattr(attn, name, wrapped)
            params[f"L{li}.{name}.A"] = wrapped.lora_A
            params[f"L{li}.{name}.B"] = wrapped.lora_B
    n = sum(p.numel() for p in params.values())
    logger.info("LoRA injected: %d tensors, %.2fM params (r=%d)",
                len(params), n / 1e6, r)
    return params


def remove_lora(model):
    """Unwrap all LoRALinear modules (restores exact pre-injection weights)."""
    removed = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(attn, name, None)
            if isinstance(mod, LoRALinear):
                setattr(attn, name, mod.base)
                removed += 1
    logger.info("LoRA removed from %d modules", removed)


def reset_lora(params: dict[str, nn.Parameter]):
    """Re-zero B and re-init A -> adapter is identity again."""
    with torch.no_grad():
        for name, p in params.items():
            if name.endswith(".B"):
                p.zero_()
            else:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
