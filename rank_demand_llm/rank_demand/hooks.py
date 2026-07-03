"""Measurement forward pass: hidden states, attention weights, Q/K captures.

Strategy
--------
One prefill forward with `output_hidden_states=True` (+ `output_attentions=True`
when T <= exact_attention_max_tokens and the model runs eager attention).
Hidden states for ALL layers are cheap (L+1 x T x d, fp16). Attention weights
are the expensive part: transformers materializes all layers' (n_heads, T, T)
maps when output_attentions=True, ~n_layers*n_heads*T^2*2 bytes (Qwen2.5-7B at
T=2048: ~6.6 GB) — acceptable at the default cap, dangerous above; guarded.

Q/K pre-softmax capture uses forward hooks on q_proj/k_proj of the selected
layers, applies the model's own rotary embedding, and forms
S = QK^T / sqrt(d_head) per selected head (GQA-aware).
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger("rank_demand.hooks")


def select_layers(num_layers: int, fractions=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[int]:
    idx = sorted({min(num_layers - 1, round(f * (num_layers - 1))) for f in fractions})
    return idx


def select_heads(num_heads: int, T: int, all_heads_max_tokens: int,
                 sampled_heads=(0, "middle", "last")) -> list[int]:
    if T <= all_heads_max_tokens:
        return list(range(num_heads))
    out = []
    for h in sampled_heads:
        if h == "middle":
            out.append(num_heads // 2)
        elif h == "last":
            out.append(num_heads - 1)
        else:
            out.append(int(h))
    return sorted(set(out))


class QKCapture:
    """Capture q_proj/k_proj outputs on selected layers during one forward."""

    def __init__(self, model, layer_indices: list[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        layers = self.model.model.layers
        for li in self.layer_indices:
            attn = layers[li].self_attn
            self._handles.append(attn.q_proj.register_forward_hook(self._mk(li, self.q)))
            self._handles.append(attn.k_proj.register_forward_hook(self._mk(li, self.k)))
        return self

    @staticmethod
    def _mk(li: int, store: dict):
        def hook(_mod, _inp, out):
            store[li] = out.detach()
        return hook

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        return False

    def score_matrix(self, layer_idx: int, head: int) -> torch.Tensor | None:
        """RoPE-applied pre-softmax scores S = QK^T/sqrt(d_head) for one head.

        Returns (T, T) tensor (no causal mask applied — we analyze the raw
        score structure; caller may mask). None if capture missing.
        """
        if layer_idx not in self.q or layer_idx not in self.k:
            return None
        cfg = self.model.config
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        q = self.q[layer_idx]  # (1, T, n_heads*d_head)
        k = self.k[layer_idx]  # (1, T, n_kv*d_head)
        T = q.shape[1]
        d_head = q.shape[-1] // n_heads
        q = q.view(1, T, n_heads, d_head).transpose(1, 2).float()
        k = k.view(1, T, n_kv, d_head).transpose(1, 2).float()
        # apply the model's rotary embedding (Qwen2/Llama style)
        try:
            rot = self.model.model.rotary_emb
            pos = torch.arange(T, device=q.device).unsqueeze(0)
            cos, sin = rot(q, pos)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        except Exception as e:
            logger.warning("RoPE application failed (%s); using raw QK^T "
                           "(diagnostic approximation).", e)
        kv_head = head // (n_heads // n_kv)
        S = (q[0, head] @ k[0, kv_head].T) / math.sqrt(d_head)
        return S

    def clear(self):
        self.q.clear()
        self.k.clear()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (1, H, T, d); cos/sin: (1, T, d)
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


@torch.no_grad()
def measurement_forward(
    model,
    input_ids: torch.Tensor,
    collect_attention: bool,
    qk_layers: list[int] | None = None,
):
    """One prefill pass. Returns (hidden_states tuple, attentions tuple|None,
    QKCapture|None). Caller is responsible for OOM guarding.

    hidden_states: tuple of (1, T, d), length n_layers+1 (embeddings first).
    attentions: tuple of (1, n_heads, T, T) per layer, or None.
    """
    capture = None
    if qk_layers:
        capture = QKCapture(model, qk_layers).__enter__()
    try:
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=collect_attention,
            use_cache=False,
            return_dict=True,
        )
    finally:
        if capture is not None:
            capture.__exit__(None, None, None)
    attentions = out.attentions if collect_attention else None
    return out.hidden_states, attentions, capture
