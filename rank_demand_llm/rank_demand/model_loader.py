"""Model/tokenizer loading with dtype auto-detection and attention-impl switching.

Design notes
------------
- Flash attention and SDPA kernels do NOT return attention weights, so exact
  attention extraction requires attn_implementation="eager". We load the model
  once with eager and temporarily switch config._attn_implementation to "sdpa"
  for the generation pass (transformers >= 4.48 dispatches per-forward via
  config). If the runtime switch is unsupported we stay on eager (slower,
  same numbers) and log it.
- bf16 requires compute capability >= 8.0 for native speed. On Turing (sm_75,
  e.g. Quadro RTX 8000) bf16 matmul is emulated and slow -> auto mode picks
  fp16 there.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("rank_demand.model")


def _register_eager_fp32():
    """Register an eager attention variant that computes QK^T scores in fp32.

    Rationale: Qwen2.5 in fp16 overflows the pre-softmax scores (massive
    activations -> |QK^T| > 65504 -> inf -> NaN after softmax), which poisons
    late-layer hidden states in the measurement pass. SDPA doesn't hit this
    (fused fp32 accumulation), so generation is unaffected — only the eager
    extraction path needs the upcast. Weights are returned in the query dtype
    to keep the retained (layers x heads x T x T) tensor small.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    try:
        if "eager_fp32" in ALL_ATTENTION_FUNCTIONS:  # dict-like registry
            return
    except TypeError:
        pass

    def eager_fp32_attention_forward(module, query, key, value, attention_mask,
                                     scaling: float, dropout: float = 0.0,
                                     **kwargs):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups).float()
        value_states = repeat_kv(value, module.num_key_value_groups).float()
        attn_weights = torch.matmul(query.float(),
                                    key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask.float()
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states).to(query.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights.to(query.dtype)

    ALL_ATTENTION_FUNCTIONS.register("eager_fp32", eager_fp32_attention_forward)
    logger.info("registered eager_fp32 attention (fp32 QK^T scores)")


def _register_sdpa_nogqa():
    """Register an SDPA variant that repeats KV heads instead of enable_gqa.

    On sm_75, torch's memory-efficient SDPA kernel rejects `enable_gqa=True`
    (transformers 4.57 passes it for GQA models), so SDPA silently falls back
    to the math backend and materializes full fp32 T x T scores per layer —
    6.8 GiB/layer at 8k tokens, which OOMs generation. Repeating KV heads up
    front keeps the mem-efficient kernel eligible (0.4 GB peak at 8k).
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    try:
        if "sdpa_nogqa" in ALL_ATTENTION_FUNCTIONS:
            return
    except TypeError:
        pass

    def sdpa_nogqa_attention_forward(module, query, key, value, attention_mask,
                                     dropout: float = 0.0, scaling=None,
                                     is_causal=None, **kwargs):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        if hasattr(module, "num_key_value_groups"):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
            # now n_heads == n_kv_heads -> upstream fn won't set enable_gqa
        return sdpa_attention_forward(module, query, key, value,
                                      attention_mask, dropout=dropout,
                                      scaling=scaling, is_causal=is_causal,
                                      **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa_nogqa", sdpa_nogqa_attention_forward)
    logger.info("registered sdpa_nogqa attention (KV-repeat, mem-efficient kernel eligible)")


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0):
            picked = torch.bfloat16
        elif torch.cuda.is_available():
            picked = torch.float16
            logger.warning(
                "GPU compute capability %s < 8.0: no native bf16 -> using fp16. "
                "(Override with dtype: bf16 in config if you insist.)",
                torch.cuda.get_device_capability(0),
            )
        else:
            picked = torch.float32
        logger.info("dtype=auto resolved to %s", picked)
        return picked
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def load_model_and_tokenizer(
    model_id: str,
    dtype: str = "auto",
    load_4bit: bool = False,
    attn_implementation: str = "eager",
):
    """Load once with eager attention (needed for attention-weight extraction).

    Warning: flash-attention/sdpa kernels do not return attention weights;
    eager is mandatory for the measurement pass.
    """
    torch_dtype = resolve_dtype(dtype)
    if attn_implementation in ("eager", "eager_fp32") and torch_dtype == torch.float16:
        _register_eager_fp32()
        attn_implementation = "eager_fp32"
    _register_sdpa_nogqa()
    if (attn_implementation == "sdpa" and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) < (8, 0)):
        attn_implementation = "sdpa_nogqa"  # see _register_sdpa_nogqa docstring
    logger.info("Loading %s (dtype=%s, attn=%s, 4bit=%s)",
                model_id, torch_dtype, attn_implementation, load_4bit)
    if not attn_implementation.startswith("eager"):
        logger.warning(
            "attn_implementation=%s will NOT return attention weights; "
            "attention-rank metrics need eager.", attn_implementation
        )

    kwargs: dict = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            logger.error("bitsandbytes not installed; ignoring load_4bit=True")

    # Qwen2.5 does not need trust_remote_code with modern transformers; try
    # without first, fall back with it for models that require it.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            logger.warning("Model requires trust_remote_code=True; retrying.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            raise
    model.eval()
    logger.info("Model loaded: %d layers, %d heads, hidden %d, kv heads %s",
                model.config.num_hidden_layers, model.config.num_attention_heads,
                model.config.hidden_size,
                getattr(model.config, "num_key_value_heads", "n/a"))
    return model, tokenizer


@contextmanager
def attn_impl(model, impl: str):
    """Temporarily switch attention implementation (e.g. eager -> sdpa for
    generation). Falls back to no-op with a log line if unsupported."""
    cfg = model.config
    # sm<80: plain sdpa + GQA dispatches enable_gqa, which the mem-efficient
    # kernel rejects -> math backend -> O(T^2) fp32 scores. Use KV-repeat variant.
    if (impl == "sdpa" and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) < (8, 0)):
        impl = "sdpa_nogqa"
    old = getattr(cfg, "_attn_implementation", None)
    switched = False
    try:
        if old is not None and impl != old:
            try:
                cfg._attn_implementation = impl
                switched = True
            except Exception as e:  # pragma: no cover
                logger.warning("Cannot switch attn implementation to %s (%s); "
                               "staying on %s (slower but identical numbers).",
                               impl, e, old)
        yield model
    finally:
        if switched:
            cfg._attn_implementation = old
