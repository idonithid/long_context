"""Prompt building and deterministic generation."""
from __future__ import annotations

import logging
import time

import torch

from .model_loader import attn_impl

logger = logging.getLogger("rank_demand.gen")


def build_prompt(tokenizer, task_input: str, answer_prefix: str = "") -> str:
    """Chat-template the RULER task input; append answer_prefix as assistant
    prefill (this mirrors RULER's model_template + answer_prefix convention).
    Falls back to a plain wrapper if the tokenizer has no chat template."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task_input}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text + answer_prefix
    logger.warning("Tokenizer has no chat template; using plain prompt wrapper.")
    return f"{task_input}\n{answer_prefix}"


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    generate_attn_impl: str = "sdpa",
) -> dict:
    """Deterministic (greedy) generation. Returns prediction text + timing +
    peak GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with attn_impl(model, generate_attn_impl):
        out = model.generate(
            input_ids=input_ids,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    new_tokens = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    peak = None
    if torch.cuda.is_available():
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return {
        "prediction_text": text,
        "generation_time_sec": round(dt, 2),
        "max_gpu_memory_gb": peak,
        "num_new_tokens": int(new_tokens.shape[0]),
    }
