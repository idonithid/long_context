# rank_demand_llm

**Research question.** Which long-context tasks induce high-rank
attention/representation structure in open-source LLMs, and does rank predict
correctness *beyond* context length, task type, evidence position, and
attention entropy?

The object of study is **rank demand**: whether different RULER task families
(single-needle, multi-needle, multi-hop tracing, aggregation, QA) induce
measurably different attention/hidden-state rank. This is explicitly *not* a
claim that "long context = low rank bias".

**Key confound.** Attention rank correlates with attention entropy/diffuseness,
but they are not the same thing (uniform attention has maximal entropy and rank
1; one-hot diagonal attention has zero entropy and full rank — see
`tests/test_rank_metrics.py`). Entropy controls are computed everywhere and
the regression ablations test whether rank adds predictive value beyond entropy.

## Why full attention rank is memory-limited

A full attention matrix at length T costs O(T²) per head. Qwen2.5-7B has
28 layers × 28 heads; materializing all maps at T=8192 would need
28·28·8192²·2B ≈ 105 GB. Policy:

| context | attention metrics |
|---|---|
| T ≤ 2048 (default cap) | exact per-head: entropy, max mass, stable rank, block-compressed effective rank; exact effective rank of full A only for T ≤ 1024 |
| T ≤ 4096 | only with explicit `--allow_exact_attention_4096` |
| T ≥ 8192 | attention skipped (recorded with `skip_reason`); hidden-state metrics only |

Hidden-state metrics (T×d) are cheap at all lengths and computed everywhere
(with logged stride subsampling above 8192 tokens). Only spectra, scalar ranks,
entropies, block matrices, and survival curves are stored — never full
attention matrices (unless `store_attention` is enabled).

Attention-weight extraction requires eager attention — flash attention / SDPA
kernels do not return attention weights. The measurement pass runs a custom
`eager_fp32` implementation (registered at load time when weights are fp16):
Qwen2.5's massive activations overflow fp16 QK^T scores (inf → NaN through
softmax, poisoning late-layer hidden states), so pre-softmax scores are
computed in fp32. The generation pass switches to SDPA, which does not hit the
overflow.

## Hardware note

This box has Quadro RTX 8000 (Turing, sm_75): no native bf16, no flash-attn.
`dtype: auto` therefore resolves to **fp16** here and bf16 on Ampere+.

## Reproduce

```bash
bash scripts/setup_env.sh                     # conda env `rank_demand` (py3.10)
bash scripts/download_ruler.sh                # clone NVIDIA/RULER + essays + SQuAD
python scripts/prepare_ruler_data.py --config configs/ruler_qwen2p5_7b.yaml --mode smoke
bash scripts/run_smoke_test.sh                # 2 families x 5 samples @ 1024
bash scripts/run_main_ruler.sh                # 5 families x 50 x {1k,2k,4k,8k} in tmux
bash scripts/make_report.sh results/main_run reports/main
```

Data generation uses the official `external/RULER/scripts/data/prepare.py`
(HF tokenizer, `model_template_type base`); prompts are then wrapped with the
Qwen chat template and RULER's `answer_prefix` is appended as assistant
prefill, mirroring RULER's own model-template convention. Scoring ports
RULER's `string_match_all` / `string_match_part`.

## Outputs

- `results/<run>/results.jsonl` — one row per sample: prediction, correctness,
  parse status, timing, peak GPU memory, metric summary, skip records.
- `results/<run>/metrics/<sample_id>.npz` — spectra, block matrices, survival curves.
- `reports/summary.md` + `reports/figures/*.png` — tables, 5 main plots,
  regression ablations (`correct ~ length+family` → `+entropy` → `+entropy+rank`),
  preliminary verdicts A–E.

## Metrics

- **Hidden-state rank** (per selected layer: 0, ¼, ½, ¾, final): effective rank
  `exp(H(λ))` of token covariance, normalized r_eff/d, stable rank, participation ratio.
- **Attention rank** (selected layers × heads, T ≤ cap): row entropy, max mass,
  stable rank, block-compressed effective rank (64-token blocks), exact
  effective rank (T ≤ 1024).
- **Pre-softmax scores**: RoPE-applied S = QK᷀ᵀ/√d for selected heads (stable +
  block rank).
- **Evidence survival**: ‖H_E V_k V_kᵀ‖²_F/‖H_E‖²_F for evidence tokens vs
  count-matched random tokens, k ∈ {8…256}.
- **Empirical NTK per-prompt features** (`scripts/run_ntk_features.py`): gradient
  of the teacher-forced answer cross-entropy w.r.t. a small parameter subset
  (q_proj/v_proj of the selected layers, ~9M params), per prompt. Features:
  gradient norm / self-kernel k(x,x), per-layer norms, and a CountSketch of the
  gradient so the cross-sample NTK Gram (spectrum, effective rank, within- vs
  across-family cosine, family block structure) is computable without storing
  full gradients. Uses loss scaling against fp16 underflow and gradient
  checkpointing for memory.
- **Low-rank truncation diagnostic** (`--run_attention_truncation`, T ≤ 1024):
  offline ‖AV − A_kV‖/‖AV‖ for k ∈ {4…128}. A live attention-truncation
  intervention (replacing AV with A_kV in the forward pass) is a TODO; the
  offline diagnostic avoids invasive monkeypatching of Qwen attention.

## Evidence positions

RULER doesn't export evidence spans uniformly; we recover them by exact string
search: needle values (niah — `exact`), answer-variable occurrences (vt —
`approximate`), gold answers (qa — `exact`/`unavailable`), aggregation tasks
(`unavailable`). Status is stored per sample; token positions are recomputed
against the final chat-templated prompt via fast-tokenizer offsets.

## Known limitations

- In-sample AUC at smoke scale is descriptive only; cross-validation needs the
  main-run N.
- Attention entropy over a causal T×T map mixes row lengths (early queries see
  few keys); comparisons are therefore within-length, and length is always a
  regression covariate.
- vt evidence spans over-cover the minimal assignment chain (all occurrences of
  answer variables are marked).
- Block-compressed rank depends on block size (64 default); it is a bounded
  proxy, not the exact rank, above T=1024.
- Single model (Qwen2.5-7B-Instruct); Llama-3.1-8B replication is planned but
  not run.
