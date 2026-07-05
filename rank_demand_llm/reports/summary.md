# Rank Demand — run report

- command: `/home/initzan/long_context/rank_demand_llm/rank_demand/eval_ruler.py --config configs/ruler_qwen2p5_7b.yaml --mode smoke --tasks niah_single_1 vt --lengths 1024 --num_samples 5 --output_dir results/smoke_run`
- model: `Qwen/Qwen2.5-7B-Instruct`
- mode: smoke | tasks: ['niah_single_1', 'vt'] | lengths: [1024]
- samples: 10 rows (5 per task x length)
- hardware: {"cuda": true, "device_index": 0, "name": "Quadro RTX 8000", "total_gb": 47.6, "free_gb": 47.4, "compute_capability": "7.5"}
- failures: 0 hard errors, 0 OOM-skipped metric blocks, parse statuses: {'ok': 10}

## Accuracy by task family x context length

| task_family       |   1024 |
|:------------------|-------:|
| multi_hop_tracing |      1 |
| single_needle     |      1 |

## Mean rank/entropy metrics by task family

| task_family       |   hidden_r_eff_mid |   hidden_r_eff_tokennorm_mid |   attn_block_r_eff_mean |   attn_entropy_mean |   attn_stable_rank_mean |
|:------------------|-------------------:|-----------------------------:|------------------------:|--------------------:|------------------------:|
| multi_hop_tracing |               3.15 |                        18.24 |                    6.11 |                4.99 |                   23.44 |
| single_needle     |               2.26 |                         7.32 |                    4.83 |                5.19 |                   19.01 |

## Regression (correct ~ predictors, standardized, one-hot family)

```json
{
  "n": 10,
  "skip_reason": "only 10 usable samples; need >=20 for regression"
}
```
_AUC is in-sample; treat as descriptive until the main run has enough N for CV._

## Plots

![accuracy_vs_length](figures/accuracy_vs_length.png)
![hidden_rank_vs_layer](figures/hidden_rank_vs_layer.png)
![attn_rank_vs_layer](figures/attn_rank_vs_layer.png)
![entropy_vs_rank](figures/entropy_vs_rank.png)
![correct_vs_rank](figures/correct_vs_rank.png)
![evidence_survival](figures/evidence_survival.png)
![ntk_gram](figures/ntk_gram.png)

## Empirical NTK per-prompt features

```json
{
  "n_ntk": 10,
  "n_skipped": 0,
  "self_kernel_by_family": {
    "multi_hop_tracing": {
      "mean": 104.4156,
      "std": 21.8698
    },
    "single_needle": {
      "mean": 34.0752,
      "std": 75.4723
    }
  },
  "answer_ce_by_family": {
    "multi_hop_tracing": 0.3241,
    "single_needle": 0.0126
  },
  "gram_effective_rank": 4.77,
  "gram_top1_share": 0.3557,
  "mean_cos_within_family": 0.4586,
  "mean_cos_across_family": 0.0152
}
```


## Preliminary verdict

- A. Regression skipped: only 10 usable samples; need >=20 for regression — rank-vs-entropy question needs the main run.
- B. Highest hidden effective rank (mid layer): **multi_hop_tracing** (3.1); lowest: single_needle (2.3).
- C. Correct/incorrect rank split not computable (degenerate correctness or missing metrics).
- D. Evidence survival gap (evidence - random, k=32, mid layer): -0.116 on average (no preferential survival).
- E. Continuation call: see console summary / next-command printout.
