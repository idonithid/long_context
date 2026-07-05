# Rank Demand — run report

- command: `/home/initzan/long_context/rank_demand_llm/rank_demand/eval_ruler.py --config configs/ruler_llama3p1_8b.yaml --mode main --output_dir results/llama_run`
- model: `meta-llama/Llama-3.1-8B-Instruct`
- mode: main | tasks: ['niah_single_1', 'niah_multikey_1', 'vt', 'cwe', 'qa_1'] | lengths: [1024, 2048, 4096, 8192]
- samples: 1000 rows (50 per task x length)
- hardware: {"cuda": true, "device_index": 0, "name": "Quadro RTX 8000", "total_gb": 47.6, "free_gb": 21.3, "compute_capability": "7.5"}
- failures: 0 hard errors, 20 OOM-skipped metric blocks, parse statuses: {'ok': 993, 'generation_oom': 7}

## Accuracy by task family x context length

| task_family       |   1024 |   2048 |   4096 |     8192 |
|:------------------|-------:|-------:|-------:|---------:|
| aggregation       |   0.8  |   0.7  |   0.94 | 0.8      |
| multi_hop_tracing |   1    |   1    |   1    | 1        |
| multi_needle      |   1    |   1    |   1    | 1        |
| qa                |   0.96 |   0.92 |   0.92 | 0.790698 |
| single_needle     |   1    |   1    |   1    | 1        |

## Mean rank/entropy metrics by task family

| task_family       |   hidden_r_eff_mid |   hidden_r_eff_tokennorm_mid |   attn_block_r_eff_mean |   attn_entropy_mean |   attn_stable_rank_mean |
|:------------------|-------------------:|-----------------------------:|------------------------:|--------------------:|------------------------:|
| aggregation       |              82.25 |                        81.52 |                    8.13 |                3.27 |                    5.46 |
| multi_hop_tracing |              35.82 |                        43.47 |                    7.35 |                3.31 |                    6.89 |
| multi_needle      |             122.04 |                       336.97 |                    6.45 |                3.27 |                    5.7  |
| qa                |              87.85 |                       281.07 |                    4.21 |                2.83 |                    3.32 |
| single_needle     |              28.35 |                        30.32 |                    7.74 |                3.26 |                    7.25 |

## Regression (correct ~ predictors, standardized, one-hot family)

```json
{
  "n": 487,
  "models": {
    "base: length+family": {
      "auc_in_sample": 0.8943,
      "coefficients": {
        "context_length": -0.262,
        "fam_multi_hop_tracing": 2.0982,
        "fam_multi_needle": 2.2346,
        "fam_qa": 1.0309,
        "fam_single_needle": 2.2346
      },
      "auc_cv5": 0.8716,
      "auc_cv5_ci95": [
        0.8252,
        0.9131
      ]
    },
    "+entropy": {
      "auc_in_sample": 0.8939,
      "coefficients": {
        "context_length": 0.1764,
        "entropy": -0.5999,
        "fam_multi_hop_tracing": 2.214,
        "fam_multi_needle": 2.2821,
        "fam_qa": 0.5696,
        "fam_single_needle": 2.2269
      },
      "auc_cv5": 0.8722,
      "auc_cv5_ci95": [
        0.8297,
        0.9089
      ]
    },
    "+entropy+rank": {
      "auc_in_sample": 0.8954,
      "coefficients": {
        "context_length": 0.4275,
        "entropy": 0.2122,
        "hidden_rank": -1.484,
        "hidden_rank_tokennorm": 0.2769,
        "attention_rank": 0.0575,
        "fam_multi_hop_tracing": 1.1511,
        "fam_multi_needle": 1.4849,
        "fam_qa": -1.0802,
        "fam_single_needle": 0.9718
      },
      "auc_cv5": 0.8739,
      "auc_cv5_ci95": [
        0.8338,
        0.9083
      ]
    }
  },
  "auc_metric_used": "auc_cv5",
  "delta_auc_rank_beyond_entropy": 0.0017
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
  "n_ntk": 990,
  "n_skipped": 10,
  "self_kernel_by_family": {
    "aggregation": {
      "mean": 2999.3746,
      "std": 2399.2969
    },
    "multi_hop_tracing": {
      "mean": 167.2243,
      "std": 166.6375
    },
    "multi_needle": {
      "mean": 521.0548,
      "std": 329.4154
    },
    "qa": {
      "mean": 3381.9999,
      "std": 3518.9996
    },
    "single_needle": {
      "mean": 492.8317,
      "std": 565.3775
    }
  },
  "answer_ce_by_family": {
    "aggregation": 1.811,
    "multi_hop_tracing": 0.329,
    "multi_needle": 0.2394,
    "qa": 2.085,
    "single_needle": 0.2621
  },
  "grad_norm_correct": 30.0895,
  "grad_norm_incorrect": 62.0523,
  "gram_effective_rank": 127.97,
  "gram_top1_share": 0.1014,
  "mean_cos_within_family": 0.4095,
  "mean_cos_across_family": 0.0487
}
```


## Preliminary verdict

- A. Rank vs entropy: adding rank on top of entropy changes in-sample AUC by +0.002 (rank adds little beyond entropy at this N).
- B. Highest hidden effective rank (mid layer): **multi_needle** (122.0); lowest: single_needle (28.4).
- C. Hidden r_eff (mid): correct=69.2 vs incorrect=100.9 (descriptive; N small).
- D. Evidence survival gap (evidence - random, k=32, mid layer): -0.224 on average (no preferential survival).
- E. Continuation call: see console summary / next-command printout.
