# Rank Demand — run report

- command: `/home/initzan/long_context/rank_demand_llm/rank_demand/eval_ruler.py --config configs/ruler_qwen2p5_7b.yaml --mode main --output_dir results/main_run`
- model: `Qwen/Qwen2.5-7B-Instruct`
- mode: main | tasks: ['niah_single_1', 'niah_multikey_1', 'vt', 'cwe', 'qa_1'] | lengths: [1024, 2048, 4096, 8192]
- samples: 1000 rows (50 per task x length)
- hardware: {"cuda": true, "device_index": 0, "name": "Quadro RTX 8000", "total_gb": 47.6, "free_gb": 47.4, "compute_capability": "7.5"}
- failures: 0 hard errors, 0 OOM-skipped metric blocks, parse statuses: {'ok': 1000}

## Accuracy by task family x context length

| task_family       |   1024 |   2048 |   4096 |   8192 |
|:------------------|-------:|-------:|-------:|-------:|
| aggregation       |   0.86 |   0.74 |   0.88 |   0.62 |
| multi_hop_tracing |   1    |   1    |   1    |   1    |
| multi_needle      |   1    |   1    |   1    |   1    |
| qa                |   0.94 |   0.9  |   0.88 |   0.84 |
| single_needle     |   1    |   1    |   1    |   1    |

## Mean rank/entropy metrics by task family

| task_family       |   hidden_r_eff_mid |   hidden_r_eff_tokennorm_mid |   attn_block_r_eff_mean |   attn_entropy_mean |   attn_stable_rank_mean |
|:------------------|-------------------:|-----------------------------:|------------------------:|--------------------:|------------------------:|
| aggregation       |              21.98 |                        71.79 |                    8.15 |                4.88 |                   15.32 |
| multi_hop_tracing |               2.59 |                        12.22 |                    7.14 |                5.52 |                   16.78 |
| multi_needle      |             100.54 |                       268.92 |                   10.34 |                4.7  |                   21.9  |
| qa                |             146.35 |                       205.02 |                    6.37 |                4.1  |                   12.98 |
| single_needle     |               2.11 |                         7.28 |                    5.63 |                5.73 |                   13.68 |

## Regression (correct ~ predictors, standardized, one-hot family)

```json
{
  "n": 500,
  "models": {
    "base: length+family": {
      "auc_in_sample": 0.8708,
      "coefficients": {
        "context_length": -0.3105,
        "fam_multi_hop_tracing": 2.0272,
        "fam_multi_needle": 2.0272,
        "fam_qa": 0.5151,
        "fam_single_needle": 2.0272
      },
      "auc_cv5": 0.8527,
      "auc_cv5_ci95": [
        0.8055,
        0.8966
      ]
    },
    "+entropy": {
      "auc_in_sample": 0.8705,
      "coefficients": {
        "context_length": -0.4455,
        "entropy": 0.2553,
        "fam_multi_hop_tracing": 1.8761,
        "fam_multi_needle": 2.0718,
        "fam_qa": 0.7511,
        "fam_single_needle": 1.8322
      },
      "auc_cv5": 0.8523,
      "auc_cv5_ci95": [
        0.8049,
        0.8949
      ]
    },
    "+entropy+rank": {
      "auc_in_sample": 0.8806,
      "coefficients": {
        "context_length": -0.1587,
        "entropy": 0.9306,
        "hidden_rank": -0.197,
        "hidden_rank_tokennorm": 1.0008,
        "attention_rank": -1.0905,
        "fam_multi_hop_tracing": 1.6276,
        "fam_multi_needle": 1.8068,
        "fam_qa": 0.4551,
        "fam_single_needle": 1.2269
      },
      "auc_cv5": 0.8506,
      "auc_cv5_ci95": [
        0.8077,
        0.8918
      ]
    }
  },
  "auc_metric_used": "auc_cv5",
  "delta_auc_rank_beyond_entropy": -0.0017
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
  "n_ntk": 1000,
  "n_skipped": 0,
  "self_kernel_by_family": {
    "aggregation": {
      "mean": 2623.812,
      "std": 1846.8863
    },
    "multi_hop_tracing": {
      "mean": 133.5844,
      "std": 67.5262
    },
    "multi_needle": {
      "mean": 188.8197,
      "std": 299.1789
    },
    "qa": {
      "mean": 6374.8842,
      "std": 7555.1717
    },
    "single_needle": {
      "mean": 8.0697,
      "std": 25.1289
    }
  },
  "answer_ce_by_family": {
    "aggregation": 3.0073,
    "multi_hop_tracing": 0.3671,
    "multi_needle": 0.0533,
    "qa": 4.4092,
    "single_needle": 0.0074
  },
  "grad_norm_correct": 26.1833,
  "grad_norm_incorrect": 63.4843,
  "gram_effective_rank": 173.17,
  "gram_top1_share": 0.053,
  "mean_cos_within_family": 0.2067,
  "mean_cos_across_family": 0.0201
}
```


## Preliminary verdict

- A. Rank vs entropy: adding rank on top of entropy changes in-sample AUC by -0.002 (rank adds little beyond entropy at this N).
- B. Highest hidden effective rank (mid layer): **qa** (146.3); lowest: single_needle (2.1).
- C. Hidden r_eff (mid): correct=52.9 vs incorrect=79.4 (descriptive; N small).
- D. Evidence survival gap (evidence - random, k=32, mid layer): -0.071 on average (no preferential survival).
- E. Continuation call: see console summary / next-command printout.
