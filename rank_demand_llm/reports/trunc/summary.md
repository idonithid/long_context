# Rank Demand — run report

- command: `/home/initzan/long_context/rank_demand_llm/rank_demand/eval_ruler.py --config configs/ruler_qwen2p5_7b.yaml --mode main --lengths 1024 --output_dir results/trunc_run --run_attention_truncation`
- model: `Qwen/Qwen2.5-7B-Instruct`
- mode: main | tasks: ['niah_single_1', 'niah_multikey_1', 'vt', 'cwe', 'qa_1'] | lengths: [1024]
- samples: 250 rows (50 per task x length)
- hardware: {"cuda": true, "device_index": 0, "name": "Quadro RTX 8000", "total_gb": 47.6, "free_gb": 47.4, "compute_capability": "7.5"}
- failures: 0 hard errors, 0 OOM-skipped metric blocks, parse statuses: {'ok': 250}

## Accuracy by task family x context length

| task_family       |   1024 |
|:------------------|-------:|
| aggregation       |   0.86 |
| multi_hop_tracing |   1    |
| multi_needle      |   1    |
| qa                |   0.94 |
| single_needle     |   1    |

## Mean rank/entropy metrics by task family

| task_family       |   hidden_r_eff_mid |   hidden_r_eff_tokennorm_mid |   attn_block_r_eff_mean |   attn_entropy_mean |   attn_stable_rank_mean |
|:------------------|-------------------:|-----------------------------:|------------------------:|--------------------:|------------------------:|
| aggregation       |              12.62 |                        66.48 |                    5.97 |                4.52 |                   19.9  |
| multi_hop_tracing |               3.24 |                        18.82 |                    6.13 |                4.98 |                   23.36 |
| multi_needle      |              22.58 |                       200.69 |                    7.47 |                4.4  |                   25.57 |
| qa                |               8.52 |                        69.94 |                    2.81 |                3.64 |                    9.29 |
| single_needle     |               2.22 |                         6.88 |                    4.87 |                5.19 |                   19.08 |

## Regression (correct ~ predictors, standardized, one-hot family)

```json
{
  "n": 250,
  "models": {
    "base: length+family": {
      "auc_in_sample": 0.8542,
      "coefficients": {
        "context_length": 0.0,
        "fam_multi_hop_tracing": 1.1483,
        "fam_multi_needle": 1.1483,
        "fam_qa": 0.1096,
        "fam_single_needle": 1.1483
      },
      "auc_cv5": 0.8369,
      "auc_cv5_ci95": [
        0.7596,
        0.9166
      ]
    },
    "+entropy": {
      "auc_in_sample": 0.8208,
      "coefficients": {
        "context_length": 0.0,
        "entropy": 0.5547,
        "fam_multi_hop_tracing": 0.8925,
        "fam_multi_needle": 1.199,
        "fam_qa": 0.7699,
        "fam_single_needle": 0.7956
      },
      "auc_cv5": 0.8479,
      "auc_cv5_ci95": [
        0.7632,
        0.9288
      ]
    },
    "+entropy+rank": {
      "auc_in_sample": 0.8812,
      "coefficients": {
        "context_length": 0.0,
        "entropy": 0.5141,
        "hidden_rank": -1.3411,
        "hidden_rank_tokennorm": 1.0893,
        "attention_rank": 0.0167,
        "fam_multi_hop_tracing": 0.6124,
        "fam_multi_needle": 1.1144,
        "fam_qa": 0.3384,
        "fam_single_needle": 0.5515
      },
      "auc_cv5": 0.8592,
      "auc_cv5_ci95": [
        0.7751,
        0.9408
      ]
    }
  },
  "auc_metric_used": "auc_cv5",
  "delta_auc_rank_beyond_entropy": 0.0113
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
![truncation_error](figures/truncation_error.png)

## Low-rank attention truncation (offline diagnostic)

```json
{
  "n_samples": 250,
  "rel_error_by_family_k": {
    "aggregation": {
      "4": 0.4791,
      "8": 0.4064,
      "16": 0.3529,
      "32": 0.2945,
      "64": 0.2191,
      "128": 0.1494
    },
    "multi_hop_tracing": {
      "4": 0.3943,
      "8": 0.3384,
      "16": 0.2838,
      "32": 0.2198,
      "64": 0.1578,
      "128": 0.1008
    },
    "multi_needle": {
      "4": 0.5064,
      "8": 0.463,
      "16": 0.4094,
      "32": 0.3487,
      "64": 0.2708,
      "128": 0.1805
    },
    "qa": {
      "4": 0.4121,
      "8": 0.3524,
      "16": 0.2816,
      "32": 0.2004,
      "64": 0.1179,
      "128": 0.0443
    },
    "single_needle": {
      "4": 0.3086,
      "8": 0.2523,
      "16": 0.2006,
      "32": 0.1561,
      "64": 0.1081,
      "128": 0.0709
    }
  },
  "rel_error_k16_correct": 0.3045,
  "rel_error_k16_incorrect": 0.3345,
  "mannwhitney_p": 0.1355
}
```


## Preliminary verdict

- A. Rank vs entropy: adding rank on top of entropy changes in-sample AUC by +0.011 (rank adds little beyond entropy at this N).
- B. Highest hidden effective rank (mid layer): **multi_needle** (22.6); lowest: single_needle (2.2).
- C. Hidden r_eff (mid): correct=9.7 vs incorrect=13.5 (descriptive; N small).
- D. Evidence survival gap (evidence - random, k=32, mid layer): -0.028 on average (no preferential survival).
- E. Continuation call: see console summary / next-command printout.
