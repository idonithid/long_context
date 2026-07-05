# Method experiments

```json
{
  "exp1_selfgrad": [
    {
      "model": "Qwen2.5-7B",
      "n": 1000,
      "n_incorrect": 67,
      "auc": {
        "self_grad_norm": {
          "auc": 0.7757,
          "ci95": [
            0.7383,
            0.8133
          ]
        },
        "self_answer_ce": {
          "auc": 0.899,
          "ci95": [
            0.8699,
            0.9283
          ]
        },
        "min_token_logprob": {
          "auc": 0.8726,
          "ci95": [
            0.8389,
            0.9046
          ]
        },
        "mean_token_entropy": {
          "auc": 0.902,
          "ci95": [
            0.8748,
            0.9288
          ]
        },
        "max_token_entropy": {
          "auc": 0.874,
          "ci95": [
            0.838,
            0.9076
          ]
        }
      },
      "cv_logistic": {
        "baselines": 0.9031,
        "baselines+grad": 0.9015,
        "delta": -0.0016
      },
      "within_cell": [
        {
          "cell": "aggregation@1024",
          "n": 50,
          "auc_grad": 0.282,
          "auc_ce": 0.774
        },
        {
          "cell": "aggregation@2048",
          "n": 50,
          "auc_grad": 0.435,
          "auc_ce": 0.875
        },
        {
          "cell": "aggregation@4096",
          "n": 50,
          "auc_grad": 0.417,
          "auc_ce": 0.943
        },
        {
          "cell": "aggregation@8192",
          "n": 50,
          "auc_grad": 0.455,
          "auc_ce": 0.822
        },
        {
          "cell": "qa@1024",
          "n": 50,
          "auc_grad": 0.95,
          "auc_ce": 0.901
        },
        {
          "cell": "qa@2048",
          "n": 50,
          "auc_grad": 0.702,
          "auc_ce": 0.724
        },
        {
          "cell": "qa@4096",
          "n": 50,
          "auc_grad": 0.739,
          "auc_ce": 0.848
        },
        {
          "cell": "qa@8192",
          "n": 50,
          "auc_grad": 0.744,
          "auc_ce": 0.753
        }
      ]
    },
    {
      "model": "Llama-3.1-8B",
      "n": 983,
      "n_incorrect": 57,
      "auc": {
        "self_grad_norm": {
          "auc": 0.7579,
          "ci95": [
            0.7055,
            0.8103
          ]
        },
        "self_answer_ce": {
          "auc": 0.8912,
          "ci95": [
            0.8517,
            0.9251
          ]
        },
        "min_token_logprob": {
          "auc": 0.8462,
          "ci95": [
            0.79,
            0.8924
          ]
        },
        "mean_token_entropy": {
          "auc": 0.9055,
          "ci95": [
            0.8726,
            0.9328
          ]
        },
        "max_token_entropy": {
          "auc": 0.8918,
          "ci95": [
            0.8531,
            0.9251
          ]
        }
      },
      "cv_logistic": {
        "baselines": 0.9037,
        "baselines+grad": 0.9036,
        "delta": -0.0001
      },
      "within_cell": [
        {
          "cell": "aggregation@1024",
          "n": 50,
          "auc_grad": 0.58,
          "auc_ce": 0.673
        },
        {
          "cell": "aggregation@2048",
          "n": 50,
          "auc_grad": 0.585,
          "auc_ce": 0.718
        },
        {
          "cell": "aggregation@4096",
          "n": 50,
          "auc_grad": 0.475,
          "auc_ce": 0.752
        },
        {
          "cell": "aggregation@8192",
          "n": 50,
          "auc_grad": 0.833,
          "auc_ce": 0.843
        },
        {
          "cell": "qa@1024",
          "n": 50,
          "auc_grad": 0.646,
          "auc_ce": 0.667
        },
        {
          "cell": "qa@2048",
          "n": 50,
          "auc_grad": 0.821,
          "auc_ce": 0.793
        },
        {
          "cell": "qa@4096",
          "n": 50,
          "auc_grad": 0.761,
          "auc_ce": 0.728
        },
        {
          "cell": "qa@8192",
          "n": 43,
          "auc_grad": 0.899,
          "auc_ce": 0.837
        }
      ]
    }
  ],
  "exp2_interference": {
    "summary": {
      "spearman_kernel_vs_dce": [
        0.793,
        0.0
      ],
      "pearson_kernel_vs_dce": [
        0.9774,
        0.0
      ],
      "spearman_cos_vs_dce": [
        0.6449,
        0.0005
      ],
      "off_diag_spearman": [
        0.6797,
        0.001
      ]
    },
    "measured_dce": {
      "niah_single_1": {
        "niah_single_1": -0.00654,
        "niah_multikey_1": -0.00673,
        "vt": -0.00962,
        "cwe": -0.04189,
        "qa_1": 0.0542
      },
      "niah_multikey_1": {
        "niah_single_1": -0.00654,
        "niah_multikey_1": -0.00673,
        "vt": 0.04171,
        "cwe": 0.06521,
        "qa_1": 0.06762
      },
      "vt": {
        "niah_single_1": 0.00699,
        "niah_multikey_1": 0.00576,
        "vt": -0.32077,
        "cwe": -0.08049,
        "qa_1": -0.20655
      },
      "cwe": {
        "niah_single_1": 0.00429,
        "niah_multikey_1": -0.00139,
        "vt": -0.05788,
        "cwe": -2.45432,
        "qa_1": -0.42619
      },
      "qa_1": {
        "niah_single_1": 0.04486,
        "niah_multikey_1": 0.04229,
        "vt": 0.0587,
        "cwe": -0.48109,
        "qa_1": -4.15036
      }
    },
    "measured_dacc": {
      "niah_single_1": {
        "niah_single_1": 0.0,
        "niah_multikey_1": 0.0,
        "vt": 0.0,
        "cwe": 0.0333,
        "qa_1": 0.0
      },
      "niah_multikey_1": {
        "niah_single_1": 0.0,
        "niah_multikey_1": 0.0,
        "vt": 0.0,
        "cwe": 0.0333,
        "qa_1": 0.0
      },
      "vt": {
        "niah_single_1": 0.0,
        "niah_multikey_1": 0.0,
        "vt": 0.0,
        "cwe": -0.1333,
        "qa_1": 0.0333
      },
      "cwe": {
        "niah_single_1": 0.0,
        "niah_multikey_1": 0.0,
        "vt": 0.0,
        "cwe": -0.0667,
        "qa_1": 0.0
      },
      "qa_1": {
        "niah_single_1": 0.0,
        "niah_multikey_1": 0.0,
        "vt": 0.0,
        "cwe": -0.0333,
        "qa_1": 0.0667
      }
    },
    "predicted_kernel": {
      "niah_single_1->niah_single_1": 2.8752171993255615,
      "niah_single_1->niah_multikey_1": 1.9392040967941284,
      "niah_single_1->vt": 0.6173991560935974,
      "niah_single_1->cwe": 2.1400654315948486,
      "niah_single_1->qa_1": -0.3726539611816406,
      "niah_multikey_1->niah_single_1": 2.5678744316101074,
      "niah_multikey_1->niah_multikey_1": 4.904485702514648,
      "niah_multikey_1->vt": 1.2203210592269897,
      "niah_multikey_1->cwe": 2.5874722003936768,
      "niah_multikey_1->qa_1": -0.947766900062561,
      "vt->niah_single_1": 0.29568397998809814,
      "vt->niah_multikey_1": 0.44598668813705444,
      "vt->vt": 65.71953582763672,
      "vt->cwe": 24.47873878479004,
      "vt->qa_1": 7.985065937042236,
      "cwe->niah_single_1": 0.5364062190055847,
      "cwe->niah_multikey_1": 0.5812073349952698,
      "cwe->vt": 15.370672225952148,
      "cwe->cwe": 664.2884521484375,
      "cwe->qa_1": 116.19070434570312,
      "qa_1->niah_single_1": -0.26286473870277405,
      "qa_1->niah_multikey_1": -0.9345340728759766,
      "qa_1->vt": 8.080294609069824,
      "qa_1->cwe": 118.90218353271484,
      "qa_1->qa_1": 741.4007568359375
    },
    "baselines": {
      "ntk_kernel": {
        "all": [
          0.793,
          0.0
        ],
        "off_diag": [
          0.6797,
          0.001
        ]
      },
      "ntk_cos": {
        "all": [
          0.6449,
          0.0005
        ],
        "off_diag": [
          0.6677,
          0.0013
        ]
      },
      "grad_magnitude": {
        "all": [
          0.3825,
          0.0592
        ],
        "off_diag": [
          0.2526,
          0.2826
        ]
      },
      "embed_cos": {
        "skip_reason": "no data (embeddings missing?)"
      }
    }
  },
  "exp2_variants": {
    "interference_hetero": {
      "summary": {
        "spearman_kernel_vs_dce": [
          0.9069,
          0.0
        ],
        "pearson_kernel_vs_dce": [
          0.9216,
          0.0
        ],
        "spearman_cos_vs_dce": [
          0.7415,
          0.0
        ],
        "off_diag_spearman": [
          0.8737,
          0.0
        ]
      },
      "measured_dce": {
        "gsm8k": {
          "gsm8k": -0.56731,
          "mbpp_code": -0.53038,
          "qa_1": -0.67561,
          "niah_single_1": 0.01535,
          "cwe": -0.19109
        },
        "mbpp_code": {
          "gsm8k": -0.25937,
          "mbpp_code": -1.88764,
          "qa_1": -0.74997,
          "niah_single_1": 0.01526,
          "cwe": -0.48174
        },
        "qa_1": {
          "gsm8k": -0.13585,
          "mbpp_code": -0.28823,
          "qa_1": -4.14968,
          "niah_single_1": 0.04632,
          "cwe": -0.48059
        },
        "niah_single_1": {
          "gsm8k": 0.01767,
          "mbpp_code": -0.07082,
          "qa_1": 0.05406,
          "niah_single_1": -0.00654,
          "cwe": -0.04222
        },
        "cwe": {
          "gsm8k": -0.09733,
          "mbpp_code": -0.34986,
          "qa_1": -0.42344,
          "niah_single_1": 0.004,
          "cwe": -2.45522
        }
      },
      "measured_dacc": {
        "gsm8k": {
          "gsm8k": null,
          "mbpp_code": null,
          "qa_1": null,
          "niah_single_1": null,
          "cwe": null
        },
        "mbpp_code": {
          "gsm8k": null,
          "mbpp_code": null,
          "qa_1": null,
          "niah_single_1": null,
          "cwe": null
        },
        "qa_1": {
          "gsm8k": null,
          "mbpp_code": null,
          "qa_1": null,
          "niah_single_1": null,
          "cwe": null
        },
        "niah_single_1": {
          "gsm8k": null,
          "mbpp_code": null,
          "qa_1": null,
          "niah_single_1": null,
          "cwe": null
        },
        "cwe": {
          "gsm8k": null,
          "mbpp_code": null,
          "qa_1": null,
          "niah_single_1": null,
          "cwe": null
        }
      },
      "predicted_kernel": {
        "gsm8k->gsm8k": 34.188743591308594,
        "gsm8k->mbpp_code": 19.76131820678711,
        "gsm8k->qa_1": 27.765933990478516,
        "gsm8k->niah_single_1": -0.5029459595680237,
        "gsm8k->cwe": 4.303237438201904,
        "mbpp_code->gsm8k": 17.914674758911133,
        "mbpp_code->mbpp_code": 106.32560729980469,
        "mbpp_code->qa_1": 45.81180953979492,
        "mbpp_code->niah_single_1": -0.23622092604637146,
        "mbpp_code->cwe": 32.15664291381836,
        "qa_1->gsm8k": 23.95083236694336,
        "qa_1->mbpp_code": 50.66804885864258,
        "qa_1->qa_1": 734.4278564453125,
        "qa_1->niah_single_1": -1.5644066333770752,
        "qa_1->cwe": 101.7071304321289,
        "niah_single_1->gsm8k": -0.43803170323371887,
        "niah_single_1->mbpp_code": 0.06265164166688919,
        "niah_single_1->qa_1": -1.9876066446304321,
        "niah_single_1->niah_single_1": 2.8499033451080322,
        "niah_single_1->cwe": 0.249252587556839,
        "cwe->gsm8k": 13.015181541442871,
        "cwe->mbpp_code": 41.37749099731445,
        "cwe->qa_1": 106.40061950683594,
        "cwe->niah_single_1": -0.654062032699585,
        "cwe->cwe": 663.3701782226562
      },
      "baselines": {
        "ntk_kernel": {
          "all": [
            0.9069,
            0.0
          ],
          "off_diag": [
            0.8737,
            0.0
          ]
        },
        "ntk_cos": {
          "all": [
            0.7415,
            0.0
          ],
          "off_diag": [
            0.791,
            0.0
          ]
        },
        "grad_magnitude": {
          "all": [
            0.7408,
            0.0
          ],
          "off_diag": [
            0.8045,
            0.0
          ]
        },
        "embed_cos": {
          "all": [
            -0.04,
            0.8494
          ],
          "off_diag": [
            -0.6241,
            0.0033
          ]
        }
      }
    },
    "interference_lora": {
      "summary": {
        "spearman_kernel_vs_dce": [
          0.7768,
          0.0
        ],
        "pearson_kernel_vs_dce": [
          0.88,
          0.0
        ],
        "spearman_cos_vs_dce": [
          0.6264,
          0.0008
        ],
        "off_diag_spearman": [
          0.618,
          0.0037
        ]
      },
      "measured_dce": {
        "niah_single_1": {
          "niah_single_1": -0.00654,
          "niah_multikey_1": -0.00673,
          "vt": 0.01528,
          "cwe": 0.15247,
          "qa_1": 0.14293
        },
        "niah_multikey_1": {
          "niah_single_1": -0.00654,
          "niah_multikey_1": -0.00673,
          "vt": 0.0384,
          "cwe": 0.20298,
          "qa_1": 0.13879
        },
        "vt": {
          "niah_single_1": 0.01622,
          "niah_multikey_1": 0.00452,
          "vt": -0.32103,
          "cwe": 0.09552,
          "qa_1": -0.33691
        },
        "cwe": {
          "niah_single_1": 0.00453,
          "niah_multikey_1": -0.00257,
          "vt": -0.13295,
          "cwe": -1.44861,
          "qa_1": -1.1026
        },
        "qa_1": {
          "niah_single_1": 0.10595,
          "niah_multikey_1": 0.00856,
          "vt": -0.06214,
          "cwe": -1.52055,
          "qa_1": -4.36328
        }
      },
      "measured_dacc": {
        "niah_single_1": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": -0.0333,
          "qa_1": 0.0
        },
        "niah_multikey_1": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": -0.0333,
          "qa_1": 0.0
        },
        "vt": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": -0.8,
          "qa_1": -0.0333
        },
        "cwe": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": 0.1667,
          "qa_1": -0.0333
        },
        "qa_1": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": -0.4,
          "qa_1": 0.1
        }
      },
      "predicted_kernel": {
        "niah_single_1->niah_single_1": 2.3104710578918457,
        "niah_single_1->niah_multikey_1": 1.5110113620758057,
        "niah_single_1->vt": -0.28505560755729675,
        "niah_single_1->cwe": 0.08623860776424408,
        "niah_single_1->qa_1": -0.6455853581428528,
        "niah_multikey_1->niah_single_1": 1.82119619846344,
        "niah_multikey_1->niah_multikey_1": 3.303157091140747,
        "niah_multikey_1->vt": 0.013488976284861565,
        "niah_multikey_1->cwe": 1.4463543891906738,
        "niah_multikey_1->qa_1": 0.6726239323616028,
        "vt->niah_single_1": -0.46419474482536316,
        "vt->niah_multikey_1": -0.2117864340543747,
        "vt->vt": 64.78033447265625,
        "vt->cwe": 19.139873504638672,
        "vt->qa_1": 8.70913028717041,
        "cwe->niah_single_1": -0.269805908203125,
        "cwe->niah_multikey_1": 0.9358251690864563,
        "cwe->vt": 13.309161186218262,
        "cwe->cwe": 650.2139282226562,
        "cwe->qa_1": 87.71892547607422,
        "qa_1->niah_single_1": -0.4896923899650574,
        "qa_1->niah_multikey_1": -0.09551432728767395,
        "qa_1->vt": 3.0830559730529785,
        "qa_1->cwe": 115.03238677978516,
        "qa_1->qa_1": 736.8578491210938
      },
      "baselines": {
        "ntk_kernel": {
          "all": [
            0.7768,
            0.0
          ],
          "off_diag": [
            0.618,
            0.0037
          ]
        },
        "ntk_cos": {
          "all": [
            0.6264,
            0.0008
          ],
          "off_diag": [
            0.5504,
            0.0119
          ]
        },
        "grad_magnitude": {
          "all": [
            0.3794,
            0.0614
          ],
          "off_diag": [
            0.2827,
            0.2272
          ]
        },
        "embed_cos": {
          "all": [
            0.0158,
            0.9403
          ],
          "off_diag": [
            -0.2977,
            0.2023
          ]
        }
      }
    },
    "interference_llama": {
      "summary": {
        "spearman_kernel_vs_dce": [
          0.8192,
          0.0
        ],
        "pearson_kernel_vs_dce": [
          0.7456,
          0.0
        ],
        "spearman_cos_vs_dce": [
          0.73,
          0.0
        ],
        "off_diag_spearman": [
          0.6917,
          0.0007
        ]
      },
      "measured_dce": {
        "niah_single_1": {
          "niah_single_1": -0.24601,
          "niah_multikey_1": -0.16374,
          "vt": -0.07152,
          "cwe": 0.0415,
          "qa_1": -0.0252
        },
        "niah_multikey_1": {
          "niah_single_1": -0.24569,
          "niah_multikey_1": -0.16407,
          "vt": -0.01632,
          "cwe": 0.02483,
          "qa_1": -0.02
        },
        "vt": {
          "niah_single_1": -0.0834,
          "niah_multikey_1": -0.04317,
          "vt": -0.33847,
          "cwe": 0.09309,
          "qa_1": 0.07854
        },
        "cwe": {
          "niah_single_1": -0.05411,
          "niah_multikey_1": -0.06537,
          "vt": 0.03469,
          "cwe": -0.87678,
          "qa_1": -0.22297
        },
        "qa_1": {
          "niah_single_1": -0.10351,
          "niah_multikey_1": -0.06902,
          "vt": -0.01785,
          "cwe": -0.22321,
          "qa_1": -2.06606
        }
      },
      "measured_dacc": {
        "niah_single_1": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": 0.1333,
          "qa_1": -0.0333
        },
        "niah_multikey_1": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": 0.1333,
          "qa_1": -0.0333
        },
        "vt": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": -0.8333,
          "qa_1": 0.0
        },
        "cwe": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": 0.1333,
          "qa_1": 0.0
        },
        "qa_1": {
          "niah_single_1": 0.0,
          "niah_multikey_1": 0.0,
          "vt": 0.0,
          "cwe": 0.0667,
          "qa_1": 0.0333
        }
      },
      "predicted_kernel": {
        "niah_single_1->niah_single_1": 312.35516357421875,
        "niah_single_1->niah_multikey_1": 229.97015380859375,
        "niah_single_1->vt": 22.06943702697754,
        "niah_single_1->cwe": -68.59485626220703,
        "niah_single_1->qa_1": 12.91842269897461,
        "niah_multikey_1->niah_single_1": 230.08834838867188,
        "niah_multikey_1->niah_multikey_1": 270.7086486816406,
        "niah_multikey_1->vt": 17.010526657104492,
        "niah_multikey_1->cwe": -76.12599182128906,
        "niah_multikey_1->qa_1": 12.790755271911621,
        "vt->niah_single_1": 22.11078643798828,
        "vt->niah_multikey_1": 25.521154403686523,
        "vt->vt": 70.416259765625,
        "vt->cwe": -17.704912185668945,
        "vt->qa_1": -13.260766983032227,
        "cwe->niah_single_1": -99.64913177490234,
        "cwe->niah_multikey_1": -132.1710968017578,
        "cwe->vt": -27.145221710205078,
        "cwe->cwe": 969.8645629882812,
        "cwe->qa_1": 125.96876525878906,
        "qa_1->niah_single_1": 12.217448234558105,
        "qa_1->niah_multikey_1": 13.340195655822754,
        "qa_1->vt": -10.37936782836914,
        "qa_1->cwe": 117.55318450927734,
        "qa_1->qa_1": 590.6998901367188
      },
      "baselines": {
        "ntk_kernel": {
          "all": [
            0.8192,
            0.0
          ],
          "off_diag": [
            0.6917,
            0.0007
          ]
        },
        "ntk_cos": {
          "all": [
            0.73,
            0.0
          ],
          "off_diag": [
            0.591,
            0.0061
          ]
        },
        "grad_magnitude": {
          "all": [
            0.0885,
            0.6741
          ],
          "off_diag": [
            0.0932,
            0.6958
          ]
        },
        "embed_cos": {
          "all": [
            0.4377,
            0.0287
          ],
          "off_diag": [
            0.0241,
            0.9198
          ]
        }
      }
    }
  }
}
```
