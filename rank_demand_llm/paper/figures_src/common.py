"""Shared loaders for figure scripts. Reads real result JSON files."""
import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# Task orders per setting (row = train capability, col = eval capability)
ORDER = {
    "interference": ["niah_single_1", "niah_multikey_1", "vt", "cwe", "qa_1"],
    "interference_lora": ["niah_single_1", "niah_multikey_1", "vt", "cwe", "qa_1"],
    "interference_llama": ["niah_single_1", "niah_multikey_1", "vt", "cwe", "qa_1"],
    "interference_hetero": ["gsm8k", "mbpp_code", "qa_1", "niah_single_1", "cwe"],
}

LABELS = {
    "niah_single_1": "single",
    "niah_multikey_1": "multikey",
    "vt": "vt",
    "cwe": "cwe",
    "qa_1": "qa",
    "gsm8k": "gsm8k",
    "mbpp_code": "mbpp",
}


def load_matrices(setting):
    """Return (order, measured_dce[i][j], predicted_kernel[i][j]) as dicts."""
    d = os.path.join(RESULTS, setting)
    with open(os.path.join(d, "measured.json")) as f:
        meas = json.load(f)
    with open(os.path.join(d, "predicted_kernel.json")) as f:
        pred = json.load(f)
    order = ORDER[setting]
    dce = {i: {} for i in order}
    ker = {i: {} for i in order}
    for i in order:
        for j in order:
            dce[i][j] = meas[i]["post"][j]["dce"]
            ker[i][j] = pred[f"{i}->{j}"]["mean_kernel"]
    return order, dce, ker


def load_summary(setting):
    d = os.path.join(RESULTS, setting)
    with open(os.path.join(d, "summary.json")) as f:
        return json.load(f)


SETTINGS = [
    ("interference", "RULER (subset)"),
    ("interference_hetero", "Heterogeneous"),
    ("interference_lora", "LoRA"),
    ("interference_llama", "Llama-3.1-8B"),
]
