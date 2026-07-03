"""RULER data conversion + evidence-position recovery.

RULER's official generators (external/RULER/scripts/data/prepare.py) emit
jsonl rows: {index, input, outputs, length, answer_prefix, ...}. We convert to
our schema and recover evidence spans by exact string search.

Evidence semantics per task family:
- niah_*: the needle values (`outputs`) appear verbatim in the context ->
  status "exact" when all found.
- vt: answers are variable names whose assignment chain is the evidence; all
  occurrences of each answer variable and the queried value are evidence ->
  status "approximate" (occurrences over-cover the minimal chain).
- cwe / fwe: evidence is distributed over the whole list -> "unavailable".
- qa_1: gold answer string searched in the documents; "exact" if found,
  otherwise "unavailable" (answers are sometimes paraphrased).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("rank_demand.data")


def find_all(hay: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), hay)]


def recover_evidence(task: str, input_text: str, outputs: list[str],
                     metadata: dict) -> dict:
    """Return evidence_text, evidence_positions (char spans in input_text),
    evidence_position_status."""
    spans: list[tuple[int, int]] = []
    evidence_text: list[str] = []
    status = "unavailable"

    if task.startswith("niah"):
        evidence_text = list(outputs)
        found = [find_all(input_text, o) for o in outputs]
        spans = [s for f in found for s in f]
        if all(f for f in found):
            status = "exact"
        elif any(f for f in found):
            status = "approximate"
    elif task == "vt":
        # answers are variable names; queried value also evidence if present
        evidence_text = list(outputs)
        qv = metadata.get("query")
        if qv:
            evidence_text.append(str(qv))
        found = [find_all(input_text, e) for e in evidence_text]
        spans = [s for f in found for s in f]
        status = "approximate" if spans else "unavailable"
    elif task.startswith("qa"):
        evidence_text = list(outputs)
        found = [find_all(input_text, o) for o in outputs]
        spans = [s for f in found for s in f]
        status = "exact" if all(f for f in found) and spans else (
            "approximate" if spans else "unavailable")
    else:  # cwe, fwe: distributed evidence
        status = "unavailable"

    return {
        "evidence_text": evidence_text,
        "evidence_positions": spans,
        "evidence_position_status": status,
    }


def convert_ruler_row(row: dict, task: str, task_family: str,
                      target_context_length: int, sample_idx: int) -> dict:
    """RULER generator row -> our sample schema."""
    outputs = row["outputs"]
    if isinstance(outputs, str):
        outputs = [outputs]
    input_text = row["input"]
    meta = {k: v for k, v in row.items() if k not in ("input", "outputs")}
    ev = recover_evidence(task, input_text, outputs, meta)
    return {
        "sample_id": f"{task}_{target_context_length}_{sample_idx}",
        "ruler_task": task,
        "task_family": task_family,
        "target_context_length": target_context_length,
        "prompt_text": input_text,
        "answer_prefix": row.get("answer_prefix", ""),
        "expected_answer": outputs,
        **ev,
        "metadata": meta,
    }


def evidence_token_positions(tokenizer, full_prompt: str,
                             evidence_text: list[str]) -> list[int]:
    """Map evidence strings to token indices of the FINAL (chat-templated)
    prompt using the fast tokenizer's offset mapping."""
    enc = tokenizer(full_prompt, return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"]
    char_spans = []
    for ev in evidence_text:
        char_spans.extend(find_all(full_prompt, ev))
    tok_idx: set[int] = set()
    for cs, ce in char_spans:
        for ti, (ts, te) in enumerate(offsets):
            if ts < ce and te > cs:  # overlap
                tok_idx.add(ti)
    return sorted(tok_idx)
