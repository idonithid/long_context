"""Sanity checks for rank metrics. Run: python tests/test_rank_metrics.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_demand.rank_metrics import (  # noqa: E402
    attention_entropy,
    attention_matrix_metrics,
    attention_truncation_error,
    block_compress,
    effective_rank_from_eigs,
    evidence_survival,
    hidden_state_metrics,
    matrix_effective_rank,
    participation_ratio_from_eigs,
    stable_rank,
)

torch.manual_seed(0)
np.random.seed(0)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def main():
    d = 256

    # --- effective rank ---
    eye_eigs = torch.ones(d)
    check("identity eigs -> r_eff == d",
          abs(effective_rank_from_eigs(eye_eigs) - d) < 1e-3,
          f"(got {effective_rank_from_eigs(eye_eigs):.2f}, want {d})")

    rank1_eigs = torch.zeros(d); rank1_eigs[0] = 5.0
    check("rank-1 eigs -> r_eff ~= 1",
          abs(effective_rank_from_eigs(rank1_eigs) - 1.0) < 1e-3,
          f"(got {effective_rank_from_eigs(rank1_eigs):.4f})")

    # --- stable rank ---
    I = torch.eye(d)
    check("identity matrix stable rank == d", abs(stable_rank(I) - d) < 1e-2)
    r1 = torch.outer(torch.randn(d), torch.randn(d))
    check("rank-1 matrix stable rank ~= 1", abs(stable_rank(r1) - 1.0) < 1e-3)
    check("stable rank scale-invariant",
          abs(stable_rank(3.7 * r1) - stable_rank(r1)) < 1e-4)

    # --- participation ratio ---
    check("identity PR == d", abs(participation_ratio_from_eigs(eye_eigs) - d) < 1e-3)
    check("rank-1 PR == 1", abs(participation_ratio_from_eigs(rank1_eigs) - 1.0) < 1e-3)

    # --- matrix effective rank ---
    check("identity matrix_effective_rank == d",
          abs(matrix_effective_rank(I) - d) < 1e-2)
    check("rank-1 matrix_effective_rank ~= 1",
          abs(matrix_effective_rank(r1) - 1.0) < 1e-2)

    # --- hidden state metrics ---
    T = 512
    H_iso = torch.randn(T, d)  # isotropic -> high rank
    m_iso = hidden_state_metrics(H_iso)
    check("isotropic hidden states have high r_eff", m_iso["r_eff"] > 0.6 * d,
          f"(r_eff={m_iso['r_eff']:.1f}/{d})")
    H_low = torch.randn(T, 3) @ torch.randn(3, d)  # rank 3
    m_low = hidden_state_metrics(H_low)
    check("rank-3 hidden states have r_eff <= ~3", m_low["r_eff"] < 4.0,
          f"(r_eff={m_low['r_eff']:.2f})")

    # subsampling path
    m_sub = hidden_state_metrics(torch.randn(1000, 32), max_tokens=100)
    check("subsampling logs stride", m_sub["subsample_stride"] == 10,
          f"(stride={m_sub['subsample_stride']}, used={m_sub['num_tokens_used']})")

    # --- attention: uniform vs one-hot ---
    Tq = 256
    A_uniform = torch.full((Tq, Tq), 1.0 / Tq)
    A_onehot = torch.eye(Tq)
    ent_u = attention_entropy(A_uniform)
    ent_o = attention_entropy(A_onehot)
    check("uniform attention entropy == log T",
          abs(ent_u - np.log(Tq)) < 1e-3, f"(got {ent_u:.3f}, log T={np.log(Tq):.3f})")
    check("one-hot attention entropy ~= 0", ent_o < 1e-3, f"(got {ent_o:.5f})")

    mu = attention_matrix_metrics(A_uniform, block_size=32)
    mo = attention_matrix_metrics(A_onehot, block_size=32)
    check("uniform attention is rank-1 (stable rank ~1) despite max entropy",
          mu["stable_rank"] < 1.01, f"(sr={mu['stable_rank']:.3f})")
    check("one-hot attention is full rank (stable rank == T)",
          abs(mo["stable_rank"] - Tq) < 1e-2, f"(sr={mo['stable_rank']:.1f})")
    check("uniform block r_eff ~= 1", mu["block_r_eff"] < 1.05,
          f"(got {mu['block_r_eff']:.3f})")
    check("one-hot block r_eff == num blocks",
          abs(mo["block_r_eff"] - mo["num_blocks"]) < 1e-2,
          f"(got {mo['block_r_eff']:.2f}, blocks={mo['num_blocks']})")
    check("exact r_eff computed for small T", mo["exact_r_eff"] is not None)

    big = attention_matrix_metrics(torch.full((1500, 1500), 1 / 1500.0),
                                   block_size=64, exact_rank_max_tokens=1024)
    check("exact r_eff skipped for T>1024 with reason",
          big["exact_r_eff"] is None and "skip_reason" in str(big),
          f"({big.get('exact_r_eff_skip_reason')})")

    # block_compress rows sum to 1
    bc = block_compress(torch.softmax(torch.randn(200, 200), dim=-1), 64)
    check("block-compressed rows sum to 1",
          torch.allclose(bc.sum(-1), torch.ones(bc.shape[0]), atol=1e-4))

    # --- evidence survival ---
    # construct H where evidence tokens live in the top-1 principal direction
    v = torch.randn(d); v /= v.norm()
    H = 0.1 * torch.randn(T, d)
    ev_idx = list(range(0, 50))
    H[ev_idx] += 10.0 * v
    surv = evidence_survival(H, ev_idx, ks=[1, 8, 64])
    check("evidence survival increases with k",
          all(a <= b + 1e-6 for a, b in zip(surv["evidence_survival"],
                                            surv["evidence_survival"][1:])))
    check("planted evidence survives at k=1 more than random tokens",
          surv["evidence_survival"][0] > (surv["random_survival"][0] + 0.2),
          f"(ev={surv['evidence_survival'][0]:.3f}, rand={surv['random_survival'][0]:.3f})")

    surv_empty = evidence_survival(H, [])
    check("empty evidence -> skip_reason", "skip_reason" in surv_empty)

    # --- truncation error ---
    A = torch.softmax(torch.randn(128, 128), dim=-1)
    Vm = torch.randn(128, 64)
    tr = attention_truncation_error(A, Vm, ks=[1, 8, 128])
    check("truncation error decreases with k",
          tr["rel_error"][0] >= tr["rel_error"][-1] - 1e-6, f"({tr['rel_error']})")
    check("full-rank truncation error ~= 0", tr["rel_error"][-1] < 1e-4,
          f"(got {tr['rel_error'][-1]:.2e})")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All rank-metric sanity checks passed.")


if __name__ == "__main__":
    main()
