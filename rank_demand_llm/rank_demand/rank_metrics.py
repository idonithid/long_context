"""Rank / spectrum metrics for hidden states and attention matrices.

All functions accept torch tensors (any device); heavy linear algebra runs in
float32. Every metric that can be skipped returns a dict that may contain
`skip_reason` instead of silently omitting values.

Definitions
-----------
effective rank (Roy & Vetterli 2007):
    p_i = lambda_i / sum_j lambda_j ;  r_eff = exp(-sum_i p_i log p_i)
stable rank: ||M||_F^2 / ||M||_2^2
participation ratio: (sum_i lambda_i)^2 / sum_i lambda_i^2
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

logger = logging.getLogger("rank_demand.metrics")

EPS = 1e-12


# ---------------------------------------------------------------------------
# Spectrum-level primitives
# ---------------------------------------------------------------------------

def robust_eigvalsh(C: torch.Tensor) -> torch.Tensor:
    """eigvalsh with CPU-float64 fallback: cuSOLVER's syevd fails to converge
    on ill-conditioned fp32 inputs (error 99x), notably on Turing GPUs."""
    try:
        return torch.linalg.eigvalsh(C)
    except Exception as e:
        logger.debug("GPU eigvalsh failed (%s); retrying on CPU float64", e)
    C64 = C.detach().cpu().double()
    C64 = 0.5 * (C64 + C64.T)  # enforce exact symmetry
    return torch.linalg.eigvalsh(C64).to(C.dtype)


def robust_eigh(C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """eigh (values, vectors) with the same CPU-float64 fallback."""
    try:
        return torch.linalg.eigh(C)
    except Exception as e:
        logger.debug("GPU eigh failed (%s); retrying on CPU float64", e)
    dev = C.device
    C64 = C.detach().cpu().double()
    C64 = 0.5 * (C64 + C64.T)
    vals, vecs = torch.linalg.eigh(C64)
    return vals.to(dtype=C.dtype, device=dev), vecs.to(dtype=C.dtype, device=dev)


def effective_rank_from_eigs(eigs: torch.Tensor) -> float:
    """exp(entropy) of the normalized eigenvalue distribution. Eigs clipped at 0."""
    eigs = torch.clamp(eigs.double(), min=0.0)
    total = eigs.sum()
    if total <= EPS:
        return 0.0
    p = eigs / total
    ent = -(p * torch.log(p + EPS)).sum()
    return float(torch.exp(ent))


def participation_ratio_from_eigs(eigs: torch.Tensor) -> float:
    eigs = torch.clamp(eigs.double(), min=0.0)
    s1 = eigs.sum()
    s2 = (eigs ** 2).sum()
    if s2 <= EPS:
        return 0.0
    return float(s1 ** 2 / s2)


def robust_svdvals(M: torch.Tensor) -> torch.Tensor:
    """svdvals with CPU-float64 fallback (cuSOLVER can fail to converge)."""
    try:
        return torch.linalg.svdvals(M)
    except Exception as e:
        logger.debug("GPU svdvals failed (%s); retrying on CPU float64", e)
    return torch.linalg.svdvals(M.detach().cpu().double()).to(M.dtype)


def stable_rank(M: torch.Tensor) -> float:
    """||M||_F^2 / ||M||_2^2 via largest singular value."""
    M = M.float()
    fro2 = float((M ** 2).sum())
    if fro2 <= EPS:
        return 0.0
    smax2 = float(robust_svdvals(M)[0]) ** 2
    if smax2 <= EPS:
        return 0.0
    return fro2 / smax2


def matrix_effective_rank(M: torch.Tensor) -> float:
    """Effective rank of a general matrix via singular values (p_i from s_i^2)."""
    s = robust_svdvals(M.float())
    return effective_rank_from_eigs(s ** 2)


# ---------------------------------------------------------------------------
# A. Hidden-state metrics
# ---------------------------------------------------------------------------

def hidden_state_metrics(
    H: torch.Tensor,
    center: bool = True,
    max_tokens: int = 8192,
) -> dict:
    """Token-covariance spectrum metrics for H in R^{T x d}.

    Returns dict with r_eff, r_eff_normalized, stable_rank, participation_ratio,
    spectrum (np.ndarray), subsample_stride, num_tokens_used.
    """
    T, d = H.shape
    stride = 1
    if T > max_tokens:
        stride = int(np.ceil(T / max_tokens))
        H = H[::stride]
        logger.debug("hidden-state subsample stride=%d (T=%d)", stride, T)
    H = H.float()
    # fp16 inference can overflow to inf in late layers (Qwen massive
    # activations); sanitize but record how much was touched.
    finite = torch.isfinite(H)
    nonfinite_frac = float((~finite).sum()) / H.numel()
    if nonfinite_frac > 0:
        logger.warning("hidden states contain %.2e non-finite fraction; clamping",
                       nonfinite_frac)
        H = torch.nan_to_num(H, nan=0.0, posinf=6.5e4, neginf=-6.5e4)
    if center:
        H = H - H.mean(dim=0, keepdim=True)
    Tn = H.shape[0]
    # covariance in R^{d x d}; if T < d the min(T,d) trick via Gram is cheaper
    if Tn < d:
        C = (H @ H.T) / Tn  # Gram matrix, same nonzero eigenvalues
    else:
        C = (H.T @ H) / Tn
    eigs = robust_eigvalsh(C)
    eigs = torch.clamp(eigs, min=0.0)
    sr = stable_rank(H)
    r_eff = effective_rank_from_eigs(eigs)
    # Token-normalized variant: LLM "massive activation" outlier tokens can
    # dominate the covariance and pin r_eff near 1; normalizing each token to
    # unit norm keeps directional diversity visible. Reported alongside, never
    # instead of, the primary (spec) metric.
    Hn = H / (H.norm(dim=1, keepdim=True) + EPS)
    if Tn < d:
        Cn = (Hn @ Hn.T) / Tn
    else:
        Cn = (Hn.T @ Hn) / Tn
    r_eff_tokennorm = effective_rank_from_eigs(
        torch.clamp(robust_eigvalsh(Cn), min=0.0))
    return {
        "r_eff": r_eff,
        "r_eff_normalized": r_eff / d,
        "r_eff_tokennorm": r_eff_tokennorm,
        "stable_rank": sr,
        "participation_ratio": participation_ratio_from_eigs(eigs),
        "spectrum": eigs.flip(0).cpu().numpy(),  # descending
        "subsample_stride": stride,
        "num_tokens_used": int(Tn),
        "hidden_dim": int(d),
        "centered": bool(center),
        "nonfinite_fraction": nonfinite_frac,
    }


# ---------------------------------------------------------------------------
# B. Attention-matrix metrics
# ---------------------------------------------------------------------------

def attention_entropy(A: torch.Tensor) -> float:
    """Mean over query rows of -sum_j A_ij log A_ij. A rows assumed ~normalized."""
    A = A.float()
    ent = -(A * torch.log(A + EPS)).sum(dim=-1)
    return float(ent.mean())


def max_attention_mass(A: torch.Tensor) -> float:
    return float(A.float().max(dim=-1).values.mean())


def block_compress(A: torch.Tensor, block_size: int) -> torch.Tensor:
    """Sum-pool A (T x T) into B x B blocks, then row-normalize.

    Trailing partial blocks are kept (padded implicitly by summing fewer cells).
    """
    T = A.shape[0]
    B = int(np.ceil(T / block_size))
    A = A.float()
    # pad to multiple of block_size
    pad = B * block_size - T
    if pad > 0:
        A = torch.nn.functional.pad(A, (0, pad, 0, pad))
    blocks = A.reshape(B, block_size, B, block_size).sum(dim=(1, 3))
    row_sums = blocks.sum(dim=-1, keepdim=True)
    blocks = blocks / (row_sums + EPS)
    return blocks


def attention_matrix_metrics(
    A: torch.Tensor,
    block_size: int = 64,
    exact_rank_max_tokens: int = 1024,
) -> dict:
    """Metrics for one attention matrix A in R^{T x T} (rows ~sum to 1).

    Exact effective rank only for T <= exact_rank_max_tokens; otherwise skipped
    with skip_reason.
    """
    T = A.shape[0]
    out: dict = {
        "T": int(T),
        "entropy": attention_entropy(A),
        "max_mass": max_attention_mass(A),
        "stable_rank": stable_rank(A),
    }
    blocks = block_compress(A, block_size)
    out["block_size"] = int(block_size)
    out["num_blocks"] = int(blocks.shape[0])
    out["block_r_eff"] = matrix_effective_rank(blocks)
    out["block_matrix"] = blocks.cpu().numpy()
    if T <= exact_rank_max_tokens:
        out["exact_r_eff"] = matrix_effective_rank(A)
    else:
        out["exact_r_eff"] = None
        out["exact_r_eff_skip_reason"] = (
            f"T={T} > exact_rank_max_tokens={exact_rank_max_tokens}"
        )
    return out


def score_matrix_metrics(S: torch.Tensor, block_size: int = 64) -> dict:
    """Pre-softmax score matrix diagnostics (stable rank + block effective rank).

    S is not row-stochastic, so block compression uses |S| sum-pooling without
    row normalization; effective rank computed from singular values directly.
    """
    T = S.shape[0]
    out: dict = {"T": int(T), "stable_rank": stable_rank(S)}
    B = int(np.ceil(T / block_size))
    pad = B * block_size - T
    Sf = S.float().abs()
    if pad > 0:
        Sf = torch.nn.functional.pad(Sf, (0, pad, 0, pad))
    blocks = Sf.reshape(B, block_size, B, block_size).mean(dim=(1, 3))
    out["block_r_eff"] = matrix_effective_rank(blocks)
    out["block_size"] = int(block_size)
    return out


# ---------------------------------------------------------------------------
# D. Evidence survival
# ---------------------------------------------------------------------------

def evidence_survival(
    H: torch.Tensor,
    evidence_idx: Sequence[int],
    ks: Sequence[int] = (8, 16, 32, 64, 128, 256),
    center: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    """S_E(k) = ||H_E V_k V_k^T||_F^2 / ||H_E||_F^2 with V_k top-k eigvecs of
    token covariance; plus the same for count-matched random non-evidence rows.
    """
    T, d = H.shape
    evidence_idx = sorted({i for i in evidence_idx if 0 <= i < T})
    if not evidence_idx:
        return {"skip_reason": "no valid evidence positions"}
    H = torch.nan_to_num(H.float(), nan=0.0, posinf=6.5e4, neginf=-6.5e4)
    if center:
        H = H - H.mean(dim=0, keepdim=True)
    C = (H.T @ H) / T
    eigs, V = robust_eigh(C)  # ascending
    V = V.flip(-1)  # descending eigenvalue order

    ks_use = sorted({min(int(k), d) for k in ks if k >= 1})
    ev = torch.tensor(evidence_idx, dtype=torch.long, device=H.device)
    H_E = H[ev]

    if rng is None:
        rng = np.random.default_rng(0)
    non_ev = np.setdiff1d(np.arange(T), np.asarray(evidence_idx))
    n = min(len(evidence_idx), len(non_ev))
    rand_idx = rng.choice(non_ev, size=n, replace=False) if n > 0 else np.array([], dtype=int)
    H_R = H[torch.tensor(rand_idx, dtype=torch.long, device=H.device)] if n > 0 else None

    def survival(Hs: torch.Tensor) -> list[float]:
        denom = float((Hs ** 2).sum()) + EPS
        vals = []
        for k in ks_use:
            Vk = V[:, :k]
            proj = Hs @ Vk
            vals.append(float((proj ** 2).sum()) / denom)
        return vals

    out = {
        "ks": ks_use,
        "evidence_survival": survival(H_E),
        "num_evidence_tokens": len(evidence_idx),
    }
    if H_R is not None and len(rand_idx) > 0:
        out["random_survival"] = survival(H_R)
        out["num_random_tokens"] = int(n)
    else:
        out["random_survival"] = None
        out["random_skip_reason"] = "no non-evidence tokens available"
    return out


# ---------------------------------------------------------------------------
# Low-rank reconstruction diagnostic (offline intervention proxy)
# ---------------------------------------------------------------------------

def attention_truncation_error(
    A: torch.Tensor,
    V: torch.Tensor,
    ks: Sequence[int] = (4, 8, 16, 32, 64, 128),
) -> dict:
    """Reconstruction error ||AV - A_k V||_F / ||AV||_F where A_k is the rank-k
    SVD truncation of A. A: (T,T), V: (T, d_head)."""
    A = A.float()
    V = V.float()
    T = A.shape[0]
    AV = A @ V
    denom = float(torch.linalg.norm(AV)) + EPS
    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    except Exception as e:
        return {"skip_reason": f"svd_failed: {e}"}
    out_ks, errs = [], []
    for k in ks:
        k = min(int(k), T)
        Ak = (U[:, :k] * S[:k]) @ Vh[:k]
        err = float(torch.linalg.norm(Ak @ V - AV)) / denom
        out_ks.append(k)
        errs.append(err)
    return {"ks": out_ks, "rel_error": errs}
