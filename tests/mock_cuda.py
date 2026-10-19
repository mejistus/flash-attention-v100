# ======================================================================================
# mock_cuda.py — GPU-free fake `flash_attn_2_cuda` extension (pure torch, CPU).
#
# Purpose
# -------
# `flash_attn/flash_attn_interface.py` does `import flash_attn_2_cuda` at module
# import time.  This module fabricates that extension in sys.modules so the *python
# wrapper* can be exercised end-to-end on a machine with no NVIDIA GPU.
#
# Scope / fidelity contract
# -------------------------
#  EXACT (mirrored from kernel/*.cu + include/mat_mul.h, Python side):
#    - function names, argument order and return tuple arity of the pybind API
#    - output dtypes/shapes (out fp16 like q, lse fp32, dmask empty, dense (B,H,M,N)
#      or varlen (T_Q,H_Q,max_seqlen_k) like the kernels allocate)
#    - GQA head mapping  kv_head = q_head // (H_Q // H_K)
#    - causal offset rule  keep j <= i + (N - M)   (seqlen_offset = seqlen_k-seqlen_q)
#    - window rule         keep  i+off-win_l <= j <= i+off+win_r
#    - alibi               score += slope * -|i - (j - off_feat)| , off_feat = off if
#                          (causal or window) else 0       [mat_mul.h SEQLEN_OFFSET rule]
#    - softcap             score = softcap * tanh(score / softcap)
#    - C++ clamp mirrors: M==1 (no alibi) -> causal off; win >= N -> -1
#    - N==0 -> out 0, lse = +inf (official empty-KV sentinel); fully-masked rows
#      within a non-empty KV -> out 0, lse = -inf (NEG_INF handling)
#    - ONLY D in {16,32,64,128,256} dispatchable  -> RuntimeError("Unsupported D: ...")
#    - kvcache: in-place cache update at cache_seqlens[b]; attention over
#      cache_seqlen + T_new; cache_seqlens=None -> attend the FULL cache
#      (cache_seqlen = k_cache.size(1), upstream semantics / research/07 F-3);
#      causal offset aligns q to the END of the KV run; int cache_seqlens conversion
#      is the wrapper's job (we receive a tensor).
#  APPROXIMATE / NOT MODELLED (documented, tests must not rely on them):
#    - dropout: mock computes NO dropout.  dmask is returned all-ones (keep-all) and
#      rng_state zeros.  ALLOW_DROPOUT=False by default -> dropout_p>0 raises.
#    - device: everything runs on CPU even though the real op requires CUDA.
#    - dtype: accepts fp16 (kernel-faithful) and fp32 (convenience; real op rejects).
#    - rotary_cos/sin, leftpad_k, paged block_table in fwd_kvcache -> NotImplementedError.
#    - performance/occupancy/scheduling: not modelled at all.
#
# This is a WRAPPER-SEMANTICS harness.  It is NOT evidence that the CUDA kernel is
# numerically correct; that requires a real V100 (see research/05_verification_plan.md).
# ======================================================================================
from __future__ import annotations

import math
import sys
import types
from typing import Optional, Tuple

import torch

__all__ = [
    "install",
    "uninstall",
    "is_installed",
    "SUPPORTED_DIMS",
    "ALLOW_DROPOUT",
    "dense_attention_reference",
    "varlen_attention_reference",
    "kvcache_attention_reference",
]

SUPPORTED_DIMS = (16, 32, 64, 128, 256)
NEG_INF = float("-inf")
POS_INF = float("inf")

#: Set to True only in a test that checks *shapes* with dropout_p>0; the math is NOT dropout.
ALLOW_DROPOUT = False

#: Call recorder: tests can assert exactly what the wrapper handed to the extension
#: (dtypes, shapes, converted scalars) without a GPU.  Each entry:
#:   {"fn": "fwd"|"bwd"|"varlen_fwd"|"varlen_bwd"|"fwd_kvcache",
#:    "args": [...], "meta": {...}}
CALL_LOG: list = []


def _record(fn: str, args, meta=None) -> None:
    def enc(x):
        if torch.is_tensor(x):
            return ("Tensor", tuple(x.shape), str(x.dtype))
        return x

    CALL_LOG.append({"fn": fn, "args": [enc(a) for a in args], "meta": meta or {}})


def last_call(fn: str = None):
    for entry in reversed(CALL_LOG):
        if fn is None or entry["fn"] == fn:
            return entry
    return None

_NEG_INF_CLAMP = -1e30  # kernel uses -1e30 (NEG_INF in kernel.h) for masking


# ======================================================================================
# helpers
# ======================================================================================
def _require_fp16_or_fp32(t: torch.Tensor, name: str) -> None:
    if t.dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"{name} must be fp16 (or fp32 in the mock)")


def _check_tail_contiguous(*tensors: Optional[torch.Tensor]) -> None:
    for i, t in enumerate(tensors):
        if t is not None and t.stride(-1) != 1:
            raise RuntimeError(f"tensor #{i} last dim must be contiguous")


def _check_gqa(H_Q: int, H_K: int) -> None:
    if H_K <= 0 or H_Q % H_K != 0:
        raise RuntimeError("H_Q must be divisible by H_K for GQA/MQA")


def _check_head_dim(D: int) -> None:
    if D % 8 != 0:
        raise RuntimeError("head dimension must be multiple of 8")
    if D > 256:
        raise RuntimeError("head dimension must be <= 256")
    if D not in SUPPORTED_DIMS:
        raise RuntimeError(f"Unsupported D: {D}")


def _expand_kv_heads(x: torch.Tensor, H_Q: int) -> torch.Tensor:
    """(..., H_K, S, D) -> (..., H_Q, S, D); kv head = q head // (H_Q // H_K)."""
    H_K = x.shape[-3]
    if H_K == H_Q:
        return x
    return x.repeat_interleave(H_Q // H_K, dim=-3)


def _keep_mask(M: int, N: int, offset: int, causal: bool, win_l: int, win_r: int,
               device: torch.device) -> torch.Tensor:
    """Boolean (M, N), True = attended. Mirrors mat_mul.h masking with seqlen_offset."""
    i = torch.arange(M, device=device).view(M, 1)
    d = torch.arange(N, device=device).view(1, N) - offset  # key position relative to q row i
    keep = torch.ones(M, N, dtype=torch.bool, device=device)
    if causal:
        keep &= d <= i
    if win_l is not None and win_l >= 0:
        keep &= d >= i - win_l
    if win_r is not None and win_r >= 0:
        keep &= d <= i + win_r
    return keep


def _alibi_bias(alibi_slopes: torch.Tensor, B: int, H_Q: int, M: int, N: int,
                offset: int, device: torch.device) -> torch.Tensor:
    """slope * -|i - (j - offset)| with the kernel's SEQLEN_OFFSET rule."""
    a = alibi_slopes.float()
    if a.dim() == 1:
        a = a.view(1, H_Q, 1, 1)
    elif a.dim() == 2:
        a = a.view(B, H_Q, 1, 1)
    else:
        raise RuntimeError("alibi_slopes must be [H_Q] or [B, H_Q]")
    i = torch.arange(M, device=device).view(M, 1)
    j = torch.arange(N, device=device).view(1, N)
    dist = (i - (j - offset)).abs().float()
    return a * (-dist)


def _softmax_from_scores(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """scores (..., M, N) with -inf masked -> (normalised p, lse)."""
    m = scores.amax(dim=-1)                                    # (..., M)
    finite = torch.isfinite(m)
    shifted = torch.where(finite.unsqueeze(-1), scores - m.unsqueeze(-1),
                          torch.full_like(scores, NEG_INF))
    p = torch.exp(shifted)
    psum = p.sum(dim=-1)
    lse = torch.where(finite, m + torch.log(psum.clamp_min(torch.finfo(torch.float32).tiny)),
                      torch.full_like(m, NEG_INF))
    # Normalise before returning: callers use p directly as the attention weights.
    # Fully-masked rows have psum == 0 (p is all zeros) -> they stay zero.
    p = p / psum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    return p, lse


def _scores_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                 causal: bool, win_l: int, win_r: int, softcap: float,
                 alibi_slopes: Optional[torch.Tensor]):
    """
    (B, H_Q, M, D) fp16/fp32, (B, H_K, N, D) -> (out_like_q, lse (B,H_Q,M), p fp32)
    Implements the exact forward feature set of kernel/forward.cu.
    """
    B, H_Q, M, D = q.shape
    H_K, N = k.shape[-3], k.shape[-2]
    _check_gqa(H_Q, H_K)
    _check_head_dim(D)

    # ---- C++ wrapper-level clamps (forward.cu) ----
    if M == 1 and alibi_slopes is None:
        causal = False
    if win_l >= N:
        win_l = -1
    if win_r >= N:
        win_r = -1

    kx = _expand_kv_heads(k, H_Q)
    vx = _expand_kv_heads(v, H_Q)

    offset = N - M
    keep = _keep_mask(M, N, offset, causal, win_l, win_r, q.device)

    scores = torch.matmul(q.float(), kx.float().transpose(-1, -2)) * float(scale)

    # alibi uses SEQLEN_OFFSET = seqlen_offset only if causal or window feature active
    if alibi_slopes is not None:
        alibi_off = offset if (causal or win_l >= 0 or win_r >= 0) else 0
        scores = scores + _alibi_bias(alibi_slopes, B, H_Q, M, N, alibi_off, q.device)

    if softcap > 0.0:
        scores = float(softcap) * torch.tanh(scores / float(softcap))

    scores = scores.masked_fill(~keep, NEG_INF)

    p, lse = _softmax_from_scores(scores)
    out = torch.matmul(p, vx.float()).to(q.dtype)
    return out, lse, p


# ======================================================================================
# public reference entry points (also used directly by tests as an oracle is NOT the
# point — test_wrapper_cpu.py carries an independent implementation; these are what the
# fake extension itself executes).
# ======================================================================================
def dense_attention_reference(q, k, v, softmax_scale, causal=False, window_size=(-1, -1),
                              softcap=0.0, alibi_slopes=None):
    out, lse, _ = _scores_core(q, k, v, softmax_scale, bool(causal),
                               int(window_size[0]), int(window_size[1]),
                               float(softcap), alibi_slopes)
    return out, lse


def varlen_attention_reference(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale,
                               causal=False, window_size=(-1, -1), softcap=0.0,
                               alibi_slopes=None, seqused_k=None):
    """q (T,H_Q,D), k/v (T,H_K,D), cu_seqlens int32 1-D tensors."""
    T, H_Q, D = q.shape
    cuq = [int(x) for x in cu_seqlens_q.tolist()]
    cuk = [int(x) for x in cu_seqlens_k.tolist()]
    B = len(cuq) - 1
    outs, lses = [], []
    for b in range(B):
        qs, qe = cuq[b], cuq[b + 1]
        ks, ke = cuk[b], cuk[b + 1]
        if seqused_k is not None:
            used = int(seqused_k[b])
            ke = ks + (0 if used <= 0 else min(ke - ks, used))
        if qe == qs or ke == ks:
            outs.append(torch.zeros(qe - qs, H_Q, D, dtype=q.dtype, device=q.device))
            # empty KV -> upstream +inf sentinel (research/07 F-8; mirrors the kernel
            # early-exit branch, upstream flash_fwd_kernel.h writes INFINITY).
            lses.append(torch.full((H_Q, qe - qs), POS_INF, dtype=torch.float32, device=q.device))
            continue
        # _scores_core expects BHSD layouts: q (B,H_Q,M,D), k/v (B,H_K,N,D).
        qb = q[qs:qe].transpose(0, 1).unsqueeze(0)      # (Lq,H_Q,D) -> (1,H_Q,Lq,D)
        kb = k[ks:ke].transpose(0, 1).unsqueeze(0)      # (Lk,H_K,D) -> (1,H_K,Lk,D)
        vb = v[ks:ke].transpose(0, 1).unsqueeze(0)
        ob, lb, _ = _scores_core(qb, kb, vb, softmax_scale, bool(causal),
                                 int(window_size[0]), int(window_size[1]),
                                 float(softcap), alibi_slopes)
        outs.append(ob.squeeze(0).transpose(0, 1))      # (1,H_Q,Lq,D) -> (Lq,H_Q,D)
        lses.append(lb.squeeze(0))                      # (H_Q, Lq)
    out = torch.cat(outs, dim=0) if outs else torch.empty(0, H_Q, D, dtype=q.dtype, device=q.device)
    lse = torch.cat(lses, dim=-1) if lses else torch.empty(H_Q, 0, dtype=torch.float32, device=q.device)
    return out, lse


def kvcache_attention_reference(q, k_cache, v_cache, k, v, cache_seqlens,
                                softmax_scale, causal=False, window_size=(-1, -1),
                                softcap=0.0, alibi_slopes=None, cache_batch_idx=None):
    """
    Mirrors kernel/forward_kvcache.cu for the non-paged, non-rotary path.
    k_cache/v_cache (B, N, H_K, D) are updated IN PLACE when k/v are given.
    Returns (out (B,T_Q,H_Q,D), lse (B,H_Q,T_Q)).
    """
    B, T_Q, H_Q, D = q.shape
    _check_head_dim(D)
    N, H_K = k_cache.shape[1], k_cache.shape[2]
    _check_gqa(H_Q, H_K)
    if T_Q > N:
        raise RuntimeError("If key is supplied, it must have seqlen <= the seqlen of the KV cache")

    # clamps (forward_kvcache.cu)
    if T_Q == 1 and alibi_slopes is None:
        causal = False
    if causal:
        window_size = (window_size[0], 0)
    if window_size[0] >= N:
        window_size = (-1, window_size[1])
    if window_size[1] >= N:
        window_size = (window_size[0], -1)

    T_new = k.shape[1] if k is not None else 0
    if T_new > 0 and cache_seqlens is None:
        # Upstream flash_api.cpp: "If key is supplied, seqlens_k must also be passed in".
        # None means "attend the full cache", so there is no append position left.
        raise RuntimeError("If key is supplied, cache_seqlens must also be passed in")
    out = torch.empty_like(q)
    lse = torch.empty(B, H_Q, T_Q, dtype=torch.float32, device=q.device)

    for b in range(B):
        cache_row = int(cache_batch_idx[b]) if cache_batch_idx is not None else b
        # cache_seqlens=None -> full cache (upstream semantics, research/07 F-3);
        # leftpad_k is not modelled here.
        pos = int(cache_seqlens[b]) if cache_seqlens is not None else N
        if pos < 0 or pos > N:
            raise RuntimeError(f"cache_seqlens[{b}]={pos} out of range")
        if T_new > 0:
            if pos + T_new > N:
                raise RuntimeError(f"cache overflow: cache_seqlens[{b}]+seqlen_new > {N}")
            k_cache[cache_row, pos:pos + T_new].copy_(k[b])
            v_cache[cache_row, pos:pos + T_new].copy_(v[b])
        total = pos + T_new
        kb = k_cache[cache_row:cache_row + 1, :total]        # (1,total,H_K,D)
        vb = v_cache[cache_row:cache_row + 1, :total]
        qb = q[b:b + 1]                                      # (1,T_Q,H_Q,D)
        if total == 0:
            out[b].zero_()
            lse[b].fill_(POS_INF)  # empty KV -> +inf sentinel (upstream, research/07 F-8)
            continue
        # _scores_core expects BHSD layouts: (1,T,H,D) -> (1,H,T,D).
        ob, lb, _ = _scores_core(qb.permute(0, 2, 1, 3), kb.permute(0, 2, 1, 3),
                                 vb.permute(0, 2, 1, 3), softmax_scale, bool(causal),
                                 int(window_size[0]), int(window_size[1]),
                                 float(softcap), alibi_slopes)
        out[b].copy_(ob.squeeze(0).permute(1, 0, 2))         # (1,H_Q,T,D) -> (T,H_Q,D)
        lse[b].copy_(lb.squeeze(0))                          # (H_Q, T_Q)
    return out, lse


# ======================================================================================
# fake extension functions (pybind-surface mirrors; see kernel/api.cpp)
# ======================================================================================
def _maybe_dropout_check(p_dropout: float) -> None:
    if p_dropout > 0.0 and not ALLOW_DROPOUT:
        raise NotImplementedError(
            "mock_cuda: dropout simulation is not faithful; set mock_cuda.ALLOW_DROPOUT=True "
            "for shape-only tests or run on a GPU."
        )


def fwd(q, k, v, out, alibi_slopes, p_dropout, softmax_scale, is_causal,
        window_left, window_right, softcap, return_softmax, gen):
    """Mirrors flash_attn_2_cuda.fwd -> (out, lse, dmask, rng_state)."""
    _record("fwd", [q, k, v, out, alibi_slopes, p_dropout, softmax_scale, is_causal,
                    window_left, window_right, softcap, return_softmax, gen])
    # Kernel host check comes first: "Softcapping does not support dropout"
    # (kernel/forward.cu:343; upstream flash_api.cpp has the same check).
    if softcap > 0.0 and p_dropout > 0.0:
        raise RuntimeError("Softcapping does not support dropout")
    _maybe_dropout_check(float(p_dropout))
    if q.dim() != 4:
        raise RuntimeError("fwd expects q of shape (B, M, H, D)")

    # Official FA2 dense layout is BMHD [B, M, H, D]; reference math is BHSD.
    qb = q.permute(0, 2, 1, 3)
    kb = k.permute(0, 2, 1, 3)
    vb = v.permute(0, 2, 1, 3)
    out_ref, lse, _ = _scores_core(qb, kb, vb, float(softmax_scale), bool(is_causal),
                                   int(window_left), int(window_right), float(softcap),
                                   alibi_slopes)
    out_ref = out_ref.permute(0, 2, 1, 3).contiguous()
    if out is not None:
        out.copy_(out_ref)
        out_t = out
    else:
        out_t = out_ref

    B, M, H_Q, N = q.shape[0], q.shape[1], q.shape[2], k.shape[1]
    if return_softmax and p_dropout > 0.0:
        dmask = torch.ones(B, H_Q, M, N, dtype=q.dtype, device=q.device)  # keep-all (no dropout)
    else:
        dmask = torch.empty(0, dtype=q.dtype, device=q.device)
    rng_state = torch.zeros(2, dtype=torch.int64, device=q.device)
    return out_t, lse, dmask, rng_state


def bwd(dout, q, k, v, out, lse, dq, dk, dv, alibi_slopes, p_dropout, softmax_scale,
        is_causal, window_left, window_right, softcap, deterministic, gen, rng_state):
    """Mirrors flash_attn_2_cuda.bwd -> (dq, dk, dv[, dsoftmax_sum]).

    Gradients are produced by torch.autograd through the same fp32 reference math
    used by fwd(); leaf grads are quantized to the input dtype exactly like the
    kernel writes fp16 outputs.  The 4th return (dsoftmax_sum) is not produced
    because the wrapper never reads it."""
    _record("bwd", [dout, q, k, v, out, lse, dq, dk, dv, alibi_slopes, p_dropout,
                    softmax_scale, is_causal, window_left, window_right, softcap,
                    deterministic, gen, rng_state])
    _maybe_dropout_check(float(p_dropout))
    if softcap > 0.0 and p_dropout > 0.0:
        raise RuntimeError("Softcapping does not support dropout")

    with torch.enable_grad():
        qd = q.detach().permute(0, 2, 1, 3).clone().requires_grad_(True)
        kd = k.detach().permute(0, 2, 1, 3).clone().requires_grad_(True)
        vd = v.detach().permute(0, 2, 1, 3).clone().requires_grad_(True)
        out_ref, _, _ = _scores_core(qd, kd, vd, float(softmax_scale), bool(is_causal),
                                     int(window_left), int(window_right), float(softcap),
                                     alibi_slopes)
        grad_out = dout.detach().permute(0, 2, 1, 3).to(out_ref.dtype)
        g = torch.autograd.grad(out_ref, (qd, kd, vd), grad_outputs=grad_out)

    gq = g[0].permute(0, 2, 1, 3).contiguous().to(q.dtype)
    gk = g[1].permute(0, 2, 1, 3).contiguous().to(k.dtype)
    gv = g[2].permute(0, 2, 1, 3).contiguous().to(v.dtype)
    if dq is not None:
        dq.copy_(gq)
    if dk is not None:
        dk.copy_(gk)
    if dv is not None:
        dv.copy_(gv)
    return gq, gk, gv


def varlen_fwd(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k,
               block_table, alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout,
               softmax_scale, zero_tensors, is_causal, window_left, window_right,
               softcap, return_softmax, gen, num_splits):
    """Mirrors flash_attn_2_cuda.varlen_fwd -> (out, lse, dmask, rng_state)."""
    _record("varlen_fwd", [q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k,
                           leftpad_k, block_table, alibi_slopes, max_seqlen_q,
                           max_seqlen_k, p_dropout, softmax_scale, zero_tensors,
                           is_causal, window_left, window_right, softcap,
                           return_softmax, gen, num_splits])
    _maybe_dropout_check(float(p_dropout))
    if softcap > 0.0 and p_dropout > 0.0:
        raise RuntimeError("Softcapping does not support dropout")
    if leftpad_k is not None:
        raise NotImplementedError("mock_cuda.varlen_fwd: leftpad_k is not modelled")
    if block_table is not None:
        raise NotImplementedError("mock_cuda.varlen_fwd: paged block_table is not modelled")
    if num_splits not in (0, 1):
        raise NotImplementedError("mock_cuda.varlen_fwd: num_splits > 1 is not modelled")

    while cu_seqlens_q.dim() > 1:
        cu_seqlens_q = cu_seqlens_q.squeeze(0)
    while cu_seqlens_k.dim() > 1:
        cu_seqlens_k = cu_seqlens_k.squeeze(0)
    out_ref, lse = varlen_attention_reference(
        q, k, v, cu_seqlens_q, cu_seqlens_k, float(softmax_scale), bool(is_causal),
        (int(window_left), int(window_right)), float(softcap), alibi_slopes,
        seqused_k=seqused_k)
    if out is not None:
        out.copy_(out_ref)
        out_t = out
    else:
        out_t = out_ref

    H_Q = q.shape[1]
    T_Q = q.shape[0]
    N_max = int(max_seqlen_k)
    if return_softmax and p_dropout > 0.0:
        # Kernel layout: (T_Q, H_Q, max_seqlen_k) -- research/07 F-6 (fork varlen
        # dmask is the tight 3-D tensor, NOT upstream's padded 4-D one).
        dmask = torch.ones(T_Q, H_Q, N_max, dtype=q.dtype, device=q.device)  # keep-all
    else:
        dmask = torch.empty(0, dtype=q.dtype, device=q.device)
    rng_state = torch.zeros(2, dtype=torch.int64, device=q.device)
    return out_t, lse, dmask, rng_state


def varlen_bwd(dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
               alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale,
               zero_tensors, is_causal, window_left, window_right, softcap,
               deterministic, gen, rng_state):
    """Mirrors flash_attn_2_cuda.varlen_bwd -> (dq, dk, dv[, dsoftmax_sum])."""
    _record("varlen_bwd", [dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens_q,
                           cu_seqlens_k, alibi_slopes, max_seqlen_q, max_seqlen_k,
                           p_dropout, softmax_scale, zero_tensors, is_causal,
                           window_left, window_right, softcap, deterministic,
                           gen, rng_state])
    _maybe_dropout_check(float(p_dropout))
    if softcap > 0.0 and p_dropout > 0.0:
        raise RuntimeError("Softcapping does not support dropout")

    while cu_seqlens_q.dim() > 1:
        cu_seqlens_q = cu_seqlens_q.squeeze(0)
    while cu_seqlens_k.dim() > 1:
        cu_seqlens_k = cu_seqlens_k.squeeze(0)

    with torch.enable_grad():
        qd = q.detach().clone().requires_grad_(True)
        kd = k.detach().clone().requires_grad_(True)
        vd = v.detach().clone().requires_grad_(True)
        out_ref, _ = varlen_attention_reference(
            qd, kd, vd, cu_seqlens_q, cu_seqlens_k, float(softmax_scale), bool(is_causal),
            (int(window_left), int(window_right)), float(softcap), alibi_slopes)
        grad_out = dout.detach().to(out_ref.dtype)
        g = torch.autograd.grad(out_ref, (qd, kd, vd), grad_outputs=grad_out)

    gq = g[0].detach().to(q.dtype)
    gk = g[1].detach().to(k.dtype)
    gv = g[2].detach().to(v.dtype)
    if dq is not None:
        dq.copy_(gq)
    if dk is not None:
        dk.copy_(gk)
    if dv is not None:
        dv.copy_(gv)
    return gq, gk, gv


def fwd_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin,
                cache_batch_idx, leftpad_k, block_table, alibi_slopes, out,
                softmax_scale, is_causal, window_left, window_right, softcap,
                is_rotary_interleaved, num_splits):
    """Mirrors flash_attn_2_cuda.fwd_kvcache -> (out, softmax_lse)."""
    _record("fwd_kvcache", [q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos,
                            rotary_sin, cache_batch_idx, leftpad_k, block_table,
                            alibi_slopes, out, softmax_scale, is_causal,
                            window_left, window_right, softcap,
                            is_rotary_interleaved, num_splits])
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("mock_cuda.fwd_kvcache: rotary is not modelled")
    if leftpad_k is not None:
        raise NotImplementedError("mock_cuda.fwd_kvcache: leftpad_k is not modelled")
    if block_table is not None:
        raise NotImplementedError("mock_cuda.fwd_kvcache: paged block_table is not modelled")
    if softcap > 0.0 and (window_left >= 0 or window_right >= 0):
        raise RuntimeError("Softcap + window not supported")
    if softcap > 0.0 and alibi_slopes is not None:
        raise RuntimeError("Softcap + ALiBi not supported")

    out_ref, lse = kvcache_attention_reference(
        q, k_cache, v_cache, k, v, cache_seqlens, float(softmax_scale),
        bool(is_causal), (int(window_left), int(window_right)), float(softcap),
        alibi_slopes, cache_batch_idx=cache_batch_idx)
    if out is not None:
        out.copy_(out_ref)
        out_t = out
    else:
        out_t = out_ref
    return out_t, lse


# ======================================================================================
# sys.modules injection
# ======================================================================================
_MODULE_NAME = "flash_attn_2_cuda"

_FUNCS = {
    "fwd": fwd,
    "bwd": bwd,
    "varlen_fwd": varlen_fwd,
    "varlen_bwd": varlen_bwd,
    "fwd_kvcache": fwd_kvcache,
}


def _build_module() -> types.ModuleType:
    mod = types.ModuleType(_MODULE_NAME)
    mod.__doc__ = "MOCK flash_attn_2_cuda (CPU reference; see mock_cuda.py)"
    mod.IS_MOCK = True
    for name, fn in _FUNCS.items():
        setattr(mod, name, fn)
    return mod


def install() -> types.ModuleType:
    """Fabricate the `flash_attn_2_cuda` extension module.

    Call BEFORE `import flash_attn` (or any module that imports the extension).
    """
    mod = _build_module()
    sys.modules[_MODULE_NAME] = mod
    return mod


def uninstall() -> None:
    sys.modules.pop(_MODULE_NAME, None)


def is_installed() -> bool:
    return sys.modules.get(_MODULE_NAME) is not None and getattr(
        sys.modules.get(_MODULE_NAME), "IS_MOCK", False
    )
