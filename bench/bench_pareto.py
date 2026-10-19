#!/usr/bin/env python
"""Pareto benchmark for the V100 flash-attn build vs PyTorch SDPA.

Measures peak CUDA memory (allocated/reserved) and latency for:
  * dense GQA training (fwd+bwd)          flash vs SDPA(mem-efficient)
  * packed varlen GQA training            flash (packed) vs dense padded SDPA
  * MHA vs GQA varlen training            isolates the GQA backward expansion buffers
  * inference prefill (no grad)           flash vs SDPA
  * kvcache decode                        flash

Usage (on the GPU box):
    python bench/bench_pareto.py --json bench/results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings

import torch
import flash_attn
from torch.nn.attention import SDPBackend, sdpa_kernel

DEV = "cuda"
DT = torch.float16


def _reset() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak() -> tuple[float, float]:
    return (torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20)


def _sdpa(q, k, v, causal, gqa):
    # PyTorch's fused SDPA kernels on sm_70 require equal Q/KV head counts. That is the
    # practical V100 baseline: expand KV heads (extra memory + a backward reduction)
    # before calling mem-efficient attention.
    if gqa and k.shape[2] != q.shape[2]:
        rep = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return o.transpose(1, 2)


def _time(fn, iters: int) -> float:
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def _row(r):
    print(f"{r['case']:60s} {r['peak_alloc_mib']:11.1f} {r['peak_reserved_mib']:9.1f} {r['ms']:8.3f}", flush=True)
    return r


def bench_train(name, make_leaves, fwd, iters):
    """fwd(leaves) -> out; backward with a fixed grad; returns metrics."""
    _reset()
    leaves, grad = make_leaves()
    fn = lambda: torch.autograd.backward(fwd(leaves), grad)
    ms = _time(fn, iters)
    alloc, reserved = _peak()
    return _row(dict(case=name, peak_alloc_mib=round(alloc, 1),
                     peak_reserved_mib=round(reserved, 1), ms=round(ms, 3)))


def bench_forward(name, make_inputs, fwd, iters):
    _reset()
    args = make_inputs()
    ms = _time(lambda: fwd(args), iters)
    alloc, reserved = _peak()
    return _row(dict(case=name, peak_alloc_mib=round(alloc, 1),
                     peak_reserved_mib=round(reserved, 1), ms=round(ms, 3)))


def _dense(B, N, H, HK, D):
    q = torch.randn(B, N, H, D, device=DEV, dtype=DT) * 0.5
    k = torch.randn(B, N, HK, D, device=DEV, dtype=DT) * 0.5
    v = torch.randn(B, N, HK, D, device=DEV, dtype=DT) * 0.5
    g = torch.randn(B, N, H, D, device=DEV, dtype=DT) * 0.1
    return q, k, v, g


def _varlen(T, H, HK, D, nseq):
    lens = [T // nseq] * nseq
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                      device=DEV, dtype=torch.int32)
    q = torch.randn(T, H, D, device=DEV, dtype=DT) * 0.5
    k = torch.randn(T, HK, D, device=DEV, dtype=DT) * 0.5
    v = torch.randn(T, HK, D, device=DEV, dtype=DT) * 0.5
    g = torch.randn(T, H, D, device=DEV, dtype=DT) * 0.1
    return q, k, v, g, cu, max(lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
          f"| flash_attn {flash_attn.__version__}")
    rows = []

    # 1) dense GQA fwd+bwd
    for N in (2048, 4096, 8192):
        def leaves(N=N):
            q, k, v, g = _dense(1, N, 32, 8, 128)
            return ((q.requires_grad_(True), k.requires_grad_(True),
                     v.requires_grad_(True)), g)
        rows.append(bench_train(f"dense_gqa_fwd_bwd N={N} H=32/HK=8 D=128", leaves,
                                lambda L: flash_attn.flash_attn_func(*L, causal=True), args.iters))

    # 2) packed varlen GQA fwd+bwd
    for T in (8192, 16384, 32768):
        def leaves(T=T):
            q, k, v, g, cu, ml = _varlen(T, 32, 8, 128, 4)
            return ((q.requires_grad_(True), k.requires_grad_(True),
                     v.requires_grad_(True), cu, ml), g)
        def fwd(L):
            q, k, v, cu, ml = L
            return flash_attn.flash_attn_varlen_func(q, k, v, cu, cu,
                                                     max_seqlen_q=ml, max_seqlen_k=ml,
                                                     causal=True)
        rows.append(bench_train(f"varlen_gqa_fwd_bwd T={T} (4 seqs) H=32/HK=8 D=128",
                                leaves, fwd, args.iters))

    # 3) MHA vs GQA varlen: expansion-buffer cost (same T)
    for HK in (32, 8):
        def leaves(HK=HK):
            q, k, v, g, cu, ml = _varlen(16384, 32, HK, 128, 4)
            return ((q.requires_grad_(True), k.requires_grad_(True),
                     v.requires_grad_(True), cu, ml), g)
        def fwd(L):
            q, k, v, cu, ml = L
            return flash_attn.flash_attn_varlen_func(q, k, v, cu, cu,
                                                     max_seqlen_q=ml, max_seqlen_k=ml,
                                                     causal=True)
        tag = "MHA" if HK == 32 else "GQA(8)"
        rows.append(bench_train(f"varlen_{tag}_fwd_bwd T=16384 H=32 D=128",
                                leaves, fwd, args.iters))

    # 3b) dense MHA (H == HK) flash vs SDPA
    for N in (4096, 8192):
        def leaves(N=N):
            q, k, v, g = _dense(1, N, 32, 32, 128)
            return ((q.requires_grad_(True), k.requires_grad_(True),
                     v.requires_grad_(True)), g)
        rows.append(bench_train(f"dense_mha_fwd_bwd N={N} H=32/HK=32 D=128", leaves,
                                lambda L: flash_attn.flash_attn_func(*L, causal=True), args.iters))
        rows.append(bench_train(f"dense_mha_fwd_bwd_SDPA N={N} H=32/HK=32 D=128", leaves,
                                lambda L: _sdpa(*L, causal=True, gqa=False), args.iters))

    # 4) SDPA baselines (same shapes)
    for N in (4096, 8192):
        def leaves(N=N):
            q, k, v, g = _dense(1, N, 32, 8, 128)
            return ((q.requires_grad_(True), k.requires_grad_(True),
                     v.requires_grad_(True)), g)
        rows.append(bench_train(f"dense_gqa_fwd_bwd_SDPA N={N} H=32/HK=8 D=128",
                                leaves, lambda L: _sdpa(*L, causal=True, gqa=True), args.iters))

    # packed-varlen alternative for SDPA: padded dense batch (4 x N/4), no pad mask
    for T in (16384, 32768):
        B, N = 4, T // 4
        def leaves(B=B, N=N):
            q, k, v, g = _dense(B, N, 32, 8, 128)
            return ((q.requires_grad_(True), k.requires_grad_(True),
                     v.requires_grad_(True)), g)
        rows.append(bench_train(f"padded_dense_SDPA B={B} N={N} (tok={T}) H=32/HK=8 D=128",
                                leaves, lambda L: _sdpa(*L, causal=True, gqa=True), args.iters))

    # 5) inference prefill (no grad)
    for N in (4096, 8192):
        def inputs(N=N):
            q, k, v, _ = _dense(1, N, 32, 8, 128)
            return q, k, v
        rows.append(bench_forward(f"prefill_fwd N={N} H=32/HK=8 D=128", inputs,
                                  lambda L: flash_attn.flash_attn_func(*L, causal=True),
                                  args.iters))
    for N in (4096, 8192):
        def inputs(N=N):
            q, k, v, _ = _dense(1, N, 32, 8, 128)
            return q, k, v
        rows.append(bench_forward(f"prefill_fwd_SDPA N={N} H=32/HK=8 D=128", inputs,
                                  lambda L: _sdpa(*L, causal=True, gqa=True), args.iters))

    # 6) kvcache decode
    for N in (4096, 8192):
        def inputs(N=N):
            q = torch.randn(1, 1, 32, 128, device=DEV, dtype=DT) * 0.5
            kc = torch.randn(1, N, 8, 128, device=DEV, dtype=DT) * 0.5
            vc = torch.randn(1, N, 8, 128, device=DEV, dtype=DT) * 0.5
            sl = torch.tensor([N], device=DEV, dtype=torch.int32)
            return q, kc, vc, sl
        rows.append(bench_forward(
            f"kvcache_decode cache={N} H=32/HK=8 D=128", inputs,
            lambda L: flash_attn.flash_attn_with_kvcache(L[0], L[1], L[2], cache_seqlens=L[3]),
            args.iters))

    print(f"\n{'case':60s} {'peak_alloc':>11s} {'peak_res':>9s} {'ms':>8s}")
    for r in rows:
        print(f"{r['case']:60s} {r['peak_alloc_mib']:11.1f} {r['peak_reserved_mib']:9.1f} {r['ms']:8.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows}, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
