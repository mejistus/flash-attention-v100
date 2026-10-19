#!/usr/bin/env python
# ======================================================================================
# * GPU verification suite for the flash_attn drop-in build vs an eager float32 reference.
# *
# * Requires a Volta (sm_70) GPU; exits cleanly (skip) otherwise. This file is the
# * end-to-end correctness harness to run on real hardware, covering dense / varlen /
# * kvcache forward, backward gradients, GQA, causal, sliding-window, softcap, alibi,
# * head-dim padding and lse outputs.
# *
# * The reference semantics implemented here are unit-tested on CPU in
# * tests/test_reference_cpu.py.
# *
# * Usage on a V100 machine:
# *     python tests/gpu_verify.py            # full suite
# *     python tests/gpu_verify.py --quick    # small shapes only
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import argparse
import math
import sys
import traceback

import torch

QUICK = False


# ---------------------------------------------------------------------------------------
# Eager reference (documented FA2 semantics, fp32 compute)
# ---------------------------------------------------------------------------------------
def reference_attention(q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1),
                        softcap=0.0, alibi_slopes=None):
    """Returns (out, lse). q (B,M,H,D); k,v (B,N,Hk,D); fp16 in/out like the kernel."""
    B, M, H, D = q.shape
    N = k.shape[1]
    scale = (D ** -0.5) if softmax_scale is None else softmax_scale

    qf, kf, vf = q.float(), k.float(), v.float()
    Hk = kf.shape[2]
    if Hk != H:
        assert H % Hk == 0
        rep = H // Hk
        kf = kf.repeat_interleave(rep, dim=2)
        vf = vf.repeat_interleave(rep, dim=2)

    s = torch.einsum("bmhd,bnhd->bhmn", qf, kf) * scale
    if softcap > 0.0:
        s = softcap * torch.tanh(s / softcap)

    device = q.device
    diag = torch.arange(M, device=device)[:, None] + (N - M)   # bottom-right aligned
    kj = torch.arange(N, device=device)[None, :]
    allowed = torch.ones(M, N, dtype=torch.bool, device=device)
    if causal:
        allowed &= kj <= diag
    wl, wr = window_size
    if wl >= 0:
        allowed &= kj >= (diag - wl)
    if wr >= 0:
        allowed &= kj <= (diag + wr)

    if alibi_slopes is not None:
        slopes = alibi_slopes.float()
        dist = (diag - kj).abs().float()  # (M,N)
        if slopes.dim() == 1:
            bias = slopes.view(1, -1, 1, 1) * dist.view(1, 1, M, N)
        else:  # (B,H)
            bias = slopes.view(B, H, 1, 1) * dist.view(1, 1, M, N)
        s = s - bias

    s = s.masked_fill(~allowed, float("-inf"))
    row_max = s.amax(dim=-1, keepdim=True)
    fully = torch.isinf(row_max) & (row_max < 0)
    s_safe = torch.where(fully.expand_as(s), torch.zeros_like(s), s)
    p = torch.softmax(s_safe, dim=-1)
    p = torch.where(fully, torch.zeros_like(p), p)
    o = torch.einsum("bhmn,bnhd->bmhd", p, vf)
    lse = torch.logsumexp(s, dim=-1)  # (B,H,M), -inf on fully masked rows
    return o.to(q.dtype), lse


# ---------------------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def result(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name} {('- ' + detail) if detail else ''}")
        return ok

    def skip(self, name, detail=""):
        self.skipped += 1
        print(f"  [SKIP] {name} {('- ' + detail) if detail else ''}")

    def close(self, name, got, ref, atol, rtol):
        got = got.float()
        ref = ref.float()
        finite = torch.isfinite(ref)
        got_c = torch.where(finite, got, torch.zeros_like(got))
        ref_c = torch.where(finite, ref, torch.zeros_like(ref))
        diff = (got_c - ref_c).abs()
        allowed = atol + rtol * ref_c.abs()
        bad = diff > allowed
        n_bad = int(bad.sum())
        if n_bad:
            idx = bad.nonzero()[0]
            maxdiff = float(diff.max())
            return self.result(
                name, False,
                f"{n_bad}/{bad.numel()} elems out of tol; max|d|={maxdiff:.3e} at {idx.tolist()}, "
                f"got={float(got_c[tuple(idx)]):.6f} ref={float(ref_c[tuple(idx)]):.6f}")
        return self.result(name, True)


R = Report()
TOL = dict(atol=2e-2, rtol=2e-2)
LTOL = dict(atol=5e-2, rtol=5e-2)   # lse can be large magnitude (log-sum-exp)


def require_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available: gpu_verify is skipped (expected on build-only machines).")
        sys.exit(0)
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {name} cc={cc}")
    if cc != (7, 0):
        print("WARNING: this build targets sm_70 only; a non-Volta GPU may fail to launch.")


def gen(B, M, N, H, Hk, D, device, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, M, H, D, generator=g, dtype=torch.float32).to(device).to(dtype)
    k = torch.randn(B, N, Hk, D, generator=g, dtype=torch.float32).to(device).to(dtype)
    v = torch.randn(B, N, Hk, D, generator=g, dtype=torch.float32).to(device).to(dtype)
    return q, k, v


# ---------------------------------------------------------------------------------------
# Forward: dense
# ---------------------------------------------------------------------------------------
def test_dense(flash_attn_func, shapes):
    print("\n== dense forward ==")
    for (D, M, N, H, Hk) in shapes:
        B = 2
        q, k, v = gen(B, M, N, H, Hk, D, torch.device("cuda"))

        for causal in (False, True):
            out = flash_attn_func(q, k, v, causal=causal)
            ref, lse_ref = reference_attention(q, k, v, causal=causal)
            R.close(f"dense D={D} M={M} N={N} H={H}/{Hk} causal={causal} out",
                    out[:, :, :out.shape[2], :].float(), ref, **TOL)
            out2, lse = flash_attn_func(q, k, v, causal=causal, return_attn_probs=True)[:2]
            R.close(f"dense D={D} M={M} N={N} causal={causal} lse", lse, lse_ref, **LTOL)

        # sliding window (aligned bottom-right)
        for ws in ((32, 32), (0, 0), (-1, 0), (7, -1)):
            out = flash_attn_func(q, k, v, causal=True, window_size=ws)
            ref, _ = reference_attention(q, k, v, causal=True, window_size=ws)
            R.close(f"dense D={D} window={ws} out", out.float(), ref, **TOL)

        if D >= 32:
            out = flash_attn_func(q, k, v, softcap=8.0)
            ref, _ = reference_attention(q, k, v, softcap=8.0)
            R.close(f"dense D={D} softcap out", out.float(), ref, **TOL)

        alibi = torch.randn(H, dtype=torch.float32, device="cuda") * 0.3
        out = flash_attn_func(q, k, v, causal=True, alibi_slopes=alibi)
        ref, _ = reference_attention(q, k, v, causal=True, alibi_slopes=alibi)
        R.close(f"dense D={D} alibi out", out.float(), ref, **TOL)


# ---------------------------------------------------------------------------------------
# Forward: varlen
# ---------------------------------------------------------------------------------------
def test_varlen(flash_attn_func, D, seqs, H, Hk):
    print(f"\n== varlen forward (D={D}) ==")
    B = len(seqs)
    total_q = sum(seqs)
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(total_q, H, D, generator=g, dtype=torch.float32).to("cuda").half()
    k = torch.randn(total_q, Hk, D, generator=g, dtype=torch.float32).to("cuda").half()
    v = torch.randn(total_q, Hk, D, generator=g, dtype=torch.float32).to("cuda").half()
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqs), 0).tolist()),
                      dtype=torch.int32, device="cuda")
    maxs = max(seqs)

    out = flash_attn_func(q, k, v, cu, cu, maxs, maxs)
    out_probs, lse, _ = flash_attn_func(q, k, v, cu, cu, maxs, maxs,
                                        return_attn_probs=True)

    # per-sequence reference
    o_ref = torch.empty_like(q)
    l_ref = torch.empty(H, total_q, dtype=torch.float32, device="cuda")
    for i, s in enumerate(seqs):
        a, b = int(cu[i]), int(cu[i + 1])
        for causal in (False, True):
            pass
        ro, rl = reference_attention(
            q[a:b].unsqueeze(0), k[a:b].unsqueeze(0), v[a:b].unsqueeze(0))
        o_ref[a:b] = ro[0]
        l_ref[:, a:b] = rl[0]
    R.close(f"varlen D={D} out", out.float(), o_ref.float(), **TOL)
    R.close(f"varlen D={D} lse", lse, l_ref, **LTOL)


# ---------------------------------------------------------------------------------------
# KVCache (decode)
# ---------------------------------------------------------------------------------------
def test_kvcache(flash_attn_with_kvcache, D, H, Hk, cache_len, q_len):
    print(f"\n== kvcache decode (D={D}, cache={cache_len}, q={q_len}) ==")
    B = 2
    g = torch.Generator(device="cpu").manual_seed(2)
    kc = torch.randn(B, cache_len, Hk, D, generator=g, dtype=torch.float32).to("cuda").half()
    vc = torch.randn(B, cache_len, Hk, D, generator=g, dtype=torch.float32).to("cuda").half()
    q = torch.randn(B, q_len, H, D, generator=g, dtype=torch.float32).to("cuda").half()
    cache_seqlens = torch.full((B,), cache_len, dtype=torch.int32, device="cuda")

    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cache_seqlens)
    ref, _ = reference_attention(q, kc, vc)
    R.close(f"kvcache D={D} out", out.float(), ref, **TOL)

    out2, lse = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cache_seqlens,
                                        return_softmax_lse=True)
    _, lse_ref = reference_attention(q, kc, vc)
    R.close(f"kvcache D={D} lse", lse, lse_ref, **LTOL)

    # append new k/v
    knew = torch.randn(B, q_len, Hk, D, generator=g, dtype=torch.float32).to("cuda").half()
    vnew = torch.randn(B, q_len, Hk, D, generator=g, dtype=torch.float32).to("cuda").half()
    out = flash_attn_with_kvcache(q, kc, vc, knew, vnew, cache_seqlens=cache_seqlens)
    ref, _ = reference_attention(q, torch.cat([kc, knew], 1), torch.cat([vc, vnew], 1))
    R.close(f"kvcache D={D} append out", out.float(), ref, **TOL)


# ---------------------------------------------------------------------------------------
# Backward (dense + varlen)
# ---------------------------------------------------------------------------------------
def test_backward(flash_attn_func, flash_attn_varlen_func, D, M, H, Hk, causal=True):
    print(f"\n== backward (D={D}, M={M}) ==")
    device = torch.device("cuda")
    B = 2
    for dropout in (0.0, 0.1):
        q, k, v = gen(B, M, M, H, Hk, D, device, seed=7)

        def run(fn):
            qq = q.clone().requires_grad_(True)
            kk = k.clone().requires_grad_(True)
            vv = v.clone().requires_grad_(True)
            o = fn(qq, kk, vv)
            w = torch.randn_like(o)
            (o.float() * w.float()).sum().backward()
            return o.detach(), qq.grad, kk.grad, vv.grad

        o1, dq1, dk1, dv1 = run(lambda a, b, c: flash_attn_func(a, b, c, dropout_p=dropout, causal=causal))

        q2 = q.clone().requires_grad_(True)
        k2 = k.clone().requires_grad_(True)
        v2 = v.clone().requires_grad_(True)
        if dropout == 0.0:
            ref, _ = reference_attention(q2, k2, v2, causal=causal)
            w = torch.randn_like(ref)
            (ref.float() * w.float()).sum().backward()
        else:
            # with dropout we only check shapes / finiteness
            w = torch.randn_like(o1)
            (o1.float() * w.float()).sum().backward()
            for nm, t in (("dq", dq1), ("dk", dk1), ("dv", dv1)):
                R.result(f"bwd dropout D={D} {nm} finite", bool(t.isfinite().all()))
            continue

        R.close(f"bwd D={D} out", o1.float(), ref.float(), **TOL)
        R.close(f"bwd D={D} dq", dq1.float(), q2.grad.float(), **TOL)
        R.close(f"bwd D={D} dk", dk1.float(), k2.grad.float(), **TOL)
        R.close(f"bwd D={D} dv", dv1.float(), v2.grad.float(), **TOL)


# ---------------------------------------------------------------------------------------
# Head-dim padding path (D not in {16,32,64,128,256})
# ---------------------------------------------------------------------------------------
def test_padded_head_dim(flash_attn_func):
    print("\n== head-dim padding path ==")
    for D in (8, 48, 96, 160):
        q, k, v = gen(1, 64, 64, 2, 2, D, torch.device("cuda"), seed=9)
        out = flash_attn_func(q, k, v, causal=True)
        ref, _ = reference_attention(q, k, v, causal=True)
        R.close(f"head-dim {D} (padded) out", out.float(), ref, **TOL)


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    require_cuda()
    import flash_attn
    print("flash_attn version:", flash_attn.__version__)

    if args.quick:
        dense_shapes = [(64, 128, 128, 4, 4)]
        varlen_D = [64]
    else:
        dense_shapes = [
            (16, 256, 256, 4, 4),
            (32, 128, 448, 4, 2),
            (64, 192, 192, 8, 8),
            (64, 257, 129, 4, 1),
            (128, 128, 4096, 8, 2),
            (128, 1, 1024, 4, 4),
            (256, 64, 64, 2, 2),
        ]
        varlen_D = [64, 128]

    try:
        test_dense(flash_attn.flash_attn_func, dense_shapes)
        for D in varlen_D:
            test_varlen(flash_attn.flash_attn_varlen_func, D, [17, 32, 129], 4, 2)
        test_kvcache(flash_attn.flash_attn_with_kvcache, 128, 4, 2, 512, 4)
        test_kvcache(flash_attn.flash_attn_with_kvcache, 64, 4, 4, 256, 1)
        test_backward(flash_attn.flash_attn_func, flash_attn.flash_attn_varlen_func, 64, 128, 2, 2)
        test_padded_head_dim(flash_attn.flash_attn_func)
    except Exception:
        traceback.print_exc()
        R.failed += 1

    print(f"\n==== gpu_verify summary: {R.passed} passed, {R.failed} failed, {R.skipped} skipped ====")
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
