#!/usr/bin/env python
# ======================================================================================
# * CPU unit tests for the eager reference in tests/gpu_verify.py.
# * The GPU suite compares the flash_attn build against this reference, so the reference's
# * semantics (causal window/alignment, sliding window, softcap, alibi, GQA, fully
# * masked rows) must themselves be proven correct - on CPU, here.
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import _stub_cuda_backend as sb  # noqa: E402
from gpu_verify import reference_attention  # noqa: E402

R = sb.RESULT
torch.manual_seed(0)


def manual_attention(q, k, v, scale, allowed, bias=None, softcap=0.0):
    """Ground-truth python loop; q/k/v: (M, D),(N, D),(N, D) fp32; allowed (M, N) bool."""
    M, D = q.shape
    N = k.shape[0]
    out = torch.zeros(M, v.shape[-1], dtype=torch.float32)
    lse = torch.empty(M, dtype=torch.float32)
    for i in range(M):
        idx = allowed[i].nonzero().flatten().tolist()
        if len(idx) == 0:
            lse[i] = float("-inf")
            continue
        sc = torch.tensor([float(torch.dot(q[i], k[j]) * scale) for j in idx])
        if softcap > 0.0:
            sc = softcap * torch.tanh(sc / softcap)
        if bias is not None:
            sc = sc + torch.tensor([float(bias[i, j]) for j in idx])
        m = sc.max()
        p = torch.exp(sc - m)
        lse[i] = m + torch.log(p.sum())
        p = p / p.sum()
        out[i] = sum(p[t] * v[j] for t, j in enumerate(idx))
    return out, lse


def cmp(name, got, ref, atol=1e-5):
    got = got.float()
    ref = ref.float()
    ok = torch.allclose(got, ref, atol=atol, rtol=0)
    if not ok:
        R.fail(name, f"max|d|={(got - ref).abs().max().item():.3e}")
    else:
        R.ok(name)


def test_causal_non_causal():
    M = N = 5
    D = 3
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    v = torch.randn(N, 2)
    scale = D ** -0.5

    for causal in (False, True):
        allowed = torch.ones(M, N, dtype=torch.bool)
        if causal:
            allowed = torch.tril(allowed)
        mo, ml = manual_attention(q, k, v, scale, allowed)
        ro, rl = reference_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                     causal=causal)
        cmp(f"reference causal={causal} out", ro[0, :, 0], mo)
        cmp(f"reference causal={causal} lse", rl[0, 0], ml)


def test_bottom_right_alignment():
    # M < N: query i aligns to key i + (N - M); causal row 0 sees keys 0..N-M.
    M, N, D = 2, 5, 3
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    v = torch.randn(N, 1)
    scale = D ** -0.5
    allowed = torch.zeros(M, N, dtype=torch.bool)
    for i in range(M):
        allowed[i, : (i + N - M + 1)] = True
    mo, ml = manual_attention(q, k, v, scale, allowed)
    ro, rl = reference_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                 causal=True)
    cmp("reference bottom-right alignment out", ro[0, :, 0], mo)
    cmp("reference bottom-right alignment lse", rl[0, 0], ml)


def test_window():
    M = N = 6
    D = 4
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    v = torch.randn(N, 1)
    scale = D ** -0.5
    for (wl, wr) in ((0, 0), (2, 1), (3, -1), (-1, 2)):
        allowed = torch.zeros(M, N, dtype=torch.bool)
        for i in range(M):
            # causal + window (bottom-right aligned; M == N here): the causal bound caps
            # the right edge at the diagonal, and wr can only shrink it further.
            lo = 0 if wl < 0 else max(0, i - wl)
            hi = i if wr < 0 else min(i, i + wr)
            allowed[i, lo:hi + 1] = True
        mo, ml = manual_attention(q, k, v, scale, allowed)
        ro, rl = reference_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                     causal=True, window_size=(wl, wr))
        cmp(f"reference window={wl},{wr} out", ro[0, :, 0], mo)
        cmp(f"reference window={wl},{wr} lse", rl[0, 0], ml)


def test_softcap():
    M = N = 4
    D = 3
    q = torch.randn(M, D) * 2
    k = torch.randn(N, D) * 2
    v = torch.randn(N, 1)
    scale = D ** -0.5
    allowed = torch.ones(M, N, dtype=torch.bool)
    mo, ml = manual_attention(q, k, v, scale, allowed, softcap=1.5)
    ro, rl = reference_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                 softcap=1.5)
    cmp("reference softcap out", ro[0, :, 0], mo)
    cmp("reference softcap lse", rl[0, 0], ml)


def test_alibi():
    M = N = 5
    D = 2
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    v = torch.randn(N, 1)
    scale = D ** -0.5
    slope = 0.7
    allowed = torch.tril(torch.ones(M, N, dtype=torch.bool))
    bias = torch.zeros(M, N)
    for i in range(M):
        for j in range(N):
            bias[i, j] = -slope * abs(i - j)
    mo, ml = manual_attention(q, k, v, scale, allowed, bias=bias)
    ro, rl = reference_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                 causal=True, alibi_slopes=torch.tensor([slope]))
    cmp("reference alibi out", ro[0, :, 0], mo)
    cmp("reference alibi lse", rl[0, 0], ml)


def test_fully_masked_rows():
    # M=3, N=1, causal: rows 0,1 have no valid key (bottom-right alignment).
    q = torch.randn(3, 2)
    k = torch.randn(1, 2)
    v = torch.randn(1, 4)
    o, lse = reference_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                 causal=True)
    ok_o = bool((o[0, :2] == 0).all()) and not bool((o[0, 2] == 0).all())
    ok_l = bool(torch.isinf(lse[0, 0, :2]).all()) and bool((lse[0, 0, :2] < 0).all()) \
        and bool(torch.isfinite(lse[0, 0, 2]))
    R.ok("reference fully-masked rows -> zero out, -inf lse") if (ok_o and ok_l) else \
        R.fail("reference fully-masked rows", f"o={o[0, :, 0].tolist()} lse={lse[0, 0].tolist()}")


def test_gqa():
    B, M, N, D = 1, 3, 4, 5
    q = torch.randn(B, M, 4, D)
    k = torch.randn(B, N, 2, D)   # Hk=2, H=4 -> group of 2
    v = torch.randn(B, N, 2, D)
    o, lse = reference_attention(q, k, v, causal=True)
    # manual per head
    for h in range(4):
        hk = h // 2
        ro, rl = reference_attention(q[:, :, h:h + 1], k[:, :, hk:hk + 1], v[:, :, hk:hk + 1],
                                     causal=True)
        cmp(f"reference GQA head {h} out", o[:, :, h], ro[:, :, 0])
        cmp(f"reference GQA head {h} lse", lse[:, h], rl[:, 0])


def test_scale():
    q = torch.ones(1, 2, 1, 8)
    k = torch.ones(1, 2, 1, 8)
    v = torch.randn(1, 2, 1, 8)
    o_default, _ = reference_attention(q, k, v)
    o_custom, _ = reference_attention(q, k, v, softmax_scale=2.0)
    # q=k=ones(D=8) -> dot = 8; default scale = D^-0.5 -> logits 8/sqrt(8) = 2*sqrt(2);
    # two equal keys -> lse = logits + log(2).
    _, lse_d = reference_attention(q, k, v)
    _, lse_c = reference_attention(q, k, v, softmax_scale=2.0)
    expect_default = 8.0 / math.sqrt(8.0) + math.log(2.0)
    R.ok("reference default scale = D^-0.5") \
        if abs(float(lse_d[0, 0, 0]) - expect_default) < 1e-4 else \
        R.fail("reference default scale",
               f"lse={float(lse_d[0, 0, 0])} expected {expect_default}")
    # logsumexp over 2 keys with score 2*8=16 -> 16 + log(2)
    expect = 16.0 + math.log(2.0)
    R.ok("reference custom scale affects lse") if abs(float(lse_c[0, 0, 0]) - expect) < 1e-4 else \
        R.fail("reference custom scale", f"lse={float(lse_c[0,0,0])} expected {expect}")


if __name__ == "__main__":
    print("== CPU unit tests for the GPU-suite reference implementation ==")
    test_causal_non_causal()
    test_bottom_right_alignment()
    test_window()
    test_softcap()
    test_alibi()
    test_fully_masked_rows()
    test_gqa()
    test_scale()
    R.finish("test_reference_cpu")
