#!/usr/bin/env python
# ======================================================================================
# test_wrapper_cpu.py — GPU-free wrapper-semantics tests for flash_attn.
#
# How it works
# ------------
#  1. `mock_cuda` fabricates `flash_attn_2_cuda` in sys.modules (mock_cuda.install()),
#     exactly like the compiled extension but with CPU torch reference math.
#  2. The REAL wrapper package (`flash_attn`) is imported and
#     exercised: layout permutes, head-dim padding, GQA, window/softcap/alibi plumbing,
#     return shapes, grad flow through torch.autograd.Function, varlen packing,
#     kvcache in-place update / int->tensor cache_seqlens conversion, etc.
#  3. Numerics are checked against an INDEPENDENT oracle built on torch.softmax /
#     torch.logsumexp over fp32-upcast inputs.
#
# What this does NOT prove
# ------------------------
#  The CUDA kernel itself.  Mock == reference math up to documented approximations
#  (no dropout, no rotary/leftpad/paged).  Kernel numerics need a real V100.
#
# Run:
#   cd <repo root or tests/>
#   PYTHONPATH=<repo root> python test_wrapper_cpu.py
#   (env FA_V100_REPO=/path/to/repo if auto-detection fails)
# ======================================================================================
from __future__ import annotations

import os
import sys
import traceback
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo() -> str:
    cands = [
        os.environ.get("FA_V100_REPO"),
        os.path.dirname(HERE),               # promoted layout: <repo>/tests/...
        os.path.dirname(os.path.dirname(HERE)),
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "flash_attn", "flash_attn_interface.py")):
            return os.path.abspath(c)
    raise RuntimeError("Could not locate the repo root; set FA_V100_REPO=/path/to/flash-attention-v100")


REPO = _find_repo()
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import mock_cuda  # noqa: E402

mock_cuda.install()  # MUST happen before flash_attn is imported anywhere

import flash_attn  # noqa: E402
from flash_attn.flash_attn_interface import (  # noqa: E402
    flash_attn_func,
    flash_attn_gpu,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    round_multiple,
)
from _stub_cuda_backend import install_cpu_dispatch  # noqa: E402

# Upstream's torch.library custom ops are CUDA-only; route the wrapper's calls to
# their Python implementations, which forward to flash_attn_2_cuda (the mock here).
install_cpu_dispatch()

torch.manual_seed(0)

# fp16 out/grads are compared after upcasting; tolerances are dominated by the final
# fp16 quantization of the mock (the mock's internal math is fp32, like the kernel's).
TOL_F16 = dict(atol=2e-2, rtol=2e-2)
TOL_F32 = dict(atol=2e-3, rtol=2e-3)


# ======================================================================================
# independent oracle (torch.softmax / torch.logsumexp based)
# ======================================================================================
def ref_softmax_attention(q, k, v, scale=None, causal=False, window=(-1, -1),
                          softcap=0.0, alibi=None):
    """(B,H,M,D) inputs (fp16 or fp32) -> (out fp32, lse fp32 (B,H,M)).

    Independent re-implementation; mirrors kernel/*.cu semantics:
      causal: j <= i + (N-M); window: i+off-wl <= j <= i+off+wr;
      alibi:  -|i-(j-off_feat)| * slope with off_feat=(N-M) only if causal/window;
      softcap: softcap*tanh(s/softcap); all-masked row -> out 0, lse -inf.
    """
    B, H, M, D = q.shape
    H_K = k.shape[1]
    assert H % H_K == 0, "GQA heads"
    if scale is None:
        scale = float(D) ** -0.5
    kx = k.repeat_interleave(H // H_K, dim=1) if H != H_K else k
    vx = v.repeat_interleave(H // H_K, dim=1) if H != v.shape[1] else v
    N = kx.shape[2]
    offset = N - M

    s = torch.matmul(q.float(), kx.float().transpose(-1, -2)) * float(scale)

    ii = torch.arange(M).view(M, 1)
    jj = torch.arange(N).view(1, N)
    keep = torch.ones(M, N, dtype=torch.bool)
    if causal:
        keep &= jj <= ii + offset
    wl, wr = window
    if wl >= 0:
        keep &= jj >= ii + offset - wl
    if wr >= 0:
        keep &= jj <= ii + offset + wr

    if alibi is not None:
        a = alibi.float()
        a = a.view(1, H, 1, 1) if a.dim() == 1 else a.view(B, H, 1, 1)
        off_feat = offset if (causal or wl >= 0 or wr >= 0) else 0
        s = s + a * (-((ii - jj + off_feat).abs().float()))

    if softcap > 0.0:
        s = float(softcap) * torch.tanh(s / float(softcap))

    keep_b = keep.view(1, 1, M, N)
    s = s.masked_fill(~keep_b, float("-inf"))

    lse = torch.logsumexp(s, dim=-1)                     # (B,H,M); -inf for empty rows
    p = torch.softmax(s, dim=-1)                         # NaN only on fully-masked rows
    p = torch.where(keep_b, p, torch.zeros_like(p))      # scrub NaN in empty rows
    out = torch.matmul(p, vx.float())
    return out, lse


def ref_dense_from_bmhd(q, k, v, dout=None, **kw):
    """q/k/v (B,M,H,D) -> (out (B,M,H,D) fp32, lse (B,H,M)) and optional grads."""
    qh, kh, vh = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    if dout is None:
        out, lse = ref_softmax_attention(qh, kh, vh, **kw)
        return out.permute(0, 2, 1, 3), lse
    qh.requires_grad_(True)
    kh.requires_grad_(True)
    vh.requires_grad_(True)
    out, lse = ref_softmax_attention(qh, kh, vh, **kw)
    g = torch.autograd.grad(out, (qh, kh, vh),
                            grad_outputs=dout.permute(0, 2, 1, 3).float())
    return [x.permute(0, 2, 1, 3).contiguous() for x in g]


def ref_varlen_from_flat(q, k, v, cu_q, cu_k, dout=None, **kw):
    """q (T,H,D), k/v (T,H_K,D) -> (out (T,H,D) fp32, lse (H,T)) / grads."""
    ou, ls = [], []
    params = None
    for b in range(len(cu_q) - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        if qe == qs:
            continue
        qb = q[qs:qe].unsqueeze(0).permute(0, 2, 1, 3).float()
        kb = k[ks:ke].unsqueeze(0).permute(0, 2, 1, 3).float()
        vb = v[ks:ke].unsqueeze(0).permute(0, 2, 1, 3).float()
        if dout is None:
            o, l = ref_softmax_attention(qb, kb, vb, **kw)
            ou.append(o.squeeze(0).permute(1, 0, 2))
        else:
            qb.requires_grad_(True)
            kb.requires_grad_(True)
            vb.requires_grad_(True)
            o, _ = ref_softmax_attention(qb, kb, vb, **kw)
            db = dout[qs:qe].unsqueeze(0).permute(0, 2, 1, 3).float()
            gb = torch.autograd.grad(o, (qb, kb, vb), grad_outputs=db)
            params = params or [[], [], []]
            params[0].append(gb[0].squeeze(0).permute(1, 0, 2))
            params[1].append(gb[1].squeeze(0).permute(1, 0, 2))
            params[2].append(gb[2].squeeze(0).permute(1, 0, 2))
            ls.append(None)
    if dout is None:
        out = torch.cat(ou, dim=0)
        # lse (H,T): recompute per-batch lse in one pass for ordering
        lses = []
        for b in range(len(cu_q) - 1):
            qs, qe = int(cu_q[b]), int(cu_q[b + 1])
            ks, ke = int(cu_k[b]), int(cu_k[b + 1])
            if qe == qs:
                continue
            qb = q[qs:qe].unsqueeze(0).permute(0, 2, 1, 3).float()
            kb = k[ks:ke].unsqueeze(0).permute(0, 2, 1, 3).float()
            vb = v[ks:ke].unsqueeze(0).permute(0, 2, 1, 3).float()
            _, l = ref_softmax_attention(qb, kb, vb, **kw)
            lses.append(l.squeeze(0))            # (H, Lq)
        return out, torch.cat(lses, dim=-1)
    return [torch.cat(g, dim=0) for g in params]


# ======================================================================================
# tiny assertion helpers
# ======================================================================================
def close(name, got, want, f16=True, exact_inf=True):
    kw = TOL_F16 if f16 else TOL_F32
    torch.testing.assert_close(got.float(), want.float(), msg=f"{name}: values differ", **kw)


def half(x):
    return x.half().contiguous()


def rand_bmhd(B, M, H, D, requires_grad=False, dtype=torch.float16):
    x = torch.randn(B, M, H, D, dtype=torch.float32) * 0.5
    x = x.to(dtype).contiguous()
    if requires_grad:
        x.requires_grad_(True)
    return x


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ======================================================================================
# 0. import / alias surface
# ======================================================================================
@test
def test_import_surface():
    assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
    from flash_attn.flash_attn_interface import (  # noqa: F401
        flash_attn_func as f1, flash_attn_varlen_func as f2, flash_attn_with_kvcache as f3,
    )
    assert f1 is flash_attn_func and f2 is flash_attn_varlen_func and f3 is flash_attn_with_kvcache
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401
    import flash_attn_2_cuda  # real extension or the CPU test stub
    assert flash_attn_2_cuda.fwd is not None
    assert hasattr(flash_attn_gpu, "fwd")  # official: the extension module, not a function alias
    assert flash_attn_func is flash_attn.flash_attn_func
    assert flash_attn_varlen_func is flash_attn.flash_attn_varlen_func
    assert flash_attn_with_kvcache is flash_attn.flash_attn_with_kvcache
    # The fork-only aliases no longer exist in the official surface.
    assert not hasattr(flash_attn, "flash_attn_varlen_gpu")
    assert not hasattr(flash_attn, "flash_attn_with_kvcache_gpu")
    assert mock_cuda.is_installed()


# ======================================================================================
# 1. dense forward
# ======================================================================================
@test
def test_dense_forward_basic():
    B, M, H, D = 2, 8, 2, 32
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    out = flash_attn_func(q, k, v)
    assert out.shape == (B, M, H, D) and out.dtype == torch.float16 and out.is_contiguous()
    ref, lse_ref = ref_dense_from_bmhd(q, k, v)
    close("dense out", out, ref)
    # wrapper must hand the extension BHSD, contiguous, padded q
    call = mock_cuda.last_call("fwd")
    assert call is not None
    (_, qshp, _), (_, kshp, _), (_, vshp, _) = call["args"][0:3]
    assert qshp == (B, M, H, D) and kshp == (B, M, H, D) and vshp == (B, M, H, D)


@test
def test_dense_causal_window_combos():
    B, H, D = 2, 2, 64
    cases = [
        dict(M=8, N=8, causal=True, window=(-1, -1)),
        dict(M=8, N=8, causal=True, window=(4, -1)),
        dict(M=8, N=8, causal=True, window=(-1, 2)),
        dict(M=8, N=8, causal=True, window=(2, 2)),
        dict(M=7, N=7, causal=False, window=(3, -1)),
        dict(M=7, N=7, causal=False, window=(-1, 3)),
        dict(M=5, N=9, causal=True, window=(-1, -1)),
        dict(M=8, N=8, causal=False, window=(64, 64)),  # window >= N: disabled by C++
    ]
    for c in cases:
        q = rand_bmhd(B, c["M"], H, D)
        k = rand_bmhd(B, c["N"], H, D)
        v = rand_bmhd(B, c["N"], H, D)
        out = flash_attn_func(q, k, v, causal=c["causal"], window_size=c["window"])
        ref, _ = ref_dense_from_bmhd(q, k, v, causal=c["causal"], window=c["window"])
        close(f"dense {c}", out, ref)


@test
def test_dense_softcap():
    B, M, H, D = 1, 6, 2, 64
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    for causal in (False, True):
        out = flash_attn_func(q, k, v, causal=causal, softcap=50.0)
        ref, _ = ref_dense_from_bmhd(q, k, v, causal=causal, softcap=50.0)
        close(f"softcap causal={causal}", out, ref)
    # softcap + dropout must raise (mirrors kernel check)
    try:
        flash_attn_func(q, k, v, dropout_p=0.1, softcap=50.0)
    except RuntimeError as e:
        assert "Softcapping" in str(e)
    else:
        raise AssertionError("softcap+dropout should raise")


@test
def test_dense_alibi():
    B, M, H, D = 2, 6, 4, 64
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    slopes1 = torch.randn(H, dtype=torch.float32) * 0.1          # [H]
    slopes2 = torch.randn(B, H, dtype=torch.float32) * 0.1       # [B, H]
    for alibi, tag in ((slopes1, "1d"), (slopes2, "2d")):
        for causal in (True, False):
            out = flash_attn_func(q, k, v, causal=causal, alibi_slopes=alibi)
            ref, _ = ref_dense_from_bmhd(q, k, v, causal=causal, alibi=alibi)
            close(f"alibi {tag} causal={causal}", out, ref)
        # hmm: non-causal alibi uses off_feat=0 (mat_mul.h SEQLEN_OFFSET rule)
        out = flash_attn_func(q, k, v, causal=False, window_size=(3, 3), alibi_slopes=alibi)
        ref, _ = ref_dense_from_bmhd(q, k, v, causal=False, window=(3, 3), alibi=alibi)
        close(f"alibi {tag} noncausal+window", out, ref)


@test
def test_dense_gqa():
    B, M, D = 2, 8, 64
    for H, H_K in ((4, 2), (4, 1), (2, 2)):
        q = rand_bmhd(B, M, H, D)
        k = rand_bmhd(B, M, H_K, D)
        v = rand_bmhd(B, M, H_K, D)
        for causal in (True, False):
            out = flash_attn_func(q, k, v, causal=causal)
            ref, _ = ref_dense_from_bmhd(q, k, v, causal=causal)
            close(f"gqa H={H} H_K={H_K} causal={causal}", out, ref)


@test
def test_dense_head_dim_padding():
    B, H = 1, 2
    # (head_dim_og, padded) pairs: padding lands on a dispatchable D
    for og, padded in ((9, 16), (15, 16), (121, 128), (250, 256)):
        M = N = 3
        q = rand_bmhd(B, M, H, og)
        k = rand_bmhd(B, N, H, og)
        v = rand_bmhd(B, N, H, og)
        out = flash_attn_func(q, k, v, causal=True)
        assert out.shape == (B, M, H, og), (og, out.shape)
        ref, _ = ref_dense_from_bmhd(q, k, v, causal=True)
        close(f"pad {og}->{padded}", out, ref)
        # extension must receive the PADDED head dim
        (_, qshp, _) = mock_cuda.last_call("fwd")["args"][0]
        assert qshp == (B, M, H, padded), (og, qshp)
        # and the zero-padding must not change the result vs reference


@test
def test_dense_unsupported_head_dim():
    # Head dims up to 256 are zero-padded to the next instantiated size
    # (16/32/64/128/256), so the only unsupported case is > 256 -- upstream raises a
    # RuntimeError there too (padded head_size must be <= 256).
    B, M, H = 1, 4, 2
    for og in (257, 300):
        q = rand_bmhd(B, M, H, og)
        k = v = q.clone()
        try:
            flash_attn_func(q, k, v)
        except RuntimeError as e:
            assert "Unsupported D" in str(e), (og, str(e))
        else:
            raise AssertionError(f"head_dim {og} should raise Unsupported D")


@test
def test_dense_return_attn_probs():
    B, M, H, D = 2, 8, 2, 32
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    out, lse, dmask = flash_attn_func(q, k, v, return_attn_probs=True)
    assert out.shape == (B, M, H, D)
    assert lse.shape == (B, H, M) and lse.dtype == torch.float32
    assert dmask.numel() == 0, "dropout=0 must yield empty dmask (matches kernel)"
    _, lse_ref = ref_dense_from_bmhd(q, k, v)
    close("lse", lse, lse_ref, f16=False)


# ======================================================================================
# 2. dense backward
# ======================================================================================
@test
def test_dense_backward():
    B, M, H, D = 2, 6, 2, 32
    for causal, window in ((True, (-1, -1)), (False, (2, 2))):
        q = rand_bmhd(B, M, H, D, requires_grad=True)
        k = rand_bmhd(B, M, H, D, requires_grad=True)
        v = rand_bmhd(B, M, H, D, requires_grad=True)
        out = flash_attn_func(q, k, v, causal=causal, window_size=window)
        dout = rand_bmhd(B, M, H, D)
        g = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)
        ref_g = ref_dense_from_bmhd(q.detach(), k.detach(), v.detach(), dout=dout,
                                    causal=causal, window=window)
        for name, got, want in zip(("dq", "dk", "dv"), g, ref_g):
            assert got.shape == want.shape and got.dtype == torch.float16
            close(f"{name} causal={causal}", got, want)
        # wrapper calls bwd with BHSD grads buffers
        call = mock_cuda.last_call("bwd")
        assert call is not None


@test
def test_dense_backward_partial_grads():
    B, M, H, D = 1, 5, 2, 32
    q = rand_bmhd(B, M, H, D)
    k = rand_bmhd(B, M, H, D)
    v = rand_bmhd(B, M, H, D, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    (dv,) = torch.autograd.grad(out, (v,), grad_outputs=rand_bmhd(B, M, H, D))
    assert dv.shape == (B, M, H, D)


@test
def test_dense_no_grad():
    B, M, H, D = 1, 4, 2, 32
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    with torch.no_grad():
        out = flash_attn_func(q, k, v, causal=True)
    assert out.requires_grad is False


# ======================================================================================
# 3. dense edges
# ======================================================================================
@test
def test_dense_edges_seqlens():
    H, D = 2, 64
    cases = [
        dict(B=1, M=1, N=1, causal=True),
        dict(B=1, M=1, N=8, causal=True),          # decode: attends all
        dict(B=1, M=4, N=1, causal=True),          # first rows fully masked
        dict(B=1, M=8, N=8, causal=False),
    ]
    for c in cases:
        q = rand_bmhd(c["B"], c["M"], H, D)
        k = rand_bmhd(c["B"], c["N"], H, D)
        v = rand_bmhd(c["B"], c["N"], H, D)
        out = flash_attn_func(q, k, v, causal=c["causal"])
        ref, lse_ref = ref_dense_from_bmhd(q, k, v, causal=c["causal"])
        close(f"edge {c}", out, ref)
        _, lse, _ = flash_attn_func(q, k, v, causal=c["causal"], return_attn_probs=True)
        close(f"edge lse {c}", lse, lse_ref, f16=False)
        if c["M"] > c["N"] and c["causal"]:
            # rows with no valid key must be exactly zero / -inf. out is (B,M,H,D) while
            # lse_ref is (B,H,M): permute out to (B,H,M,D) so the row order matches.
            nmask = int((~torch.isfinite(lse_ref)).sum())
            assert nmask > 0
            out_bhm = out.permute(0, 2, 1, 3).reshape(-1, D)
            assert torch.all(out_bhm[torch.isfinite(lse_ref).reshape(-1) == 0] == 0)


@test
def test_dense_non_multiple_of_64():
    B, M, N, H, D = 3, 37, 53, 3, 64
    q = rand_bmhd(B, M, H, D)
    k = rand_bmhd(B, N, H, D)
    v = rand_bmhd(B, N, H, D)
    out = flash_attn_func(q, k, v, causal=True)
    ref, _ = ref_dense_from_bmhd(q, k, v, causal=True)
    close("37x53", out, ref)


@test
def test_dense_b1h1():
    q = rand_bmhd(1, 16, 1, 64)
    k = rand_bmhd(1, 16, 1, 64)
    v = rand_bmhd(1, 16, 1, 64)
    out = flash_attn_func(q, k, v, causal=True)
    ref, _ = ref_dense_from_bmhd(q, k, v, causal=True)
    close("b1h1", out, ref)


@test
def test_dense_dropout_shape_only():
    B, M, H, D = 1, 4, 2, 32
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    mock_cuda.ALLOW_DROPOUT = True
    try:
        out, lse, dmask = flash_attn_func(q, k, v, dropout_p=0.1, return_attn_probs=True)
        assert out.shape == (B, M, H, D)
        # F-6 (research/07): dense S_dmask is returned in upstream's padded shape
        # (B, H, round128(M), round128(N)); the mock's keep-all block must be copied
        # into the top-left corner and everything else must be zero.
        N = M  # dense test: q/k/v all have sequence length M
        M_pad, N_pad = round_multiple(M, 128), round_multiple(N, 128)
        assert dmask.shape == (B, H, M_pad, N_pad) and dmask.dtype == torch.float16
        assert dmask[..., :M, :M].count_nonzero() == B * H * M * M  # mock keep-all = 1
        assert dmask[..., :M, M:].count_nonzero() == 0
        assert dmask[..., M:, :].count_nonzero() == 0
        assert lse.shape == (B, H, M)
    finally:
        mock_cuda.ALLOW_DROPOUT = False
    # NOTE: values NOT compared — the mock does not simulate dropout.


@test
def test_deterministic_no_warning():
    # Official 2.8.3 accepts deterministic=True silently in the forward pass; the
    # flag only selects the backward algorithm.
    q, k, v = (rand_bmhd(1, 4, 2, 32) for _ in range(3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = flash_attn_func(q, k, v, deterministic=True, causal=True)
    assert not any(issubclass(x.category, RuntimeWarning) for x in w), [x.category for x in w]
    assert out.shape == q.shape


@test
def test_sdpa_cross_check():
    """Third-party sanity check of the plain path against torch SDPA (fp32 CPU)."""
    import torch.nn.functional as F

    B, M, H, D = 1, 8, 2, 64
    q, k, v = (rand_bmhd(B, M, H, D) for _ in range(3))
    for causal in (False, True):
        out = flash_attn_func(q, k, v, causal=causal)
        qh, kh, vh = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
        ref = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
        close(f"sdpa causal={causal}", out, ref.permute(0, 2, 1, 3), f16=True)


# ======================================================================================
# 4. varlen
# ======================================================================================
def _varlen_inputs(lens_q, lens_k, H, H_K, D, dtype=torch.float16):
    Tq = sum(lens_q)
    Tk = sum(lens_k)
    q = rand_bmhd(1, Tq, H, D).reshape(Tq, H, D)
    k = rand_bmhd(1, Tk, H_K, D).reshape(Tk, H_K, D)
    v = rand_bmhd(1, Tk, H_K, D).reshape(Tk, H_K, D)
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_q, dtype=torch.int64), 0).tolist()),
                        dtype=torch.int64)
    cu_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_k, dtype=torch.int64), 0).tolist()),
                        dtype=torch.int64)
    return q, k, v, cu_q, cu_k


@test
def test_varlen_basic_and_cu_forwarding():
    lens_q = [3, 1, 7]
    lens_k = [3, 1, 9]
    q, k, v, cu_q, cu_k = _varlen_inputs(lens_q, lens_k, 2, 2, 32)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), causal=True)
    assert out.shape == (sum(lens_q), 2, 32) and out.dtype == torch.float16
    ref, lse_ref = ref_varlen_from_flat(q, k, v, cu_q, cu_k, causal=True)
    close("varlen", out, ref)
    # Official 2.8.3 forwards cu_seqlens unchanged; it does not cast int64 -> int32.
    call = mock_cuda.last_call("varlen_fwd")
    (_, cuq_shp, cuq_dtype) = call["args"][4]
    assert cuq_dtype == str(cu_q.dtype) and cuq_shp == (4,)
    # The V100 compat shim may round max_seqlen up to a kernel alignment.
    assert call["args"][10] >= max(lens_q) and call["args"][11] >= max(lens_k)


@test
def test_varlen_lse_shape():
    lens = [4, 2]
    q, k, v, cu_q, cu_k = _varlen_inputs(lens, lens, 2, 2, 64)
    out, lse, dmask = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens), max(lens),
                                             causal=True, return_attn_probs=True)
    assert lse.shape == (2, 6) and lse.dtype == torch.float32
    assert dmask.numel() == 0
    _, lse_ref = ref_varlen_from_flat(q, k, v, cu_q, cu_k, causal=True)
    close("varlen lse", lse, lse_ref, f16=False)


@test
def test_varlen_gqa_and_features():
    lens_q = [5, 3]
    lens_k = [7, 3]
    q, k, v, cu_q, cu_k = _varlen_inputs(lens_q, lens_k, 4, 1, 64)
    slopes = torch.randn(4) * 0.1
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k),
                                 causal=True, window_size=(3, 2), softcap=40.0,
                                 alibi_slopes=slopes)
    ref, _ = ref_varlen_from_flat(q, k, v, cu_q, cu_k, causal=True, window=(3, 2),
                                  softcap=40.0, alibi=slopes)
    close("varlen gqa+features", out, ref)


@test
def test_varlen_backward():
    lens = [3, 5]
    q, k, v, cu_q, cu_k = _varlen_inputs(lens, lens, 2, 2, 32)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens), max(lens), causal=True)
    dout = rand_bmhd(1, sum(lens), 2, 32).reshape(sum(lens), 2, 32)
    g = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)
    ref_g = ref_varlen_from_flat(q.detach(), k.detach(), v.detach(), cu_q, cu_k, dout=dout, causal=True)
    for name, got, want in zip(("dq", "dk", "dv"), g, ref_g):
        close(f"varlen {name}", got, want)


@test
def test_varlen_zero_len_batch():
    lens_q = [1, 0, 4]
    lens_k = [2, 0, 4]
    q, k, v, cu_q, cu_k = _varlen_inputs(lens_q, lens_k, 2, 2, 32)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), causal=True)
    assert out.shape == (5, 2, 32)
    ref, _ = ref_varlen_from_flat(q, k, v, cu_q, cu_k, causal=True)
    close("varlen zero-len", out, ref)


# ======================================================================================
# 5. kvcache
# ======================================================================================
def _kvcache_inputs(B, N, T_Q, H, H_K, D, cache_seqlens, T_new=0, dtype=torch.float16):
    q = rand_bmhd(B, T_Q, H, D)
    k_cache = rand_bmhd(B, N, H_K, D)
    v_cache = rand_bmhd(B, N, H_K, D)
    k_new = rand_bmhd(B, T_new, H_K, D) if T_new else None
    v_new = rand_bmhd(B, T_new, H_K, D) if T_new else None
    return q, k_cache, v_cache, k_new, v_new


def _ref_kvcache(q, k_cache, v_cache, cache_seqlens, causal, window, softcap=0.0,
                 alibi=None, k_new=None, v_new=None):
    """B=1 reference only (per-batch loop) -> out (B,T,H,D) fp32, lse (B,H,T)."""
    B, T_Q, H, D = q.shape
    outs, lses = [], []
    for b in range(B):
        total = int(cache_seqlens[b]) + (0 if k_new is None else k_new.shape[1])
        qb = q[b:b + 1].permute(0, 2, 1, 3).float()          # (1,H,T,D)
        kb = k_cache[b:b + 1, :total].clone()
        vb = v_cache[b:b + 1, :total].clone()
        if k_new is not None:
            kb[:, int(cache_seqlens[b]):total] = k_new[b].float()
            vb[:, int(cache_seqlens[b]):total] = v_new[b].float()
        kb = kb.permute(0, 2, 1, 3)                          # (1,H,S,D)
        vb = vb.permute(0, 2, 1, 3)
        if total == 0:
            outs.append(torch.zeros(1, T_Q, H, D))
            lses.append(torch.full((1, H, T_Q), float("inf")))  # empty KV -> +inf (F-8)
            continue
        ab = alibi[b].view(1, H) if (alibi is not None and alibi.dim() == 2) else alibi
        o, l = ref_softmax_attention(qb, kb, vb, causal=causal, window=window,
                                     softcap=softcap, alibi=ab)
        outs.append(o.permute(0, 2, 1, 3))
        lses.append(l)
    return torch.cat(outs, 0), torch.cat(lses, 0)


@test
def test_kvcache_decode_no_new_kv():
    B, N, T_Q, H, D = 2, 16, 1, 2, 64
    q, k_cache, v_cache, _, _ = _kvcache_inputs(B, N, T_Q, H, H, D, None)
    seqlens = torch.tensor([16, 9], dtype=torch.int32)
    out = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=seqlens, causal=True)
    assert out.shape == (B, T_Q, H, D)
    ref, lse_ref = _ref_kvcache(q, k_cache, v_cache, [16, 9], True, (-1, -1))
    close("kvcache decode", out, ref)
    # cache must be untouched when k/v are None
    call = mock_cuda.last_call("fwd_kvcache")
    assert call["args"][4] is None  # v_new is None -> wrapper passed k=None,v=None


@test
def test_kvcache_int_seqlens_and_inplace_update():
    B, N, T_Q, H, D = 1, 12, 3, 2, 64
    T_new = 3
    q, k_cache, v_cache, k_new, v_new = _kvcache_inputs(B, N, T_Q, H, H, D, None, T_new=T_new)
    k_cache.zero_()
    v_cache.zero_()
    out = flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new,
                                  cache_seqlens=5, causal=False)
    # int cache_seqlens must have been converted to an int32 tensor of shape (B,)
    call = mock_cuda.last_call("fwd_kvcache")
    seqlens_arg = call["args"][5]
    assert seqlens_arg[0] == "Tensor" and seqlens_arg[1] == (B,) and seqlens_arg[2] == "torch.int32"
    # in-place cache update happened at rows [5, 8)
    assert torch.equal(k_cache[0, 5:8], k_new[0])
    assert torch.equal(v_cache[0, 5:8], v_new[0])
    assert k_cache[0, :5].abs().sum() == 0
    # attention spans total = 8 keys; the cache already contains k_new/v_new at rows
    # [5, 8) (updated above), so the reference must NOT append them a second time.
    ref, _ = _ref_kvcache(q, k_cache, v_cache, [8], False, (-1, -1))
    close("kvcache inplace", out, ref)
    out2 = flash_attn_with_kvcache(q, k_cache.clone() * 0, v_cache.clone() * 0, k=None, v=None,
                                   cache_seqlens=torch.tensor([8], dtype=torch.int32), causal=False)
    assert out2.shape == (B, T_Q, H, D)


@test
def test_kvcache_return_lse_and_causal_window():
    B, N, T_Q, H, D = 2, 10, 2, 2, 32
    q, k_cache, v_cache, _, _ = _kvcache_inputs(B, N, T_Q, H, H, D, None)
    seqlens = [8, 6]
    out, lse = flash_attn_with_kvcache(q, k_cache, v_cache,
                                       cache_seqlens=torch.tensor(seqlens, dtype=torch.int32),
                                       causal=True, window_size=(-1, 3),
                                       return_softmax_lse=True)
    assert lse.shape == (B, H, T_Q) and lse.dtype == torch.float32
    ref, lse_ref = _ref_kvcache(q, k_cache, v_cache, seqlens, True, (-1, 3))
    close("kvcache w", out, ref)
    close("kvcache lse", lse, lse_ref, f16=False)


@test
def test_kvcache_gqa():
    B, N, T_Q, H, H_K, D = 1, 8, 4, 4, 1, 64
    q, k_cache, v_cache, _, _ = _kvcache_inputs(B, N, T_Q, H, H_K, D, None)
    out = flash_attn_with_kvcache(q, k_cache, v_cache,
                                  cache_seqlens=torch.tensor([8], dtype=torch.int32), causal=False)
    ref, _ = _ref_kvcache(q, k_cache, v_cache, [8], False, (-1, -1))
    close("kvcache gqa", out, ref)


@test
def test_kvcache_none_seqlens_attends_full_cache():
    """F-3 (research/07): cache_seqlens=None means "attend the FULL cache" (upstream),
    not "empty cache" as the old fork did. Both calls must attend; None must be
    numerically identical to passing the explicit full length.
    """
    B, N, T_Q, H, D = 1, 8, 2, 2, 32
    q, k_cache, v_cache, _, _ = _kvcache_inputs(B, N, T_Q, H, H, D, None)
    out_none, lse_none = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=None,
                                                 return_softmax_lse=True)
    out_full, lse_full = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=torch.tensor([N], dtype=torch.int32),
        return_softmax_lse=True)
    assert out_none.abs().sum() != 0, "cache_seqlens=None must attend the full cache"
    assert torch.equal(out_none, out_full), "None must equal explicit full length"
    assert torch.equal(lse_none, lse_full)
    ref, lse_ref = _ref_kvcache(q, k_cache, v_cache, [N], False, (-1, -1))
    close("kvcache none==full", out_none, ref)
    close("kvcache none lse", lse_none, lse_ref, f16=False)


@test
def test_kvcache_stride_assert():
    B, N, T_Q, H, D = 1, 8, 2, 2, 32
    q = rand_bmhd(B, T_Q, H, D)
    # Shape stays (B, N, H, D) but the last dimension is strided by 2, which is what
    # the wrapper's "contiguous last dimension" assert must reject.
    k_cache = rand_bmhd(B, N, H, 2 * D)[..., ::2]
    v_cache = rand_bmhd(B, N, H, D)
    try:
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=torch.tensor([8], dtype=torch.int32))
    except AssertionError:
        pass
    else:
        raise AssertionError("non-contiguous k_cache must trip the wrapper assert")


# ======================================================================================
# runner
# ======================================================================================
def main(argv=None):
    argv = argv or sys.argv[1:]
    selected = TESTS if not argv else [t for t in TESTS if t.__name__ in set(argv)]
    failed = 0
    for t in selected:
        mock_cuda.CALL_LOG.clear()
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(selected) - failed}/{len(selected)} passed")
    print(f"== test_wrapper_cpu: {len(selected) - failed}/{len(selected)} passed, "
          f"{failed} failed, 0 skipped ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
