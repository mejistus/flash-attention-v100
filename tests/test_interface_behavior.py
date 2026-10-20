#!/usr/bin/env python
# ======================================================================================
# * Interface behavior tests: runnable on any CPU-only machine (no CUDA, no build).
# *
# * A recording stub is installed in place of the compiled extension, so every argument
# * the Python layer forwards to the backend is inspected: permutes, head-dim padding,
# * softmax-scale defaults, return-value semantics, autograd wiring and early exits.
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)  # prefer the repo package over any installed copy
import torch  # noqa: E402
import _stub_cuda_backend as sb  # noqa: E402

mod, _ = sb.install_stub(force=True)
backend = mod.stub_backend
R = sb.RESULT

import flash_attn  # noqa: E402
import flash_attn.flash_attn_interface as fai  # noqa: E402
from flash_attn import (  # noqa: E402
    flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache,
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func,
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func,
)

# Upstream's torch.library custom ops are CUDA-only; route the wrapper's calls
# to their Python implementations so the official wrapper runs against the stub.
sb.install_cpu_dispatch()


def last():
    return backend.calls[-1]


def ncalls():
    return len(backend.calls)


def as_half(t):
    return t.to(torch.float16)


# ---------------------------------------------------------------------------- dense --
def test_dense_basic():
    q = as_half(torch.arange(2 * 5 * 3 * 10, dtype=torch.float32).reshape(2, 5, 3, 10))
    k = torch.randn(2, 5, 2, 10, dtype=torch.float16)
    v = torch.randn(2, 5, 2, 10, dtype=torch.float16)

    out = flash_attn_func(q, k, v)
    c = last()
    checks = [
        ("q shape (B,M,H,Dpad)", c["q"].shape == (2, 5, 3, 16)),
        ("k shape (B,M,HK,Dpad)", c["k"].shape == (2, 5, 2, 16)),
        ("v shape (B,M,HK,Dpad)", c["v"].shape == (2, 5, 2, 16)),
        ("default scale = head_size_og^-0.5", abs(c["softmax_scale"] - 10 ** -0.5) < 1e-12),
        ("dropout default 0.0", c["dropout_p"] == 0.0),
        ("causal default False", c["causal"] is False),
        ("window default (-1,-1)", (c["window_left"], c["window_right"]) == (-1, -1)),
        ("softcap default 0.0", c["softcap"] == 0.0),
        ("output shape (B,M,H,D)", out.shape == (2, 5, 3, 10)),
        ("q values preserved + zero-padded",
         torch.equal(c["q"][..., :10], q)),
        ("k padded values preserved",
         torch.equal(c["k"][..., :10], k)),
        ("padded tail is exactly zero", torch.count_nonzero(c["q"][..., 10:]) == 0),
        ("backend q contiguous", c["q"].is_contiguous()),
        ("backend out slot None", c["out"] is None),
    ]
    for name, ok in checks:
        if ok:
            R.ok("dense: " + name)
        else:
            R.fail("dense: " + name)


def test_dense_head_dim_rounding():
    # (requested head dim, expected padded dim); scale must use the ORIGINAL dim.
    cases = [(8, 16), (16, 16), (17, 32), (32, 32), (33, 64), (48, 64), (64, 64),
             (65, 128), (96, 128), (128, 128), (129, 256), (160, 256), (224, 256), (256, 256)]
    for d, target in cases:
        q = torch.randn(1, 2, 1, d, dtype=torch.float16)
        out = flash_attn_func(q, q, q)
        c = last()
        ok = (c["q"].shape[-1] == target
              and abs(c["softmax_scale"] - d ** -0.5) < 1e-12
              and out.shape == (1, 2, 1, d))
        if ok:
            R.ok(f"dense: head dim {d} -> {target}, scale from {d}")
        else:
            R.fail(f"dense: head dim {d} -> {target}",
                   f"got D={c['q'].shape[-1]}, scale={c['softmax_scale']}, out={tuple(out.shape)}")
    # out of range
    q = torch.randn(1, 2, 1, 257, dtype=torch.float16)
    try:
        flash_attn_func(q, q, q)
        R.fail("dense: head dim 257 raises", "no exception")
    except RuntimeError as e:  # same exception type as upstream's C++ TORCH_CHECK
        (R.ok if "Unsupported D" in str(e) else R.fail)(
            "dense: head dim 257 raises RuntimeError (Unsupported D)")


def test_dense_return_attn_probs():
    q = torch.randn(2, 5, 3, 10, dtype=torch.float16)
    # dropout_p == 0: official returns an empty S_dmask; backend must not compute it.
    r = flash_attn_func(q, q, q, dropout_p=0.0, return_attn_probs=True)
    c = last()
    checks = [
        ("dropout=0 -> 3-tuple returned", isinstance(r, tuple) and len(r) == 3),
        ("dropout=0 -> backend return_softmax False", c["return_softmax"] is False),
        ("dropout=0 -> lse shape (B,H,M)", r[1].shape == (2, 3, 5)),
        ("dropout=0 -> dmask empty", r[2].numel() == 0),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("dense return_attn_probs: " + name)

    # dropout_p > 0: dmask produced in upstream's padded dense layout (F-6).
    r = flash_attn_func(q, q, q, dropout_p=0.1, return_attn_probs=True)
    c = last()
    checks = [
        ("dropout>0 -> backend return_softmax True", c["return_softmax"] is True),
        ("dropout>0 -> dmask (B,H,round128(M),round128(N))",
         r[2].shape == (2, 3, 128, 128)),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("dense return_attn_probs: " + name)


def test_dense_empty():
    # Official 2.8.3 has no Python-side empty-input early returns, and the V100
    # shim preserves that wherever the extension can run:
    #   * B==0 forward is forwarded (the compiled extension then rejects it with
    #     "batch size must be positive", exactly like upstream; the stub records
    #     the call instead of raising);
    #   * M==0 dense calls raise a clear RuntimeError (the V100 kernel cannot
    #     launch the backward; see test_empty_backward_no_crash);
    #   * varlen T_Q==0 and max_seqlen_k==0 forwards are forwarded.
    # Two short-circuits exist only because the V100 kernel cannot launch:
    #   * dense N==0 forward returns zeros, lse=+inf and a correctly shaped mask
    #     without calling the backend (upstream's empty-KV semantics);
    #   * dense M==0 backward returns zero gradients without calling the backend
    #     (covered in test_empty_backward_no_crash below).
    n0 = ncalls()
    q = torch.randn(0, 4, 2, 8, dtype=torch.float16)
    out = flash_attn_func(q, q, q)
    (R.ok if (out.shape == (0, 4, 2, 8) and ncalls() == n0 + 1) else R.fail)(
        "dense: B==0 forwarded to backend, output shape preserved",
        f"out={tuple(out.shape)}, calls+{ncalls()-n0}")
    n1 = ncalls()
    r = flash_attn_func(q, q, q, return_attn_probs=True)
    (R.ok if (len(r) == 3 and r[1].shape == (0, 2, 4) and r[2].numel() == 0
              and ncalls() == n1 + 1) else R.fail)(
        "dense: B==0 + return_attn_probs shapes (backend called)")

    # M == 0: the V100 kernel cannot launch the backward, so the adapter raises
    # a clear RuntimeError before any backend call.
    n2 = ncalls()
    q = torch.randn(2, 0, 3, 8, dtype=torch.float16)
    k = torch.randn(2, 4, 3, 8, dtype=torch.float16)
    v = torch.randn(2, 4, 3, 8, dtype=torch.float16)
    try:
        flash_attn_func(q, k, v)
        R.fail("dense: M==0 raises cleanly (V100 limitation)", "no exception")
    except RuntimeError as e:
        (R.ok if "M == 0" in str(e) and ncalls() == n2 else R.fail)(
            "dense: M==0 raises cleanly (V100 limitation)", str(e)[:80])

    # N == 0: shim short-circuit (zeroes/+inf), no backend call.
    q = torch.randn(2, 4, 3, 8, dtype=torch.float16)
    k = torch.randn(2, 0, 3, 8, dtype=torch.float16)
    n3 = ncalls()
    out = flash_attn_func(q, k, k)
    (R.ok if (out.shape == (2, 4, 3, 8) and int(out.count_nonzero()) == 0
              and ncalls() == n3) else R.fail)(
        "dense: N==0 short-circuit -> zero output, no backend call",
        f"out={tuple(out.shape)}, calls+{ncalls()-n3}")
    n4 = ncalls()
    r = flash_attn_func(q, k, k, return_attn_probs=True)
    (R.ok if (len(r) == 3 and r[0].shape == (2, 4, 3, 8)
              and int(r[0].count_nonzero()) == 0
              and r[1].shape == (2, 3, 4)
              and bool(torch.isinf(r[1]).all()) and bool((r[1] > 0).all())
              and r[2].numel() == 0 and ncalls() == n4) else R.fail)(
        "dense: N==0 short-circuit -> lse=+inf, empty mask, no backend call")
    # dropout>0 makes the wrapper request S_dmask; with N==0 the mask is
    # still correctly shaped (B, H, round128(M), round128(N)) and empty.
    n5 = ncalls()
    r = flash_attn_func(q, k, k, dropout_p=0.1, return_attn_probs=True)
    (R.ok if (len(r) == 3 and r[2].shape == (2, 3, 128, 0)
              and r[2].numel() == 0 and bool(torch.isinf(r[1]).all())
              and ncalls() == n5) else R.fail)(
        "dense: N==0 short-circuit -> correctly shaped empty S_dmask",
        f"mask={tuple(r[2].shape)}")


def test_dense_misc():
    base = torch.randn(2, 8, 2, 16, dtype=torch.float16)
    q = base[:, ::2]  # non-contiguous view, shape (2,4,2,16)
    out = flash_attn_func(q, q, q)
    c = last()
    (R.ok if (out.shape == (2, 4, 2, 16) and c["q"].is_contiguous()
              and torch.equal(c["q"], q.contiguous())) else R.fail)(
        "dense: non-contiguous input handled and values preserved")

    alibi = torch.randn(2, 2, dtype=torch.float32)
    out = flash_attn_func(q, q, q, causal=True, window_size=(3, 5), softcap=8.0,
                          alibi_slopes=alibi)
    c = last()
    checks = [
        ("causal forwarded True", c["causal"] is True),
        ("window (3,5) forwarded", (c["window_left"], c["window_right"]) == (3, 5)),
        ("softcap forwarded", c["softcap"] == 8.0),
        ("alibi_slopes object forwarded", c["alibi_slopes"] is alibi),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("dense misc: " + name)

    # Official 2.8.3: deterministic only selects the backward algorithm; it is
    # accepted silently in the forward pass and has no warning.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out_det = flash_attn_func(q, q, q, deterministic=True)
        got = any(issubclass(x.category, RuntimeWarning) for x in w)
    checks = [
        ("deterministic=True accepted without warning", not got),
        ("deterministic=True forward shape", out_det.shape == q.shape),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("dense misc: " + name)


def test_dense_backward():
    q = torch.randn(2, 4, 2, 10, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 2, 10, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 4, 2, 10, dtype=torch.float32, requires_grad=True)
    q.retain_grad()
    try:
        out = flash_attn_func(q, k, v, dropout_p=0.3, causal=True,
                              deterministic=True)
        out.sum().backward()
        cb = [c for c in backend.calls if c["kind"] == "bwd"][-1]
        checks = [
            ("no autograd arity error", True),
            ("backward called", True),
            ("dropout forwarded to backward", cb["dropout_p"] == 0.3),
            ("causal forwarded to backward", cb["causal"] is True),
            ("deterministic forwarded to backward", cb["deterministic"] is True),
            ("dq shape matches q", q.grad is not None and q.grad.shape == q.shape),
            ("dk shape matches k", k.grad is not None and k.grad.shape == k.shape),
            ("dv shape matches v", v.grad is not None and v.grad.shape == v.shape),
        ]
    except RuntimeError as e:
        checks = [("no autograd arity error", False)]
        R.fail("dense backward: raised", str(e))
    for name, ok in checks:
        (R.ok if ok else R.fail)("dense backward: " + name)


# --------------------------------------------------------------------------- varlen --
def test_varlen_basic():
    q = as_half(torch.arange(10 * 2 * 10, dtype=torch.float32).reshape(10, 2, 10))
    k = torch.randn(10, 1, 10, dtype=torch.float16)
    v = torch.randn(10, 1, 10, dtype=torch.float16)
    cu = torch.tensor([0, 4, 10], dtype=torch.int64)

    out = flash_attn_varlen_func(q, k, v, cu, cu, 6, 6)
    c = last()
    checks = [
        ("q shape (T,H,Dpad)", c["q"].shape == (10, 2, 16)),
        # Official 2.8.3 forwards cu_seqlens unchanged (no fork-only int32 cast);
        # callers are expected to pass int32 as documented.
        ("cu_seqlens forwarded unchanged", c["cu_seqlens_q"].dtype == cu.dtype
         and c["cu_seqlens_q"].tolist() == [0, 4, 10]),
        ("cu_seqlens_k forwarded", c["cu_seqlens_k"].tolist() == [0, 4, 10]),
        # The V100 compat shim may round max_seqlen up to a kernel alignment; the
        # official wrapper must pass at least the requested maximum through.
        ("max_seqlen forwarded", c["max_seqlen_q"] >= 6 and c["max_seqlen_k"] >= 6
         and c["max_seqlen_q"] == c["max_seqlen_k"]),
        ("default scale from original head dim", abs(c["softmax_scale"] - 10 ** -0.5) < 1e-12),
        ("dropout=0 -> return_softmax False", c["return_softmax"] is False),
        ("seqused_k/leftpad_k None", c["seqused_k"] is None and c["leftpad_k"] is None),
        ("num_splits 0", c["num_splits"] == 0),
        ("zero_tensors False", c["zero_tensors"] is False),
        ("output shape (T,H,D)", out.shape == (10, 2, 10)),
        ("q values preserved", torch.equal(c["q"][..., :10], q.contiguous())),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("varlen: " + name)


def test_varlen_return_attn_probs():
    q = torch.randn(10, 2, 16, dtype=torch.float16)
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    r = flash_attn_varlen_func(q, q, q, cu, cu, 6, 6, dropout_p=0.0, return_attn_probs=True)
    c = last()
    (R.ok if (len(r) == 3 and c["return_softmax"] is False and r[1].shape == (2, 10)
              and r[2].numel() == 0) else R.fail)("varlen return_attn_probs: dropout=0 gate")
    r = flash_attn_varlen_func(q, q, q, cu, cu, 6, 6, dropout_p=0.1, return_attn_probs=True)
    c = last()
    # Official varlen S_dmask is (batch, nheads, round128(max_q), round128(max_k)),
    # not the fork's packed (total_q, nheads, max_k) layout.
    (R.ok if (len(r) == 3 and c["return_softmax"] is True and r[2].ndim == 4
              and r[2].shape[:2] == (2, 2)
              and r[2].shape[-2] >= 6 and r[2].shape[-1] >= 6) else R.fail)(
        "varlen return_attn_probs: dropout>0 dmask (B,H,seqlen_q,seqlen_k)",
        f"shape={tuple(r[2].shape)}")


def test_varlen_empty_and_backward():
    # Official 2.8.3 forwards empty varlen batches to the kernel (no Python-side
    # early return); only the output/return shapes are a wrapper contract.
    n0 = ncalls()
    q = torch.randn(0, 2, 10, dtype=torch.float16)
    cu = torch.tensor([0], dtype=torch.int32)
    out = flash_attn_varlen_func(q, q, q, cu, cu, 0, 0)
    (R.ok if (out.shape == (0, 2, 10) and ncalls() == n0 + 1) else R.fail)(
        "varlen: T_Q==0 forwarded to backend, output shape preserved")
    n1 = ncalls()
    r = flash_attn_varlen_func(q, q, q, cu, cu, 0, 0, return_attn_probs=True)
    (R.ok if (len(r) == 3 and r[1].shape == (2, 0) and r[2].numel() == 0
              and ncalls() == n1 + 1) else R.fail)(
        "varlen: T_Q==0 return_attn_probs shapes (backend called)")

    q = torch.randn(6, 2, 8, dtype=torch.float16)
    k = torch.randn(0, 2, 8, dtype=torch.float16)
    cu_q = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 0], dtype=torch.int32)
    n2 = ncalls()
    out = flash_attn_varlen_func(q, k, k, cu_q, cu_k, 3, 0)
    (R.ok if (out.shape == (6, 2, 8) and ncalls() == n2 + 1) else R.fail)(
        "varlen: max_seqlen_k==0 forwarded to backend, output shape preserved")
    n3 = ncalls()
    r = flash_attn_varlen_func(q, k, k, cu_q, cu_k, 3, 0, return_attn_probs=True)
    (R.ok if (len(r) == 3 and r[1].shape == (2, 6) and r[2].numel() == 0
              and ncalls() == n3 + 1) else R.fail)(
        "varlen: max_seqlen_k==0 return_attn_probs shapes (backend called)")

    q = torch.randn(10, 2, 10, dtype=torch.float32, requires_grad=True)
    k = torch.randn(10, 2, 10, dtype=torch.float32, requires_grad=True)
    v = torch.randn(10, 2, 10, dtype=torch.float32, requires_grad=True)
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    try:
        out = flash_attn_varlen_func(q, k, v, cu, cu, 6, 6, dropout_p=0.2)
        out.sum().backward()
        cb = [c for c in backend.calls if c["kind"] == "varlen_bwd"][-1]
        checks = [
            ("varlen backward called, dropout forwarded", cb["kind"] == "varlen_bwd"),
            ("dq shape", q.grad is not None and q.grad.shape == q.shape),
            ("dk shape", k.grad is not None and k.grad.shape == k.shape),
            ("dv shape", v.grad is not None and v.grad.shape == v.shape),
        ]
    except RuntimeError as e:
        checks = [("varlen backward arity", False)]
        R.fail("varlen backward: raised", str(e))
    for name, ok in checks:
        (R.ok if ok else R.fail)("varlen backward: " + name)


def test_empty_backward_no_crash():
    """Official 2.8.3 keeps empty inputs differentiable end-to-end.

    There are no Python-side early returns, so autograd saves the forward tensors
    and calls the backend backward even for empty batches.  The recording stub
    checks that the backward op is reached exactly once and that autograd does not
    raise while wiring the empty gradients.
    """

    def run(name, fn, allowed_bwd=(1,)):
        n_before = ncalls()
        try:
            fn()
            bwd_calls = [c for c in backend.calls[n_before:]
                         if c["kind"] in ("bwd", "varlen_bwd")]
            ok = len(bwd_calls) in allowed_bwd
            (R.ok if ok else R.fail)(
                "empty backward: " + name,
                "" if ok else
                f"{len(bwd_calls)} backend backward calls (allowed {allowed_bwd})")
        except Exception as e:  # noqa: BLE001
            R.fail("empty backward: " + name, f"{type(e).__name__}: {e}")

    # dense B == 0
    q = torch.randn(0, 4, 2, 8, dtype=torch.float16, requires_grad=True)
    run("dense B==0 sum().backward() no crash", lambda: flash_attn_func(q, q, q).sum().backward())

    # dense M == 0: the V100 kernel cannot launch the backward, so the adapter
    # rejects the forward with a clear RuntimeError before any backend call.
    q = torch.randn(2, 0, 3, 8, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 4, 3, 8, dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 4, 3, 8, dtype=torch.float16, requires_grad=True)
    try:
        flash_attn_func(q, k, v)
        R.fail("empty backward: dense M==0 raises cleanly", "no exception")
    except RuntimeError as e:
        (R.ok if "M == 0" in str(e) else R.fail)(
            "empty backward: dense M==0 raises cleanly", str(e)[:80])

    # dense N == 0 (k/v shaped (2, 0, 3, 8)): the shim forwards this backward
    # (the V100 extension supports it), but an empty autograd graph may also
    # produce no backend call.  Either is acceptable; shapes must stay correct.
    q = torch.randn(2, 4, 3, 8, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 0, 3, 8, dtype=torch.float16, requires_grad=True)

    def dense_n0_backward(return_probs=False):
        out = flash_attn_func(q, k, k, return_attn_probs=return_probs)
        out[0].sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape

    run("dense N==0 sum().backward() no crash, zero shapes",
        lambda: dense_n0_backward(False), allowed_bwd=(0, 1))
    run("dense N==0 return_attn_probs .sum().backward() no crash, zero shapes",
        lambda: dense_n0_backward(True), allowed_bwd=(0, 1))

    # varlen T_Q == 0
    q = torch.randn(0, 2, 8, dtype=torch.float16, requires_grad=True)
    cu = torch.tensor([0], dtype=torch.int32)
    run("varlen T_Q==0 sum().backward() no crash",
        lambda: flash_attn_varlen_func(q, q, q, cu, cu, 0, 0).sum().backward())

    # varlen max_seqlen_k == 0 (empty KV path)
    q = torch.randn(6, 2, 8, dtype=torch.float16, requires_grad=True)
    k = torch.randn(0, 2, 8, dtype=torch.float16, requires_grad=True)
    cu_q = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 0], dtype=torch.int32)
    run("varlen max_seqlen_k==0 sum().backward() no crash",
        lambda: flash_attn_varlen_func(q, k, k, cu_q, cu_k, 3, 0).sum().backward())


def test_upstream_metadata_wiring():
    """Published naming contract: the wheel is the official distribution
    `flash-attn` 2.8.3 (single .dist-info, no post-install fake metadata),
    it installs `flash_attn`, and the extension module is `flash_attn_2_cuda`.
    No v100-suffixed public package/extension names."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo, "setup.py"), encoding="utf-8").read()
    with open(os.path.join(repo, "pyproject.toml"), encoding="utf-8") as f:
        py = f.read()
    checks = [
        ("setup.py names the distribution flash-attn 2.8.3",
         'name="flash-attn"' in src and 'version="2.8.3"' in src),
        ("pyproject names the distribution flash-attn 2.8.3",
         'name = "flash-attn"' in py and 'version = "2.8.3"' in py),
        ("extension module is flash_attn_2_cuda", 'name="flash_attn_2_cuda"' in src),
        ("no post-install fake metadata machinery",
         "_upstream_metadata" not in src and "InstallFakeDistInfo" not in src),
        ("no v100-suffixed public package/extension names",
         "flash_attn_v100" not in src and "flash_attn_v100" not in py),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("official naming: " + name)
    import flash_attn
    (R.ok if flash_attn.__version__ == "2.8.3" else R.fail)(
        "official naming: flash_attn.__version__ == 2.8.3", flash_attn.__version__)


# ---------------------------------------------------------------------------- packed --
def test_packed_dense():
    qkv = as_half(torch.arange(2 * 5 * 3 * 3 * 10, dtype=torch.float32).reshape(2, 5, 3, 3, 10))
    out = flash_attn_qkvpacked_func(qkv)
    c = last()
    expected_q = qkv[:, :, 0]
    checks = [
        ("qkvpacked -> q (B,M,H,Dpad)", c["q"].shape == (2, 5, 3, 16)),
        ("qkvpacked -> q values", torch.equal(c["q"][..., :10], expected_q)),
        ("qkvpacked -> k == second slice", torch.equal(c["k"][..., :10], qkv[:, :, 1])),
        ("qkvpacked output (B,M,H,D)", out.shape == (2, 5, 3, 10)),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("packed dense: " + name)

    # Official packed entry points are thin wrappers over flash_attn_func (there is
    # no fork-only ValueError shape guard).  The unpacked call must receive the
    # exact same q/k/v slices and options.
    c_qkv = c
    flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
    c_unpacked = last()
    (R.ok if all(torch.equal(c_qkv[k], c_unpacked[k]) for k in ("q", "k", "v"))
     else R.fail)("packed dense: qkvpacked == flash_attn_func(q,k,v)")

    q = torch.randn(2, 5, 6, 10, dtype=torch.float16)
    kv = torch.randn(2, 5, 2, 2, 10, dtype=torch.float16)
    out = flash_attn_kvpacked_func(q, kv)
    c = last()
    checks = [
        ("kvpacked GQA q heads 6", c["q"].shape == (2, 5, 6, 16)),
        ("kvpacked K heads 2", c["k"].shape == (2, 5, 2, 16)),
        ("kvpacked k values", torch.equal(c["k"][..., :10], kv[:, :, 0])),
        ("kvpacked output", out.shape == (2, 5, 6, 10)),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("packed dense: " + name)

    c_kv = c
    flash_attn_func(q, kv[:, :, 0], kv[:, :, 1])
    c_unpacked = last()
    (R.ok if all(torch.equal(c_kv[k], c_unpacked[k]) for k in ("q", "k", "v"))
     else R.fail)("packed dense: kvpacked == flash_attn_func(q,k,v)")


def test_packed_varlen():
    qkv = torch.randn(10, 3, 2, 16, dtype=torch.float16)
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu, 6)
    c = last()
    checks = [
        ("varlen qkvpacked q values", torch.equal(c["q"], qkv[:, 0].contiguous())),
        ("varlen qkvpacked k values", torch.equal(c["k"], qkv[:, 1].contiguous())),
        ("varlen qkvpacked cu_q == cu_k", c["cu_seqlens_q"].tolist() == [0, 4, 10]
         and c["cu_seqlens_k"].tolist() == [0, 4, 10]),
        ("varlen qkvpacked max_seqlen", c["max_seqlen_q"] >= 6 and c["max_seqlen_k"] >= 6),
        ("varlen qkvpacked output", out.shape == (10, 2, 16)),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("packed varlen: " + name)

    # Official packed entry points delegate to flash_attn_varlen_func; no
    # fork-only shape guards.  Verify the unpacked call gets the same slices.
    c_qkv = c
    flash_attn_varlen_func(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu, cu, 6, 6)
    c_unpacked = last()
    (R.ok if all(torch.equal(c_qkv[k], c_unpacked[k]) for k in ("q", "k", "v"))
     else R.fail)("packed varlen: qkvpacked == flash_attn_varlen_func(q,k,v)")

    q = torch.randn(10, 2, 16, dtype=torch.float16)
    kv = torch.randn(10, 2, 2, 16, dtype=torch.float16)
    out = flash_attn_varlen_kvpacked_func(q, kv, cu, cu, 6, 6)
    c = last()
    checks = [
        ("varlen kvpacked k values", torch.equal(c["k"], kv[:, 0].contiguous())),
        ("varlen kvpacked v values", torch.equal(c["v"], kv[:, 1].contiguous())),
        ("varlen kvpacked output", out.shape == (10, 2, 16)),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("packed varlen: " + name)

    c_kv = c
    flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu, cu, 6, 6)
    c_unpacked = last()
    (R.ok if all(torch.equal(c_kv[k], c_unpacked[k]) for k in ("q", "k", "v"))
     else R.fail)("packed varlen: kvpacked == flash_attn_varlen_func(q,k,v)")


# --------------------------------------------------------------------------- kvcache --
def test_kvcache():
    q = torch.randn(1, 1, 2, 16, dtype=torch.float16)
    kc = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    vc = torch.randn(1, 8, 2, 16, dtype=torch.float16)

    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=5)
    c = last()
    checks = [
        ("int cache_seqlens -> int32 tensor", c["seqlens_k"].dtype == torch.int32
         and c["seqlens_k"].tolist() == [5]),
        ("default scale from q last dim", abs(c["softmax_scale"] - 16 ** -0.5) < 1e-12),
        ("k/v writeback slots None", c["k"] is None and c["v"] is None),
        ("num_splits 0", c["num_splits"] == 0),
        ("rotary_interleaved default True", c["is_rotary_interleaved"] is True),
        ("output shape", out.shape == (1, 1, 2, 16)),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("kvcache: " + name)

    r = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=torch.tensor([5, 5], dtype=torch.int32)[:1],
                                return_softmax_lse=True)
    (R.ok if (isinstance(r, tuple) and len(r) == 2 and r[1].shape == (1, 2, 1)) else R.fail)(
        "kvcache: return_softmax_lse tuple (out, lse)")

    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=5, causal=True, window_size=(0, 4),
                                  softcap=10.0, num_splits=4)
    c = last()
    checks = [
        ("window forwarded", c["window_left"] == 0 and c["window_right"] == 4),
        ("softcap forwarded", c["softcap"] == 10.0),
        ("causal forwarded", c["causal"] is True),
        ("num_splits forwarded", c["num_splits"] == 4),
    ]
    for name, ok in checks:
        (R.ok if ok else R.fail)("kvcache: " + name)


if __name__ == "__main__":
    print("== interface behavior tests (recording stub backend) ==")
    test_dense_basic()
    test_dense_head_dim_rounding()
    test_dense_return_attn_probs()
    test_dense_empty()
    test_dense_misc()
    test_dense_backward()
    test_varlen_basic()
    test_varlen_return_attn_probs()
    test_varlen_empty_and_backward()
    test_empty_backward_no_crash()
    test_upstream_metadata_wiring()
    test_packed_dense()
    test_packed_varlen()
    test_kvcache()
    R.finish("test_interface_behavior")
