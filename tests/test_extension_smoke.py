#!/usr/bin/env python
# ======================================================================================
# * Extension smoke test (post-build).
# *
# * When the compiled extension is importable it checks the pybind surface
# * (fwd/bwd/varlen_fwd/varlen_bwd/fwd_kvcache).  Independently of that, it imports
# * flash_attn (using the CPU stub if the extension is absent) and checks the official
# * torch.library custom ops registered by flash_attn 2.8.3:
# *     torch.ops.flash_attn._flash_attn_forward
# *     torch.ops.flash_attn._flash_attn_varlen_forward
# *     torch.ops.flash_attn._flash_attn_backward
# *     torch.ops.flash_attn._flash_attn_varlen_backward
# * so the script also pins the interface registration on pre-build machines.
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # prefer the repo package
sys.path.insert(0, _HERE)
import _stub_cuda_backend as sb  # noqa: E402

R = sb.RESULT

try:
    import torch
except Exception as e:  # noqa: BLE001
    R.skip("torch import", f"{type(e).__name__}: {e}")
    R.finish("test_extension_smoke")
    raise SystemExit(0)

# ---------------------------------------------------------------- compiled extension --
try:
    import flash_attn_2_cuda
    HAVE_EXT = True
except Exception as e:  # noqa: BLE001
    HAVE_EXT = False
    ext_err = e

print("== extension smoke test ==")
PYBIND_OPS = ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "fwd_kvcache")
if HAVE_EXT:
    for fn in PYBIND_OPS:
        obj = getattr(flash_attn_2_cuda, fn, None)
        if callable(obj):
            R.ok(f"flash_attn_2_cuda.{fn}")
        else:
            R.fail(f"flash_attn_2_cuda.{fn}", "missing")
else:
    R.skip("compiled extension import",
           f"not built: {type(ext_err).__name__}: {ext_err}")
    # Make `import flash_attn` (and therefore custom-op registration) possible.
    sb.install_stub()

# ------------------------------------------------------------- official custom ops --
try:
    import flash_attn  # noqa: F401  (registers the torch.ops.flash_attn namespace)
    import flash_attn.flash_attn_interface as fai
except Exception as e:  # noqa: BLE001
    R.fail("import flash_attn", f"{type(e).__name__}: {e}")
else:
    if HAVE_EXT:
        # The interface must expose the extension module (not a function alias).
        gpu = getattr(fai, "flash_attn_gpu", None)
        ok = gpu is not None and all(callable(getattr(gpu, f, None))
                                     for f in PYBIND_OPS)
        (R.ok if ok else R.fail)("flash_attn.flash_attn_interface.flash_attn_gpu "
                                 "exposes the extension surface")

    # Upstream 2.8.3 registers exactly these four torch.library custom ops.  Check
    # that each has a default overload and the official parameter name surface.
    EXPECTED_PARAMS = {
        "_flash_attn_forward": (
            "q", "k", "v", "dropout_p", "softmax_scale", "causal",
            "window_size_left", "window_size_right", "softcap", "alibi_slopes",
            "return_softmax",
        ),
        "_flash_attn_varlen_forward": (
            "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q",
            "max_seqlen_k", "dropout_p", "softmax_scale", "causal",
            "window_size_left", "window_size_right", "softcap", "alibi_slopes",
            "return_softmax", "block_table", "leftpad_k", "seqused_k",
            "zero_tensors",
        ),
        "_flash_attn_backward": (
            "dout", "q", "k", "v", "out", "softmax_lse", "dq", "dk", "dv",
            "dropout_p", "softmax_scale", "causal", "window_size_left",
            "window_size_right", "softcap", "alibi_slopes", "deterministic",
            "rng_state",
        ),
        "_flash_attn_varlen_backward": (
            "dout", "q", "k", "v", "out", "softmax_lse", "dq", "dk", "dv",
            "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k",
            "dropout_p", "softmax_scale", "causal", "window_size_left",
            "window_size_right", "softcap", "alibi_slopes", "deterministic",
            "rng_state", "zero_tensors",
        ),
    }
    for op_name, expected in EXPECTED_PARAMS.items():
        try:
            op = getattr(torch.ops.flash_attn, op_name)
            schema = op.default._schema
            names = tuple(a.name for a in schema.arguments)
            ok = names == expected
            detail = "" if ok else f"params {names}"
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        (R.ok if ok else R.fail)(f"torch.ops.flash_attn.{op_name} registered",
                                 detail)

R.finish("test_extension_smoke")
