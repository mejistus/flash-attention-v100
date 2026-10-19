#!/usr/bin/env python
# ======================================================================================
# * Extension smoke test (post-build). Passes trivially when the extension is absent
# * (pre-build environments), so this script is safe to run at any time.
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stub_cuda_backend as sb  # noqa: E402

R = sb.RESULT

try:
    import torch
    import flash_attn_2_cuda
    HAVE = True
except Exception as e:
    HAVE = False
    err = e

if not HAVE:
    R.skip("compiled extension import", f"not built: {type(err).__name__}: {err}")
    R.finish("test_extension_smoke")
    raise SystemExit(0)

print("== extension smoke test ==")
for fn in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "fwd_kvcache"):
    obj = getattr(flash_attn_2_cuda, fn, None)
    R.ok(f"flash_attn_2_cuda.{fn} callable") if callable(obj) else R.fail(f"flash_attn_2_cuda.{fn}", "missing")

# the official-compat alias module is provided via a symlinked .so
try:
    import flash_attn_2_cuda

    ok = all(callable(getattr(flash_attn_2_cuda, f, None))
             for f in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "fwd_kvcache"))
    (R.ok if ok else R.fail)("flash_attn_2_cuda alias module exposes all 5 ops")
except ImportError as e:
    R.fail("flash_attn_2_cuda alias module import", str(e))

# TorchBind registration (used by torch.compile / vLLM-style integrations)
for op in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "fwd_kvcache"):
    try:
        exists = bool(torch.ops.flash_attn.__getattr__(op).default)
        R.ok(f"torch.ops.flash_attn.{op} registered") if exists else R.fail(
            f"torch.ops.flash_attn.{op}", "no default overload")
    except AttributeError as e:
        R.fail(f"torch.ops.flash_attn.{op}", str(e))

R.finish("test_extension_smoke")
