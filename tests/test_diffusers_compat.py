#!/usr/bin/env python
# ======================================================================================
# * diffusers compatibility tests.
# *
# * diffusers (>= 0.35) imports the private entry points straight out of
# * ``flash_attn.flash_attn_interface``:
# *
# *     from flash_attn.flash_attn_interface import (
# *         _wrapped_flash_attn_backward, _wrapped_flash_attn_forward)
# *
# * and drives them from its own autograd.Function instead of calling
# * ``flash_attn_func``.  If those names are missing the whole backend is dropped
# * ("flash_attn is installed but failed to import") and diffusers silently falls
# * back to SDPA.  This module pins that contract and, when a Volta GPU is
# * present, replays diffusers' exact call sequence against an fp32 reference.
# *
# * CPU-only machines still exercise the import path (the compiled extension is
# * replaced by the recording stub used by the other interface tests).
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)  # prefer the repo package over any installed copy

import torch  # noqa: E402

# The package imports flash_attn_2_cuda (via _v100_compat); use the CPU stub when
# the compiled extension is not built.  Must happen before importing flash_attn.
try:
    import _stub_cuda_backend as sb  # noqa: E402

    sb.install_stub()
    R = sb.RESULT
except Exception:  # noqa: BLE001 - torch may be unavailable
    class _R:  # minimal fallback so the module imports without the test harness
        def ok(self, name, detail=""):
            print(f"  [PASS] {name} {detail}")

        def fail(self, name, detail=""):
            print(f"  [FAIL] {name} {detail}")

        def skip(self, name, detail=""):
            print(f"  [SKIP] {name} {detail}")

        def finish(self, title):
            pass

    R = _R()

import flash_attn  # noqa: E402
import flash_attn.flash_attn_interface as fai  # noqa: E402


def check(name, cond, detail=""):
    (R.ok if cond else R.fail)(name, detail)


# The names diffusers imports, exactly as spelled there.
DIFFUSERS_IMPORTS = (
    "_wrapped_flash_attn_forward",
    "_wrapped_flash_attn_backward",
    "_wrapped_flash_attn_varlen_forward",
    "_wrapped_flash_attn_varlen_backward",
)

# diffusers calls the dense forward with 11 positional args and the dense
# backward with 18 (diffusers/models/attention_dispatch.py, _flash_attention_forward_op
# / _flash_attention_backward_op).
FWD_ARGS = 11
BWD_ARGS = 18


def test_import_contract():
    print("\n== diffusers import contract ==")
    for name in DIFFUSERS_IMPORTS:
        check(f"flash_attn.flash_attn_interface.{name} exists", hasattr(fai, name),
              "" if hasattr(fai, name) else "MISSING")

    for name, n in (("_wrapped_flash_attn_forward", FWD_ARGS),
                    ("_wrapped_flash_attn_backward", BWD_ARGS)):
        fn = getattr(fai, name, None)
        if fn is None:
            check(f"{name} accepts {n} positional args", False, "missing")
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            check(f"{name} callable", True, type(fn).__name__)
            continue
        params = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
        check(f"{name} accepts >= {n} positional args",
              varargs or len(params) >= n,
              "torch custom op (*args)" if varargs else f"{len(params)} positional")


def test_diffusers_backend_flag():
    print("\n== diffusers backend detection ==")
    try:
        from diffusers.models.attention_dispatch import _CAN_USE_FLASH_ATTN
    except Exception as e:  # noqa: BLE001
        R.skip("diffusers installed", str(e))
        return
    check("diffusers enables the FLASH backend", bool(_CAN_USE_FLASH_ATTN),
          f"flash_attn {flash_attn.__version__}")


def test_numeric_parity():
    print("\n== diffusers call sequence (GPU) ==")
    if not torch.cuda.is_available():
        R.skip("dense forward/backward parity", "no CUDA device")
        return
    if torch.cuda.get_device_capability()[0] != 7:
        R.skip("dense forward/backward parity", "not a Volta GPU")

    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_backward,
        _wrapped_flash_attn_forward,
    )

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 256, 64
    dtype = torch.float16
    dev = "cuda"
    q = torch.randn(B, S, H, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, H, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, H, D, device=dev, dtype=dtype, requires_grad=True)

    # --- forward, exactly as diffusers does it
    out, lse, _, rng_state = _wrapped_flash_attn_forward(
        q, k, v, 0.0, D ** -0.5, False, -1, -1, 0.0, None, False)
    out = out.contiguous()
    lse = lse.permute(0, 2, 1).contiguous()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.detach().float().transpose(1, 2),
        k.detach().float().transpose(1, 2),
        v.detach().float().transpose(1, 2),
    ).transpose(1, 2)
    fmax = (out.float() - ref).abs().max().item()
    check("forward matches SDPA", fmax < 5e-3, f"max |d| = {fmax:.2e}")

    # --- backward, exactly as diffusers does it (grad_out, q, k, v, out, lse,
    # dq, dk, dv, dropout_p, scale, causal, wl, wr, softcap, alibi, deterministic, rng)
    grad_out = torch.randn_like(out)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    # diffusers saves lse in (B, S, H) and permutes it back to (B, H, S) here.
    lse_bhs = lse.permute(0, 2, 1).contiguous()
    _wrapped_flash_attn_backward(
        grad_out, q, k, v, out, lse_bhs, dq, dk, dv,
        0.0, D ** -0.5, False, -1, -1, 0.0, None, False, rng_state)

    q2 = q.detach().float().requires_grad_(True)
    k2 = k.detach().float().requires_grad_(True)
    v2 = v.detach().float().requires_grad_(True)
    o2 = torch.nn.functional.scaled_dot_product_attention(
        q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2)).transpose(1, 2)
    o2.backward(grad_out.float())
    for name, got, want in (("dq", dq, q2.grad), ("dk", dk, k2.grad), ("dv", dv, v2.grad)):
        dmax = (got.float() - want).abs().max().item()
        check(f"backward {name} matches SDPA", dmax < 1e-2, f"max |d| = {dmax:.2e}")


def main():
    print(f"flash_attn {flash_attn.__version__} from {flash_attn.__file__}")
    test_import_contract()
    test_diffusers_backend_flag()
    test_numeric_parity()
    R.finish("test_diffusers_compat")


if __name__ == "__main__":
    main()
