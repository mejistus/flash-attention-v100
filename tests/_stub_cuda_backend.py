# ======================================================================================
# * CPU stub for the flash_attn_2_cuda extension.
# *
# * Purpose: allow testing the Python interface layer (argument handling, padding,
# * permutes, return-value semantics) on a machine without a Volta GPU / without a
# * compiled extension. When the real extension is importable it is used as-is.
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import sys
import types
import torch


class _StubBackend:
    """Records every call and returns tensors with the exact expected layouts."""

    def __init__(self):
        self.calls = []

    # ------------------------------------------------------------------ dense --
    def fwd(self, q, k, v, out, alibi_slopes, p_dropout, softmax_scale, is_causal,
            window_left, window_right, softcap, return_softmax, gen):
        self.calls.append(dict(
            kind="fwd", q=q, k=k, v=v, out=out, alibi_slopes=alibi_slopes,
            dropout_p=p_dropout, softmax_scale=softmax_scale, causal=is_causal,
            window_left=window_left, window_right=window_right, softcap=softcap,
            return_softmax=return_softmax, gen=gen,
        ))
        B, M, H, _ = q.shape
        out_t = torch.empty_like(q)
        lse = torch.empty((B, H, M), dtype=torch.float32)
        dmask = torch.empty((B, H, M, k.shape[1])) if return_softmax else torch.empty(0)
        return out_t, lse, dmask, torch.empty(0, dtype=torch.uint8)

    def bwd(self, dout, q, k, v, out, lse, dq, dk, dv, alibi_slopes, p_dropout,
            softmax_scale, is_causal, window_left, window_right, softcap,
            deterministic, gen, rng_state):
        self.calls.append(dict(kind="bwd", dropout_p=p_dropout, causal=is_causal))
        return torch.empty_like(dq), torch.empty_like(dk), torch.empty_like(dv), torch.empty(0)

    # ----------------------------------------------------------------- varlen --
    def varlen_fwd(self, q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k,
                   leftpad_k, block_table, alibi_slopes, max_seqlen_q, max_seqlen_k,
                   p_dropout, softmax_scale, zero_tensors, is_causal, window_left,
                   window_right, softcap, return_softmax, gen, num_splits):
        self.calls.append(dict(
            kind="varlen_fwd", q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k, seqused_k=seqused_k, leftpad_k=leftpad_k,
            block_table=block_table, alibi_slopes=alibi_slopes,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, dropout_p=p_dropout,
            softmax_scale=softmax_scale, zero_tensors=zero_tensors, causal=is_causal,
            window_left=window_left, window_right=window_right, softcap=softcap,
            return_softmax=return_softmax, gen=gen, num_splits=num_splits,
        ))
        T, H, _ = q.shape
        out_t = torch.empty_like(q)
        lse = torch.empty((H, T), dtype=torch.float32)
        dmask = torch.empty((T, H, max_seqlen_k)) if return_softmax else torch.empty(0)
        return out_t, lse, dmask, torch.empty(0, dtype=torch.uint8)

    def varlen_bwd(self, dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens_q,
                   cu_seqlens_k, alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout,
                   softmax_scale, zero_tensors, is_causal, window_left, window_right,
                   softcap, deterministic, gen, rng_state):
        self.calls.append(dict(kind="varlen_bwd"))
        return torch.empty_like(dq), torch.empty_like(dk), torch.empty_like(dv), torch.empty(0)

    # ---------------------------------------------------------------- kvcache --
    def fwd_kvcache(self, q, kcache, vcache, k, v, seqlens_k, rotary_cos, rotary_sin,
                    cache_batch_idx, leftpad_k, block_table, alibi_slopes, out,
                    softmax_scale, is_causal, window_left, window_right, softcap,
                    is_rotary_interleaved, num_splits):
        self.calls.append(dict(
            kind="fwd_kvcache", q=q, kcache=kcache, vcache=vcache, k=k, v=v,
            seqlens_k=seqlens_k, rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            cache_batch_idx=cache_batch_idx, leftpad_k=leftpad_k,
            block_table=block_table, alibi_slopes=alibi_slopes, out=out,
            softmax_scale=softmax_scale, causal=is_causal, window_left=window_left,
            window_right=window_right, softcap=softcap,
            is_rotary_interleaved=is_rotary_interleaved, num_splits=num_splits,
        ))
        # q is (B, T, H, D) for the kvcache calling convention; lse must be (B, H, T).
        B, T, H, _ = q.shape
        return torch.empty_like(q), torch.empty((B, H, T), dtype=torch.float32)


def install_stub(force=False):
    """Ensure `import flash_attn_2_cuda` works. Returns (module, is_stub).

    force=True installs the recording stub even if the real extension is importable
    (used by behavior tests that inspect the exact arguments passed to the backend).
    """
    if not force:
        try:
            import flash_attn_2_cuda  # noqa: F401  (real compiled extension)
            return sys.modules["flash_attn_2_cuda"], False
        except Exception:
            pass
    stub = _StubBackend()
    mod = types.ModuleType("flash_attn_2_cuda")
    mod.fwd = stub.fwd
    mod.bwd = stub.bwd
    mod.varlen_fwd = stub.varlen_fwd
    mod.varlen_bwd = stub.varlen_bwd
    mod.fwd_kvcache = stub.fwd_kvcache
    mod.stub_backend = stub
    sys.modules["flash_attn_2_cuda"] = mod
    return mod, True


class _Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.messages = []

    def ok(self, name, detail=""):
        self.passed += 1
        print(f"  [PASS] {name} {('- ' + detail) if detail else ''}")

    def fail(self, name, detail=""):
        self.failed += 1
        self.messages.append(f"FAIL {name}: {detail}")
        print(f"  [FAIL] {name} {('- ' + detail) if detail else ''}")

    def skip(self, name, detail=""):
        self.skipped += 1
        print(f"  [SKIP] {name} {('- ' + detail) if detail else ''}")

    def finish(self, title):
        total = self.passed + self.failed
        print(f"\n== {title}: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped ==")
        if self.failed:
            for m in self.messages:
                print("   " + m)
            sys.exit(1)


RESULT = _Result()
