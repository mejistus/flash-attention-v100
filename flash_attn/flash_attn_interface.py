# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import torch
import warnings
import traceback
import flash_attn_2_cuda as flash_attn_gpu
from typing import Optional, Tuple, Union

def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def round_multiple(x: int, m: int) -> int:
    """Round x up to the next multiple of m (same helper as official flash-attn)."""
    return (x + m - 1) // m * m


def _pad_dmask_upstream(dmask: torch.Tensor, B: int, H: int, M: int, N: int) -> torch.Tensor:
    """Pad a dense ``S_dmask`` (B, H, M, N) to the shape official FA2 returns.

    Upstream's `_flash_attn_forward_fake` allocates
    ``(batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128))``
    and the CUDA kernel fills it padded; this fork's kernel returns the tight
    (M, N) block (research/07 F-6). Copy it into a zero-padded buffer so consumers
    indexing the upstream shape keep working. Padding is never attended, hence zero.
    """
    M_pad, N_pad = round_multiple(M, 128), round_multiple(N, 128)
    if dmask.shape[-2:] == (M_pad, N_pad):
        return dmask
    padded = dmask.new_zeros((B, H, M_pad, N_pad))
    padded[..., :M, :N].copy_(dmask)
    return padded

# ======================================================================================
# HEAD DIM HANDLING
# ======================================================================================
# The compiled kernels are instantiated for these head dimensions only. Any other head
# size (multiple of 8, <= 256) is zero-padded up to the next supported size. Padded
# entries contribute exactly zero to QK^T and P@V, so the first `head_size_og` output
# dimensions are numerically identical, and the padding is sliced away on return.
# The softmax scale is always derived from the *original* head size, matching official
# FlashAttention-2 semantics for non-power-of-two head dims.
_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128, 256)


def _round_head_dim(head_size_og: int) -> int:
    for d in _SUPPORTED_HEAD_DIMS:
        if head_size_og <= d:
            return d
    # Upstream rejects head_size > 256 from the C++ dispatcher with a RuntimeError;
    # keep the exception type compatible for drop-in callers.
    raise RuntimeError(
        f"Unsupported D: {head_size_og} (supported head dimensions: {_SUPPORTED_HEAD_DIMS})"
    )

# ======================================================================================
# DENSE ATTENTION (B, M, H, D) -> (B, H, M, D)
# ======================================================================================
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_softmax: bool,
        is_grad_enabled: bool
    ) -> torch.Tensor:
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        # Official FA2 dense layout is BMHD [B, M, H, D]. The kernel consumes it
        # directly, so q/k/v/out are never permuted or copied here (that used to cost
        # ~3-4 extra (B,M,H,D) fp16 tensors per layer and dominated training memory).
        q_ = q if q.is_contiguous() else q.contiguous()
        k_ = k if k.is_contiguous() else k.contiguous()
        v_ = v if v.is_contiguous() else v.contiguous()

        B, M, H_Q, head_size_og = q_.shape
        H_K = k_.shape[2]
        N = k_.shape[1]

        if B == 0 or M == 0:
            # Match official FA2: empty inputs return empty (well-shaped) tensors.
            out = torch.empty((B, M, H_Q, head_size_og), dtype=q.dtype, device=q.device)
            if return_softmax:
                lse = torch.empty((B, H_Q, M), dtype=torch.float32, device=q.device)
                dmask = torch.empty(0, dtype=q.dtype, device=q.device)
                return out, lse, dmask
            return out

        if N == 0:
            # Official FA2 (flash_api.cpp): seqlen_k == 0 -> zeroed output, lse = +inf.
            out = torch.zeros((B, M, H_Q, head_size_og), dtype=q.dtype, device=q.device)
            if return_softmax:
                lse = torch.full((B, H_Q, M), float("inf"), dtype=torch.float32, device=q.device)
                dmask = torch.empty(0, dtype=q.dtype, device=q.device)
                return out, lse, dmask
            return out

        target_head_dim = _round_head_dim(head_size_og)
        pad_size = target_head_dim - head_size_og
        if pad_size > 0:
            q_ = torch.nn.functional.pad(q_, [0, pad_size])
            k_ = torch.nn.functional.pad(k_, [0, pad_size])
            v_ = torch.nn.functional.pad(v_, [0, pad_size])

        q_ = q_.contiguous()
        k_ = k_.contiguous()
        v_ = v_.contiguous()

        if softmax_scale is None:
            softmax_scale = head_size_og ** -0.5

        window_left, window_right = window_size

        out_, lse_, dmask_, rng_state = flash_attn_gpu.fwd(
            q_, k_, v_, None, alibi_slopes,
            dropout_p, softmax_scale, causal,
            window_left, window_right, softcap,
            # Official FA2 only returns the dropout mask when dropout is active;
            # with dropout_p == 0 the returned S_dmask is an empty tensor.
            return_softmax and dropout_p > 0.0, None
        )

        out = out_[..., :head_size_og]
        if pad_size > 0:
            # Padded head dim: keep the official contiguous-output contract. For the
            # supported dims (no padding) this is a no-op view of the kernel output.
            out = out.contiguous()

        if is_grad:
            ctx.save_for_backward(q_, k_, v_, out_, lse_, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
            ctx.pad_size = pad_size

        if return_softmax:
            if dmask_.numel() > 0:
                # F-6 (research/07): official FA2 returns S_dmask padded to
                # (B, H, round128(M), round128(N)); the kernel allocates the tight
                # (B, H, M, N) block, so pad it here (zeros in the border).
                dmask_ = _pad_dmask_upstream(dmask_, B, H_Q, M, N)
            return out, lse_, dmask_
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.saved_tensors:
            # F-2 (research/07): the empty-input early returns above return before
            # save_for_backward, so autograd arrives here with an empty saved_tensors.
            # The output never depended on q/k/v; return one None per forward input
            # (12) instead of unpacking an empty tuple by mistake.
            return (None,) * 12
        q_, k_, v_, out_, lse_, rng_state = ctx.saved_tensors
        head_size_og = ctx.head_size_og
        pad_size = ctx.pad_size

        if pad_size > 0:
            dout = torch.nn.functional.pad(dout, [0, pad_size])
        dout_ = dout if dout.is_contiguous() else dout.contiguous()

        dq_ = torch.empty_like(q_)
        dk_ = torch.empty_like(k_)
        dv_ = torch.empty_like(v_)

        window_left, window_right = ctx.window_size

        grads = flash_attn_gpu.bwd(
            dout_, q_, k_, v_, out_, lse_,
            dq_, dk_, dv_,
            ctx.alibi_slopes,
            ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            window_left, window_right,
            ctx.softcap, ctx.deterministic, None, rng_state
        )

        dq = grads[0][..., :head_size_og]
        dk = grads[1][..., :head_size_og]
        dv = grads[2][..., :head_size_og]
        if pad_size > 0:
            dq = dq.contiguous()
            dk = dk.contiguous()
            dv = dv.contiguous()

        # Exactly one gradient slot per forward input (12 inputs): 3 tensors + 9 Nones.
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
):
    """Dense Flash Attention (B, M, H, D)"""
    if deterministic:
        warnings.warn("Forward is always deterministic. Deterministic backward is not supported.", RuntimeWarning)
        deterministic = False

    try:
        return FlashAttnFunc.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            torch.is_grad_enabled()
        )
    except Exception as e:
        print(f"[VOLTA FA2 DENSE FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


# ======================================================================================
# VARLEN ATTENTION (T, H, D) -> (T, H, D)
# ======================================================================================
class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_softmax: bool,
        block_table: Optional[torch.Tensor],
        is_grad_enabled: bool
    ) -> torch.Tensor:

        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        q = maybe_contiguous(q)
        k = maybe_contiguous(k)
        v = maybe_contiguous(v)

        head_size_og = q.size(2)

        if q.size(0) == 0:
            # Empty packed sequence (T_Q == 0): well-shaped empty tensors, official parity.
            out = torch.empty_like(q)
            if return_softmax:
                lse = torch.empty((q.size(1), 0), dtype=torch.float32, device=q.device)
                dmask = torch.empty(0, dtype=q.dtype, device=q.device)
                return out, lse, dmask
            return out

        if max_seqlen_k == 0:
            # Official FA2 (flash_api.cpp): empty KV (max_seqlen_k == 0) -> zeroed output,
            # lse = +inf.
            out = torch.zeros_like(q)
            if return_softmax:
                lse = torch.full((q.size(1), q.size(0)), float("inf"), dtype=torch.float32,
                                 device=q.device)
                dmask = torch.empty(0, dtype=q.dtype, device=q.device)
                return out, lse, dmask
            return out

        target_head_dim = _round_head_dim(head_size_og)
        pad_size = target_head_dim - head_size_og
        if pad_size > 0:
            q = torch.nn.functional.pad(q, [0, pad_size])
            k = torch.nn.functional.pad(k, [0, pad_size])
            v = torch.nn.functional.pad(v, [0, pad_size])

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if softmax_scale is None:
            softmax_scale = head_size_og ** -0.5

        window_left, window_right = window_size

        seqused_k = None
        leftpad_k = None
        num_splits = 0

        # G-3 python guard (research/08): the compiled varlen backward ignores
        # seqused_k entirely -- it would differentiate against the full cu_seqlens_k
        # range while the forward attended only the first seqused_k[b] keys, i.e.
        # silently wrong dq/dk/dv. This wrapper (upstream-2.8.3 signature) does not
        # expose seqused_k, so it is always None here; keep the check so any future
        # wiring fails loudly instead of silently. Raw-extension inference that does
        # pass seqused_k is unaffected (guard requires grad + requires_grad inputs).
        if (seqused_k is not None and torch.is_grad_enabled()
                and any(x.requires_grad for x in (q, k, v))):
            raise NotImplementedError(
                "flash_attn_varlen_func: seqused_k is not supported for differentiable "
                "attention. The backward kernel ignores seqused_k and would return "
                "silently wrong gradients for q/k/v. Run under torch.no_grad()/eval, "
                "or pass the already-truncated K/V tensors instead."
            )

        out, lse, dmask, rng_state = flash_attn_gpu.varlen_fwd(
            q, k, v, None, cu_seqlens_q, cu_seqlens_k,
            seqused_k, leftpad_k, block_table, alibi_slopes,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
            False, causal, window_left, window_right, softcap,
            return_softmax and dropout_p > 0.0, None, num_splits
        )

        out = out[..., :head_size_og].contiguous()

        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
            ctx.pad_size = pad_size
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k

        return (out, lse, dmask) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.saved_tensors:
            # F-2 (research/07): empty-input early returns (T_Q == 0 / max_seqlen_k == 0)
            # return before save_for_backward -> autograd calls us with an empty tuple.
            # The output never depended on q/k/v; return one None per forward input (17).
            return (None,) * 17
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        head_size_og = ctx.head_size_og
        pad_size = ctx.pad_size

        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        dout = maybe_contiguous(dout)
        if pad_size > 0:
            dout = torch.nn.functional.pad(dout, [0, pad_size])
        dout = dout.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        window_left, window_right = ctx.window_size

        grads = flash_attn_gpu.varlen_bwd(
            dout, q, k, v, out, lse,
            dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.dropout_p, ctx.softmax_scale, False,
            ctx.causal, window_left, window_right, ctx.softcap,
            ctx.deterministic, None, rng_state
        )

        dq = grads[0][..., :head_size_og].contiguous()
        dk = grads[1][..., :head_size_og].contiguous()
        dv = grads[2][..., :head_size_og].contiguous()

        # Exactly one gradient slot per forward input (17 inputs): 3 tensors + 14 Nones.
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
):
    """Varlen Flash Attention (T, H, D)

    ``return_attn_probs=True`` + ``dropout_p > 0`` returns ``S_dmask`` with the fork
    layout ``(T_Q, nheads, max_seqlen_k)``. This is a documented deviation from
    upstream FA2, which returns a 4-D padded tensor
    ``(batch, nheads, round128(max_seqlen_q), round128(max_seqlen_k))``. Values for
    the real tokens are the same; the padding/varlen packing differs (research/07 F-6).
    """
    if deterministic:
        warnings.warn("Forward is always deterministic. Deterministic backward is not supported.", RuntimeWarning)
        deterministic = False

    try:
        return FlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            block_table,
            torch.is_grad_enabled()
        )
    except Exception as e:
        print(f"[VOLTA FA2 VARLEN FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

# ======================================================================================
# KV ATTENTION
# ======================================================================================
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    FlashAttention with KV cache (B, M, H, D).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)

    out = torch.empty_like(q)

    out, softmax_lse = flash_attn_gpu.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
    )

    if return_softmax_lse:
        return out, softmax_lse
    return out

# ======================================================================================
# PACKED QKV / KV ATTENTION (official FA2 drop-in)
# ======================================================================================
# Signatures replicate official flash-attn 2.8.x exactly. They are thin wrappers over
# the autograd functions above: gradients flow through the unbind() views back into
# the packed tensor, giving the official packed-gradient semantics (grad wrt qkv has
# shape (batch_size, seqlen, 3, nheads, headdim)).

def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """Dense Flash Attention with QKV packed: qkv (batch_size, seqlen, 3, nheads, headdim)."""
    if qkv.dim() != 5 or qkv.size(2) != 3:
        raise ValueError(
            f"qkv must have shape (batch_size, seqlen, 3, nheads, headdim), got {tuple(qkv.shape)}"
        )
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
        alibi_slopes, deterministic, return_attn_probs,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """Dense Flash Attention with KV packed: q (B, S, HQ, D), kv (B, S, 2, HKV, D)."""
    if kv.dim() != 5 or kv.size(2) != 2:
        raise ValueError(
            f"kv must have shape (batch_size, seqlen, 2, nheads_k, headdim), got {tuple(kv.shape)}"
        )
    k, v = kv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
        alibi_slopes, deterministic, return_attn_probs,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """Varlen Flash Attention with QKV packed: qkv (total_seqlen, 3, nheads, headdim)."""
    if qkv.dim() != 4 or qkv.size(1) != 3:
        raise ValueError(
            f"qkv must have shape (total_seqlen, 3, nheads, headdim), got {tuple(qkv.shape)}"
        )
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p, softmax_scale, causal, window_size, softcap,
        alibi_slopes, deterministic, return_attn_probs,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """Varlen Flash Attention with KV packed: q (total_q, HQ, D), kv (total_k, 2, HKV, D)."""
    if kv.dim() != 4 or kv.size(1) != 2:
        raise ValueError(
            f"kv must have shape (total_seqlen_k, 2, nheads_k, headdim), got {tuple(kv.shape)}"
        )
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size, softcap,
        alibi_slopes, deterministic, return_attn_probs,
    )


flash_attn_varlen_gpu = flash_attn_varlen_func
flash_attn_with_kvcache_gpu = flash_attn_with_kvcache

__all__ = [
    "flash_attn_func", "flash_attn_gpu",
    "flash_attn_qkvpacked_func", "flash_attn_kvpacked_func",
    "flash_attn_varlen_func", "flash_attn_varlen_gpu",
    "flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func",
    "flash_attn_with_kvcache", "flash_attn_with_kvcache_gpu",
]
