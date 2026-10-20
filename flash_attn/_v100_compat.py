"""Internal V100 compatibility layer for the upstream flash-attn 2.8.3 API.

The public Python files in this package are kept byte-for-byte identical to
upstream flash-attn 2.8.3.  This private module adapts only the compiled V100
extension, which differs from the upstream extension in two implementation
details:

* the V100 kernels are specialised for head dimensions 16/32/64/128/256;
* their dropout mask / softmax_d buffers use the tight sequence lengths instead
  of upstream's 128-padded shapes.

Unsupported head dimensions are zero-padded to the next specialised dimension
and sliced back on return.  Zero-filled dimensions contribute exactly zero to
QK^T and P@V, and the softmax scale is still derived from the original head
dimension, so results are numerically identical to upstream semantics.

Empty inputs follow upstream semantics: B == 0 is rejected by the V100 extension
exactly like upstream's ``TORCH_CHECK(batch_size > 0)``; M == 0 (empty query
sequence) is rejected with a clear RuntimeError because the V100 kernel cannot
launch its backward; an empty KV sequence (N == 0) returns zeros with
softmax_lse == +inf, matching upstream's ``mha_fwd`` empty-KV branch (the V100
kernel cannot launch that case, and its backward is likewise short-circuited to
zeros).
The same conventions are used by the varlen entry points.  KV-cache appends with
padded/permuted caches are copied back explicitly, including ``cache_batch_idx``
and ``cache_leftpad`` addressing.

Nothing here changes the public API: users import the same names from
``flash_attn`` as they do from the official package.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F

_EXT_NAME = "flash_attn_2_cuda"
_real = importlib.import_module(_EXT_NAME)

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128, 256)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _target_dim(head_dim: int) -> int:
    for d in _SUPPORTED_HEAD_DIMS:
        if head_dim <= d:
            return d
    raise RuntimeError(
        f"Unsupported D: {head_dim} (supported head dimensions: "
        f"{_SUPPORTED_HEAD_DIMS})"
    )


def _pad_dim(x: Optional[torch.Tensor], target: int) -> Optional[torch.Tensor]:
    """Return a contiguous tensor whose last dimension is ``target``."""
    if x is None:
        return None
    head_dim = x.shape[-1]
    if head_dim == target:
        return x.contiguous()
    return F.pad(x, (0, target - head_dim)).contiguous()


def _slice_dim(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    return x[..., :head_dim].contiguous()


def _rng_state(q: torch.Tensor) -> torch.Tensor:
    """Placeholder RNG state for outputs that never reach the kernels."""
    return torch.empty((2,), dtype=torch.int64, device=q.device)


def _dense_dmask(
    q: torch.Tensor, return_softmax: bool, B: int, H: int, M: int, N: int
) -> torch.Tensor:
    if not return_softmax:
        return torch.empty(0, dtype=q.dtype, device=q.device)
    return torch.empty(
        (B, H, _round_up(M, 128), _round_up(N, 128)),
        dtype=q.dtype,
        device=q.device,
    )


def _varlen_dmask(
    q: torch.Tensor,
    return_softmax: bool,
    B: int,
    H: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    if not return_softmax:
        return torch.empty(0, dtype=q.dtype, device=q.device)
    return torch.empty(
        (B, H, _round_up(max_seqlen_q, 128), _round_up(max_seqlen_k, 128)),
        dtype=q.dtype,
        device=q.device,
    )


def _empty_grad(g: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if g is None:
        return torch.zeros_like(ref)
    g.zero_()
    return g


def _cache_batch_rows(
    cache_batch_idx: Optional[torch.Tensor], num_cache_rows: int
) -> Optional[torch.Tensor]:
    if cache_batch_idx is None:
        return None
    cbi = cache_batch_idx.detach().to("cpu", dtype=torch.long).reshape(-1)
    if cbi.numel() and (int(cbi.min()) < 0 or int(cbi.max()) >= num_cache_rows):
        raise IndexError(
            f"cache_batch_idx out of range for a cache with {num_cache_rows} rows"
        )
    return cbi


def _copy_kv_cache_back(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor],
    dest_rows: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    m: int,
    head_dim: int,
) -> None:
    """Copy rows appended to a padded/permuted cache copy back to the caller."""
    if m <= 0:
        return
    if cache_seqlens is None:
        raise RuntimeError("If key is supplied, cache_seqlens must also be passed in")
    sl = cache_seqlens.detach().to("cpu", dtype=torch.long).reshape(-1)
    B = sl.numel()
    if dest_rows is not None:
        dest = dest_rows.detach().to("cpu", dtype=torch.long).reshape(-1)
        if dest.numel() != B:
            raise ValueError("cache_batch_idx must have one entry per batch")
    else:
        dest = None
    lp = (
        cache_leftpad.detach().to("cpu", dtype=torch.long).reshape(-1)
        if cache_leftpad is not None
        else None
    )
    if lp is not None and lp.numel() != B:
        raise ValueError("cache_leftpad must have one entry per batch")
    cache_len = k_cache.shape[1]
    for b in range(B):
        # ``kc`` row b always corresponds to batch b (it is either the original
        # cache row or the permuted copy's row b).
        dst_row = int(dest[b]) if dest is not None else b
        start = int(sl[b]) + (int(lp[b]) if lp is not None else 0)
        end = start + m
        if start < 0 or end > cache_len:
            raise IndexError(
                f"KV-cache append out of range: rows {start}:{end} for cache "
                f"of length {cache_len} (batch {b})"
            )
        k_cache[dst_row, start:end].copy_(kc[b, start:end, :, :head_dim])
        v_cache[dst_row, start:end].copy_(vc[b, start:end, :, :head_dim])


def _prep_rotary(
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    head_dim: int,
    cache_capacity: int,
    seqlen_new: int,
    seqlen_q: int,
    cache_seqlens: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
) -> tuple:
    """Validate/pad rotary tables for the V100 kvcache kernel.

    The V100 kernel requires ``cos.shape[0] >= cache_capacity + seqlen_new``
    (upstream only requires ``>= cache_capacity``).  Rows past the positions
    actually used are filled with a neutral rotation (cos=1, sin=0).  The
    rotary width itself is left untouched: extending a NeoX-style table would
    change the (i, i + rotary_dim / 2) pairing and corrupt the rotation.
    """
    if rotary_cos is None or rotary_sin is None:
        return rotary_cos, rotary_sin
    if rotary_cos.shape[-1] * 2 > head_dim:
        raise RuntimeError(
            f"rotary_cos covers {rotary_cos.shape[-1] * 2} dims but head_dim "
            f"is {head_dim}"
        )
    needed_rows = cache_capacity + seqlen_new
    if cache_seqlens is not None:
        sl = cache_seqlens.detach().to("cpu", dtype=torch.long).reshape(-1)
        lp = (
            cache_leftpad.detach().to("cpu", dtype=torch.long).reshape(-1)
            if cache_leftpad is not None
            else None
        )
        base = 0
        if sl.numel():
            base = max(
                int(sl[b]) + (int(lp[b]) if lp is not None else 0)
                for b in range(sl.numel())
            )
        actual_rows = base + max(seqlen_q, seqlen_new) if sl.numel() else 0
    else:
        actual_rows = needed_rows
    if rotary_cos.shape[0] < actual_rows:
        raise RuntimeError(
            f"V100 compatibility: rotary_cos/rotary_sin have "
            f"{rotary_cos.shape[0]} positions but {actual_rows} are required "
            "by cache_seqlens/cache_leftpad"
        )
    if rotary_sin.shape != rotary_cos.shape:
        raise RuntimeError("rotary_cos and rotary_sin must have the same shape")
    if rotary_cos.shape[0] < needed_rows:
        pad = needed_rows - rotary_cos.shape[0]
        rotary_cos = F.pad(rotary_cos, (0, 0, 0, pad), value=1.0)
        rotary_sin = F.pad(rotary_sin, (0, 0, 0, pad), value=0.0)
    return rotary_cos, rotary_sin


def _pad_dense_mask(mask: torch.Tensor, B: int, H: int, M: int, N: int) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    tm, tn = _round_up(M, 128), _round_up(N, 128)
    if mask.shape == (B, H, tm, tn):
        return mask
    out = mask.new_zeros((B, H, tm, tn))
    m = min(M, mask.shape[-2])
    n = min(N, mask.shape[-1])
    out[:, :, :m, :n] = mask[:, :, :m, :n]
    return out


def _pad_varlen_mask(
    mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    H: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Convert the packed V100 mask (total_q, H, max_k) to upstream's layout."""
    if mask.numel() == 0:
        return mask
    if mask.ndim == 4:  # already a (B, H, M, N) mask
        B = cu_seqlens_q.numel() - 1
        return _pad_dense_mask(mask, B, H, max_seqlen_q, max_seqlen_k)
    B = cu_seqlens_q.numel() - 1
    tq, tk = _round_up(max_seqlen_q, 128), _round_up(max_seqlen_k, 128)
    out = mask.new_zeros((B, H, tq, tk))
    cu = cu_seqlens_q.detach().to("cpu", dtype=torch.long).tolist()
    n = min(tk, mask.shape[-1])
    for b in range(B):
        start, end = int(cu[b]), int(cu[b + 1])
        lq = end - start
        if lq <= 0:
            continue
        out[b, :, : min(lq, tq), :n] = mask[start:end, :, :n].permute(1, 0, 2)[
            :, : min(lq, tq), :
        ]
    return out


# ---------------------------------------------------------------------------
# Dense (BMHD) entry points


def fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    d = q.shape[-1]
    B, M, H = q.shape[0], q.shape[1], q.shape[2]
    N = k.shape[1]
    # The V100 kernel cannot launch a backward with zero queries, and PyTorch's
    # autograd node is created before this adapter runs, so fail clearly instead
    # of letting the empty backward corrupt the CUDA context.
    if M == 0:
        raise RuntimeError(
            "M == 0 (empty query sequence) is not supported by the V100 build: "
            "the compiled kernel cannot launch its backward"
        )
    # The V100 kernel cannot launch for an empty KV sequence; upstream returns
    # zeros with lse == +inf (flash_api.cpp mha_fwd empty-KV branch).  B == 0 is
    # rejected by the extension exactly like upstream.
    if N == 0:
        result = torch.zeros_like(q)
        lse = torch.full(
            (B, H, M), float("inf"), dtype=torch.float32, device=q.device
        )
        return (
            result,
            lse,
            _dense_dmask(q, return_softmax, B, H, M, N),
            _rng_state(q),
        )
    t = _target_dim(d)
    qp, kp, vp = (_pad_dim(x, t) for x in (q, k, v))
    out_p, lse, dmask, rng_state = _real.fwd(
        qp,
        kp,
        vp,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
    )
    result = _slice_dim(out_p, d)
    if return_softmax:
        dmask = _pad_dense_mask(dmask, q.shape[0], q.shape[2], q.shape[1], k.shape[1])
    return result, lse, dmask, rng_state


def bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen,
    rng_state,
):
    d = q.shape[-1]
    B, M, H = q.shape[0], q.shape[1], q.shape[2]
    N = k.shape[1]
    if M == 0 or N == 0:
        # The V100 backward cannot launch with zero queries or an empty KV
        # sequence.  This mirrors upstream's zero-gradient result; the official
        # wrapper itself still forwards all other empty cases.
        gq = _empty_grad(dq, q)
        gk = _empty_grad(dk, k)
        gv = _empty_grad(dv, v)
        softmax_d = torch.zeros(
            (B, H, _round_up(M, 128)), dtype=torch.float32, device=q.device
        )
        return gq, gk, gv, softmax_d
    t = _target_dim(d)
    dout_p = _pad_dim(dout, t)
    qp, kp, vp = (_pad_dim(x, t) for x in (q, k, v))
    op = _pad_dim(out, t)
    dqp = None if dq is None else torch.empty_like(qp)
    dkp = None if dk is None else torch.empty_like(kp)
    dvp = None if dv is None else torch.empty_like(vp)
    grads = _real.bwd(
        dout_p,
        qp,
        kp,
        vp,
        op,
        softmax_lse,
        dqp,
        dkp,
        dvp,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        gen,
        rng_state,
    )
    gq, gk, gv = (_slice_dim(g, d) for g in grads[:3])
    if dq is not None:
        dq.copy_(gq)
        gq = dq
    if dk is not None:
        dk.copy_(gk)
        gk = dk
    if dv is not None:
        dv.copy_(gv)
        gv = dv
    softmax_d = grads[3]
    if softmax_d.numel():
        tm = _round_up(softmax_d.shape[-1], 128)
        if softmax_d.shape[-1] != tm:
            softmax_d = F.pad(softmax_d, (0, tm - softmax_d.shape[-1]))
    return gq, gk, gv, softmax_d


# ---------------------------------------------------------------------------
# Variable-length entry points


def varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    block_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen=None,
    num_splits=0,
):
    d = q.shape[-1]
    T, H = q.shape[0], q.shape[1]
    B = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else 0
    t = _target_dim(d)
    qp, kp, vp = (_pad_dim(x, t) for x in (q, k, v))
    # The V100 packed-mask writer needs aligned row strides; upstream rounds the
    # mask to 128 internally.  Rounding max_seqlen here is safe (the real lengths
    # come from cu_seqlens) and keeps return_softmax aligned.
    mq = _round_up(max_seqlen_q, 64)
    mk = _round_up(max_seqlen_k, 64)
    out_p, lse, dmask, rng_state = _real.varlen_fwd(
        qp,
        kp,
        vp,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        mq,
        mk,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
        int(num_splits or 0),
    )
    result = _slice_dim(out_p, d)
    if return_softmax:
        dmask = _pad_varlen_mask(
            dmask, cu_seqlens_q, q.shape[1], max_seqlen_q, max_seqlen_k
        )
    return result, lse, dmask, rng_state


def varlen_bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen,
    rng_state,
):
    d = q.shape[-1]
    T, H = q.shape[0], q.shape[1]
    B = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else 0
    t = _target_dim(d)
    dout_p = _pad_dim(dout, t)
    qp, kp, vp = (_pad_dim(x, t) for x in (q, k, v))
    op = _pad_dim(out, t)
    dqp = None if dq is None else torch.empty_like(qp)
    dkp = None if dk is None else torch.empty_like(kp)
    dvp = None if dv is None else torch.empty_like(vp)
    mq = _round_up(max_seqlen_q, 64)
    mk = _round_up(max_seqlen_k, 64)
    grads = _real.varlen_bwd(
        dout_p,
        qp,
        kp,
        vp,
        op,
        softmax_lse,
        dqp,
        dkp,
        dvp,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        mq,
        mk,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        gen,
        rng_state,
    )
    gq, gk, gv = (_slice_dim(g, d) for g in grads[:3])
    if dq is not None:
        dq.copy_(gq)
        gq = dq
    if dk is not None:
        dk.copy_(gk)
        gk = dk
    if dv is not None:
        dv.copy_(gv)
        gv = dv
    softmax_d = grads[3]
    if softmax_d.numel():
        target = q.shape[0] + 128 * (cu_seqlens_q.numel() - 1)
        if softmax_d.shape[-1] != target:
            softmax_d = F.pad(softmax_d, (0, target - softmax_d.shape[-1]))
    return gq, gk, gv, softmax_d


# ---------------------------------------------------------------------------
# KV-cache entry point


def fwd_kvcache(
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
    window_size_left,
    window_size_right,
    softcap,
    rotary_interleaved,
    num_splits,
):
    d = q.shape[-1]
    B, M, H = q.shape[0], q.shape[1], q.shape[2]
    T_NEW = k.shape[1] if k is not None else 0
    if block_table is not None:
        cache_capacity = block_table.shape[1] * k_cache.shape[1]
    else:
        cache_capacity = k_cache.shape[1]

    if cache_capacity == 0:
        if k is not None and T_NEW > 0:
            raise RuntimeError(
                "V100 compatibility: cannot append new K/V to a zero-length "
                "KV cache"
            )
        return (
            torch.zeros_like(q),
            torch.full((B, H, M), float("inf"), dtype=torch.float32, device=q.device),
        )

    t = _target_dim(d)
    qp = _pad_dim(q, t)
    kp = _pad_dim(k, t)
    vp = _pad_dim(v, t)
    if cache_batch_idx is not None and block_table is not None:
        raise RuntimeError("Paged KVcache does not support cache_batch_idx")
    dest_rows = _cache_batch_rows(cache_batch_idx, k_cache.shape[0])
    if dest_rows is not None:
        # The V100 kernel indexes cache_seqlens by cache row instead of by batch
        # (upstream indexes it by batch and uses cache_batch_idx only for the
        # address).  Gather the addressed cache rows so that kernel row b is
        # batch b; no cache_batch_idx needs to reach the kernel.
        rows = dest_rows.to(device=k_cache.device)
        kc = _pad_dim(k_cache.index_select(0, rows), t)
        vc = _pad_dim(v_cache.index_select(0, rows), t)
        kernel_batch_idx = None
    else:
        kc = _pad_dim(k_cache, t)
        vc = _pad_dim(v_cache, t)
        kernel_batch_idx = cache_batch_idx
    # The V100 kernel appends into the cache tensor it receives.  If padding
    # the head dimension (or gathering/contiguifying the cache) produced a
    # copy, the appended rows have to be copied back ourselves.
    need_copy_back = (
        k is not None and T_NEW > 0 and kc.data_ptr() != k_cache.data_ptr()
    )
    if need_copy_back and block_table is not None:
        raise NotImplementedError(
            "V100 compatibility: appending K/V to a paged cache is only "
            "supported when the cache can be passed to the kernel without a "
            "head-dimension padding copy"
        )

    rotary_cos, rotary_sin = _prep_rotary(
        rotary_cos,
        rotary_sin,
        d,
        cache_capacity,
        T_NEW,
        M,
        cache_seqlens,
        cache_leftpad,
    )
    out_p, softmax_lse = _real.fwd_kvcache(
        qp,
        kc,
        vc,
        kp,
        vp,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        kernel_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        rotary_interleaved,
        num_splits,
    )
    result = _slice_dim(out_p, d)
    if need_copy_back:
        _copy_kv_cache_back(
            k_cache,
            v_cache,
            kc,
            vc,
            cache_seqlens,
            dest_rows,
            cache_leftpad,
            T_NEW,
            d,
        )
    return result, softmax_lse


class _ExtensionProxy(types.ModuleType):
    """Drop-in replacement for the private extension module."""

    def __init__(self, real):
        super().__init__(_EXT_NAME)
        object.__setattr__(self, "_real", real)
        self.fwd = fwd
        self.bwd = bwd
        self.varlen_fwd = varlen_fwd
        self.varlen_bwd = varlen_bwd
        self.fwd_kvcache = fwd_kvcache

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __dir__(self):
        return sorted(set(dir(object.__getattribute__(self, "_real"))) | set(vars(self)))


def patch_interface():
    """Route backward through the official Python wrapper / V100 adapter.

    The compiled extension pre-registers ``torch.ops.flash_attn._flash_attn_backward``
    for torch.compile; the official interface therefore binds its ``_wrapped_*``
    backward globals to that custom op, bypassing this adapter.  Rebinding them to
    plain Python functions that call the proxied extension keeps the documented
    public API while letting the adapter handle empty queries and exact shapes.
    The forward custom op is left intact so torch.compile still sees a custom op.
    """
    import flash_attn.flash_attn_interface as fi

    def _maybe_contiguous(x):
        return x.contiguous() if x is not None and x.stride(-1) != 1 else x

    def _python_dense_backward(
        dout, q, k, v, out, softmax_lse, dq, dk, dv,
        dropout_p, softmax_scale, causal, window_size_left, window_size_right,
        softcap, alibi_slopes, deterministic, rng_state=None,
    ):
        dout, q, k, v, out = (_maybe_contiguous(x) for x in (dout, q, k, v, out))
        return bwd(
            dout, q, k, v, out, softmax_lse, dq, dk, dv,
            alibi_slopes, dropout_p, softmax_scale, causal,
            window_size_left, window_size_right, softcap,
            deterministic, None, rng_state,
        )[3]

    def _python_varlen_backward(
        dout, q, k, v, out, softmax_lse, dq, dk, dv,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size_left, window_size_right,
        softcap, alibi_slopes, deterministic, rng_state=None, zero_tensors=False,
    ):
        dout, q, k, v, out = (_maybe_contiguous(x) for x in (dout, q, k, v, out))
        return varlen_bwd(
            dout, q, k, v, out, softmax_lse, dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k, alibi_slopes,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
            zero_tensors, causal, window_size_left, window_size_right,
            softcap, deterministic, None, rng_state,
        )[3]

    fi._wrapped_flash_attn_backward = _python_dense_backward
    fi._wrapped_flash_attn_varlen_backward = _python_varlen_backward


# Install the proxy before flash_attn_interface imports flash_attn_2_cuda.
sys.modules[_EXT_NAME] = _ExtensionProxy(_real)
