# 03 — API Parity Analysis: upstream `flash-attn` 2.8.3 vs `flash-attention-v100`

Subagent: API-parity analyst (static/AST only — no CUDA, no GPU on this machine).
Deliverables of this task:

| file | content |
|---|---|
| `research/03_api_parity.md` | this report |
| `research/03_api_spec.json` | machine-checkable symbol spec (params + defaults + required flags + recommendations) |

Scratch/reusable tooling (outside the repo, as instructed): `/tmp/opencode/fa_upstream/`
(`parity_ast.py` checker, `ast_inventory.py`, `build_spec.py`, `parity_report.json`, `parity_output.txt`, `diff_table.md`, extracted upstream tree).

---

## 1. Upstream source provenance

* Obtained from PyPI **without installing**: `flash_attn-2.8.3.tar.gz` (sdist), URL from `https://pypi.org/pypi/flash-attn/2.8.3/json`.
* **SHA256 of tarball**: `1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d`
  (first attempt without proxy produced a truncated download; re-download via proxy `127.0.0.1:7890` succeeded, `gzip -t` OK, 8,447,812 bytes).
* Extracted to `/tmp/opencode/fa_upstream/flash_attn-2.8.3/`; sdist contains the full C++/CUDA `csrc/` tree, `tests/`, `hopper/`, and the `flash_attn` python package (85 `.py` files).
* **License**: BSD-3-Clause (`Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file`). Reusing signatures and porting signature-compatible pure-python wrappers is fine; keep the copyright/license notice (the fork already does this in `flash_attn/bert_padding.py`, which is an upstream copy).
* Upstream packaging metadata: `Requires-Python: >=3.9`, `Requires-Dist: torch`, `Requires-Dist: einops`. Core `flash_attn` import needs only torch + einops.

## 2. Method

* Static AST parsing only (`ast_inventory.py`, stdlib). No upstream/import execution, no CUDA.
* The repo package is parsed with import-aware resolution: `flash_attn/*` → `flash_attn_v100/*` re-exports are followed, so a symbol reported as EXACT is compared against the *resolved* defining signature.
* Verdicts: `EXACT` (same kind + same param names/kinds/defaults), `SIG-MISMATCH`, `KIND-MISMATCH`, `MISSING`, `EXTRA`.
* **The repo working tree was being updated by the implementation agent while this analysis ran.** All verdicts below reflect the tree as of 2026-10-19 ~14:56 (UTC+08) = git `55d685a` + working-tree changes; the exact file hashes are pinned in `03_api_spec.json → repo_snapshot`. Re-run the checker to re-validate a newer tree.
* Rerun:
  ```bash
  cd /tmp/opencode/fa_upstream
  python3 parity_ast.py --upstream flash_attn-2.8.3/flash_attn \
        --repo /home/eculid/Documents/Github/flash-attention-v100/flash_attn \
        --repo-pkg /home/eculid/Documents/Github/flash-attention-v100/flash_attn_v100 \
        --json parity_report.json
  # exit code 0 iff every REQUIRED symbol is EXACT; currently 1 (1 required failure: FlashAttnVarlenFunc arg name)
  ```
* Script hashes: `parity_ast.py` sha256 `cdd5096e…` (updated during this refresh: relative-import level preserved during parsing — fixes cross-package resolution; conditional-import equivalence for watchlisted imports — `flash_attn_gpu`); `ast_inventory.py` `4e179e4b…`; `build_spec.py` `3614d8f7…`.

## 3. Upstream 2.8.3 core inventory (what `import flash_attn` users can touch)

### 3.1 `flash_attn/__init__.py` (11 lines, no `__all__`)

```python
__version__ = "2.8.3"
from flash_attn.flash_attn_interface import (
    flash_attn_func, flash_attn_kvpacked_func, flash_attn_qkvpacked_func,
    flash_attn_varlen_func, flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func, flash_attn_with_kvcache,
)
```

Exactly **7 re-exported functions** + `__version__`. `flash_attn.bert_padding` is *not* re-exported (downstream imports it as a submodule), and `flash_attn.alibi_slopes`-style helpers do not exist in 2.8.3.

### 3.2 `flash_attn/flash_attn_interface.py`

Import-time (line 11–15): `USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE","FALSE")=="TRUE"`; then either `from .flash_attn_triton_amd import interface_fa as flash_attn_gpu` or **`import flash_attn_2_cuda as flash_attn_gpu`** — so in 2.8.3 `flash_attn_gpu` is the **CUDA extension module**, not a function alias.
The module also imports `torch`, `torch.nn as nn`, `os`, `typing.{Optional,Sequence,Tuple,Union}`.

**Public API functions (exact signatures, copied verbatim from AST):**

```
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                deterministic=False, return_attn_probs=False)

flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                       softcap=0.0, alibi_slopes=None, deterministic=False,
                       return_attn_probs=False, block_table=None)

flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None,
                        cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None,
                        block_table=None, softmax_scale=None, causal=False, window_size=(-1, -1),
                        softcap=0.0, rotary_interleaved=True, alibi_slopes=None,
                        num_splits=0, return_softmax_lse=False)

flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                          window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                          deterministic=False, return_attn_probs=False)

flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                         window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                         deterministic=False, return_attn_probs=False)

flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                 softmax_scale=None, causal=False, window_size=(-1, -1),
                                 softcap=0.0, alibi_slopes=None, deterministic=False,
                                 return_attn_probs=False)

flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                dropout_p=0.0, softmax_scale=None, causal=False,
                                window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                                deterministic=False, return_attn_probs=False)
```

**Autograd classes** (`torch.autograd.Function`): `FlashAttnQKVPackedFunc`, `FlashAttnVarlenQKVPackedFunc`, `FlashAttnKVPackedFunc`, `FlashAttnVarlenKVPackedFunc`, `FlashAttnFunc`, `FlashAttnVarlenFunc`. Their `forward(ctx, …)` argument lists are the function args + `torch.is_grad_enabled()` (see `03_api_spec.json` for each method’s full param list). `FlashAttnVarlenFunc.forward` is the only one with `block_table` before `is_grad_enabled`.

**Module-level torch.compile plumbing (torch ≥ 2.4 uses `torch.library.custom_op`)**: `_torch_custom_op_wrapper`, `_torch_register_fake_wrapper` (+ `noop_*` fallbacks for torch < 2.4), the custom ops `_flash_attn_forward`, `_flash_attn_varlen_forward`, `_flash_attn_backward`, `_flash_attn_varlen_backward`, their `_…_fake` variants, and the bound op handles `_wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward` etc.

**Helpers**: `maybe_contiguous(x)` (contig if `x.stride(-1) != 1`), `round_multiple(x, m)`, `_get_block_size_n(device, head_dim, is_dropout, is_causal)`.

### 3.3 `flash_attn/bert_padding.py`

Classes `IndexFirstAxis`, `IndexPutFirstAxis`, `IndexFirstAxisResidual`; alias assigns `index_first_axis`, `index_put_first_axis`, `index_first_axis_residual`; functions:

```
unpad_input(hidden_states, attention_mask, unused_mask=None)   -> 5-tuple (hidden, indices, cu_seqlens, max_seqlen, used_seqlens)
unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length) -> 4-tuple
pad_input(hidden_states, indices, batch, seqlen)
```

Note: `unpad_input_only` does **not** exist in 2.8.3 (it was removed); nothing to port. Imports: `torch`, `torch.nn.functional`, `einops.{rearrange,repeat}`.

### 3.4 Legacy alias names (checked explicitly)

* `flash_attn_gpu` — defined only in `flash_attn/flash_attn_interface.py`, and it is the **CUDA extension module** (or the triton-AMD interface). It is **not** `= flash_attn_func`.
* `flash_attn_varlen_gpu` and `flash_attn_with_kvcache_gpu` — **do not exist anywhere** in upstream 2.8.3 (full-tree grep including `tests/` and `csrc/` returns nothing).
* The only other consumer of `flash_attn_gpu` in the upstream tree is `flash_attn/flash_attn_triton_amd/fp8.py`.
* Consequence for the fork: its `flash_attn_gpu = flash_attn_func` / `flash_attn_varlen_gpu` / `flash_attn_with_kvcache_gpu` aliases are **fork-only extras**; they do not break imports, but `flash_attn.flash_attn_interface.flash_attn_gpu` in the fork is a *function*, whereas upstream code that does `flash_attn_gpu.fwd(...)`/`.varlen_fwd(...)` expects the extension **module**. See §7.4.

### 3.5 Raw extension module `flash_attn_2_cuda`

Upstream `PYBIND11_MODULE` (csrc/flash_attn/flash_api.cpp:1478) exposes exactly 5 functions:

```
fwd(q, k, v, out, alibi_slopes, p_dropout, softmax_scale, is_causal,
    window_size_left, window_size_right, softcap, return_softmax, gen)
varlen_fwd(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table,
           alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale,
           zero_tensors, is_causal, window_size_left, window_size_right, softcap,
           return_softmax, gen)                        # 21 args
bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, p_dropout, softmax_scale,
    is_causal, window_size_left, window_size_right, softcap, deterministic, gen, rng_state)
varlen_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
           alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, zero_tensors,
           is_causal, window_size_left, window_size_right, softcap, deterministic, gen, rng_state)
fwd_kvcache(q, kcache, vcache, k, v, seqlens_k, rotary_cos, rotary_sin, cache_batch_idx,
            leftpad_k, block_table, alibi_slopes, out, softmax_scale, is_causal,
            window_size_left, window_size_right, softcap, is_rotary_interleaved, num_splits)
```

The fork (`kernel/api.cpp`) declares **two** `PYBIND11_MODULE` blocks in the same `.so` (`TORCH_EXTENSION_NAME = flash_attn_v100_cuda` and a literal `flash_attn_2_cuda`) and `setup.py` symlinks `flash_attn_2_cuda.so → flash_attn_v100_cuda.so`; both names import and expose the same 5 function names. Signature deltas for raw-extension callers:

* `fwd`/`bwd`/`fwd_kvcache`: same argument order and count as upstream.
* `varlen_fwd`: fork has a **22nd** argument `num_splits` with no default (upstream has 21) → direct positional call with 21 args raises `TypeError`. Python-level API is unaffected. Suggest a pybind default (`py::arg("num_splits") = 0`) or dropping the argument.
* Return values: both return tuples `(out, softmax_lse, dmask, rng_state)` / `(dq, dk, dv, softmax_d)` / `(out, softmax_lse)`; the fork’s python relies on the same shapes.

### 3.6 Package tree and import-time dependencies (portability triage)

85 `.py` files: `flash_attn/` core, `layers/`, `losses/`, `models/`, `modules/`, `ops/{activations,fused_dense,layer_norm,rms_norm,triton/*}`, `utils/*`, `cute/` (22 files), `flash_attn_triton*.py`, `flash_attn_triton_amd/*`, `flash_blocksparse*`.

All sub-package `__init__.py` files (`layers`, `losses`, `models`, `modules`, `ops`, `utils`) are **empty** — importing them costs nothing; the heavy work happens on submodule import:

| area | import-time deps | pure-python portable? | recommendation |
|---|---|---|---|
| `layers/rotary.py` | `einops` + `flash_attn.ops.triton.rotary` (triton) | yes, file already contains a torch fallback `apply_rotary_emb_torch` | optional P3: port, drop/replace the triton import |
| `layers/patch_embed.py` | `einops`; `FusedDense` import is guarded | yes when FusedDense missing | optional P3 |
| `losses/cross_entropy.py` | triton kernel | no | skip P4 (no triton on env) |
| `ops/activations.py` | torch only | yes | optional P3 |
| `ops/layer_norm.py`, `ops/rms_norm.py` | separate compiled ext `dropout_layer_norm` (built by upstream setup.py, **not** by the fork) | no | skip P4 unless the ext is added |
| `ops/fused_dense.py` | `fused_dense_cuda` ext + distributed | no | skip P4 |
| `ops/triton/*` | triton | no | skip P4 |
| `models/*` | `transformers`, `einops`, `sentencepiece` | no | out of scope P4 |
| `modules/*` | torch, einops, apex/triton guarded | mostly | out of scope P4 |
| `utils/benchmark.py`, `utils/distributed.py`, `utils/torch.py` | torch only | yes | optional P3 |
| `utils/testing.py` | einops | yes | optional P4 |
| `utils/generation.py`, `utils/pretrained.py`, `utils/library.py` | transformers/hf_hub (guarded), torch.library | partly | optional P4 |
| `cute/*` | nvidia cutlass python DSL; Hopper/Blackwell only | no | skip P4 (not V100-capable) |
| `flash_attn_triton*.py`, `flash_attn_triton_amd/*` | triton / ROCm | no | skip P4 |
| `flash_blocksparse*` | hydra + FA1 `flash_attn_cuda` blocksparse ext | no | skip P4 (FA1-era, unrelated to FA2 API) |

**Recommended parity scope**: core = `flash_attn/__init__.py` + `flash_attn/flash_attn_interface.py` + `flash_attn/bert_padding.py` (torch + einops only), plus the *pure-python* item `layers/rotary.py`/`ops/activations.py`/`utils/{benchmark,distributed,torch}.py` if a downstream user needs them. Everything else is out of scope.

## 4. Repo shim inventory (AST, as of the 2026-10-19 ~14:56 snapshot)

* `flash_attn/__init__.py` — `pkgutil.extend_path` (namespace merging; harmless extra), re-exports **10 names** from `flash_attn_v100` (the 7 upstream exports + `flash_attn_gpu`, `flash_attn_varlen_gpu`, `flash_attn_with_kvcache_gpu`), `__version__ = "2.8.3"`, `__all__` = those 10 + `__version__`. Extras vs upstream: the 3 `*_gpu` function aliases + `_v100_version`.
* `flash_attn/flash_attn_interface.py` — re-exports the same 10 functions **plus `FlashAttnFunc`, `FlashAttnVarlenFunc`, `maybe_contiguous`**; binds `flash_attn_gpu` to the compiled extension module (`import flash_attn_2_cuda as flash_attn_gpu`, fallback `flash_attn_v100_cuda`) — matching upstream's non-ROCm branch. Still absent here: the 4 aux classes, `_flash_attn_*` internals, `round_multiple`, `USE_TRITON_ROCM` (rows 9-29 in §5).
* `flash_attn/bert_padding.py` — full upstream-equivalent implementation (all 3 classes, 3 aliases, 3 functions; `unpad_input` returns the 5-tuple incl. `used_seqlens_in_batch`) → **all 9 symbols EXACT**.
* `flash_attn_v100/__init__.py` — `__version__ = "26.06"`, same 10 names.
* `flash_attn_v100/flash_attn_interface.py` — defines the 7 public functions + the **4 packed wrappers**, `FlashAttnFunc` (param-identical to upstream), `FlashAttnVarlenFunc` (only delta: forward arg #15 `return_attn_probs` vs upstream `return_softmax` — the single remaining required failure), `maybe_contiguous`, and a fork-only `_round_head_dim` helper (pads head dim to the next supported kernel bucket). Dense `return_attn_probs` now mirrors upstream (`return_softmax and dropout_p > 0.0` to the kernel, 3-tuple with empty dmask still returned); empty-input short-circuits added; `deterministic=True` still warns and is silently disabled (§7.2). No `round_multiple` / `_get_block_size_n` / `USE_TRITON_ROCM` / aux classes / torch.compile plumbing yet.
* `kernel/api.cpp` — dual-module pybind trick (see §3.5) + `TORCH_LIBRARY(flash_attn_v100, …)` torchbind ops (fork-only extra); unchanged in this refresh.
* Build notes (handled by other agents, updated 2026-10-19): `setup.py` now targets **CUDA ≥ 12.6** (`FLASH_ATTN_MIN_CUDA` override) with `sm_70` gencode; `pyproject.toml` pins `torch==2.6.0`; the compat `flash_attn/` shim is copied into the wheel by the custom `build_py` (`CopyAttention`) and a `flash_attn-2.8.3.dist-info` placeholder is written on install (`InstallAttention`). The earlier CUDA 12.9 / torch 2.10 pins have been replaced.

## 5. Diff table (upstream 2.8.3 → repo status)

Statuses are computed by `parity_ast.py`; full recommendations live in `03_api_spec.json`.
| # | import path | repo status | upstream signature / kind (2.8.3) | recommendation |
|---|---|---|---|---|
| 1 | `flash_attn.flash_attn_func` | **EXACT** | `(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)` | DONE (EXACT). Return semantics now match upstream: tuple only when return_attn_probs, S_dmask empty when dropout_p==0 (fixed 2026-10-19). Verify lse/dmask at runtime. |
| 2 | `flash_attn.flash_attn_kvpacked_func` | **EXACT** | `(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)` | DONE (EXACT, implemented 2026-10-19): k,v = kv.unbind(2) -> flash_attn_func; shape guard (B,S,2,Hk,D); GQA via H % Hk == 0. |
| 3 | `flash_attn.flash_attn_qkvpacked_func` | **EXACT** | `(qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)` | DONE (EXACT, implemented 2026-10-19): q,k,v = qkv.unbind(2) -> flash_attn_func; shape guard (B,S,3,H,D). |
| 4 | `flash_attn.flash_attn_varlen_func` | **EXACT** | `(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None...` | DONE (EXACT). Keep delegation; verify block_table + return_attn_probs semantics at runtime. |
| 5 | `flash_attn.flash_attn_varlen_kvpacked_func` | **EXACT** | `(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, ...` | DONE (EXACT, implemented 2026-10-19): k,v = kv.unbind(1) -> flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,...); shape guard (Tk,2,Hk,D). |
| 6 | `flash_attn.flash_attn_varlen_qkvpacked_func` | **EXACT** | `(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_p...` | DONE (EXACT, implemented 2026-10-19): q,k,v = qkv.unbind(1) -> flash_attn_varlen_func(..., cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, ...); shape guard (T,3,H,D). |
| 7 | `flash_attn.flash_attn_with_kvcache` | **EXACT** | `(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None, softmax_scale=N...` | DONE (EXACT kwargs). Verify runtime: int cache_seqlens expansion, cache_batch_idx/cache_leftpad/block_table, rotary_interleaved, num_splits, return_softmax_lse dtype/shape (B,H,S) fp32. |
| 8 | `flash_attn.flash_attn_interface.FlashAttnFunc` | **EXACT** | `class (bases=class, methods=forward/backward) - see spec` | DONE (EXACT, 2026-10-19): flash_attn_v100 defines forward (ctx + 12 args) / backward param lists identical to upstream, and the shim re-exports it. Verify apply() at runtime. |
| 9 | `flash_attn.flash_attn_interface.FlashAttnKVPackedFunc` | **MISSING** | `class (bases=class, methods=forward/backward) - see spec` | P2 port class (ctx + 12 args). |
| 10 | `flash_attn.flash_attn_interface.FlashAttnQKVPackedFunc` | **MISSING** | `class (bases=class, methods=forward/backward) - see spec` | P2 port class or make it a thin autograd.Function built on the packed wrapper. |
| 11 | `flash_attn.flash_attn_interface.FlashAttnVarlenFunc` | **SIG-MISMATCH** | `class (bases=class, methods=forward/backward) - see spec` | ONLY remaining required failure: forward arg #15 is `return_attn_probs` vs upstream `return_softmax` (backward matches). Rename in flash_attn_v100 (and its internal caller) or wrap in the shim -> flips to EXACT. |
| 12 | `flash_attn.flash_attn_interface.FlashAttnVarlenKVPackedFunc` | **MISSING** | `class (bases=class, methods=forward/backward) - see spec` | P2 port class (ctx + 15 args). |
| 13 | `flash_attn.flash_attn_interface.FlashAttnVarlenQKVPackedFunc` | **MISSING** | `class (bases=class, methods=forward/backward) - see spec` | P2 port class (ctx + 13 args). |
| 14 | `flash_attn.flash_attn_interface.USE_TRITON_ROCM` | **MISSING** | `` | P3 define USE_TRITON_ROCM = os.getenv('FLASH_ATTENTION_TRITON_AMD_ENABLE','FALSE') == 'TRUE' so upstream-style checks keep working. |
| 15 | `flash_attn.flash_attn_interface._flash_attn_backward` | **MISSING** | `(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=No...` | P2 port as plain function delegating to flash_attn_v100_cuda.bwd; returns softmax_d (not tuple). |
| 16 | `flash_attn.flash_attn_interface._flash_attn_backward_fake` | **MISSING** | `(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=No...` | P3 skip unless torch.compile parity required. |
| 17 | `flash_attn.flash_attn_interface._flash_attn_forward` | **MISSING** | `(q, k, v, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, return_softmax) -> Tuple[torch.Tensor, torch.Tensor, torch.Tens...` | P2 port as plain function calling flash_attn_v100_cuda.fwd with upstream arg order (q,k,v,dropout_p,softmax_scale,causal,window_size_left,window_size_right,softcap,alibi_slopes,return_softmax) -> (out, lse, dmask, rng_state). torch.libra... |
| 18 | `flash_attn.flash_attn_interface._flash_attn_forward_fake` | **MISSING** | `(q, k, v, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, return_softmax) -> Tuple[torch.Tensor, torch.Tensor, torch.Tens...` | P3 only needed for torch.compile fake-tensor support; skip unless torch.compile parity required. |
| 19 | `flash_attn.flash_attn_interface._flash_attn_varlen_backward` | **MISSING** | `(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left, window_size_r...` | P2 port delegating to flash_attn_v100_cuda.varlen_bwd; note extra zero_tensors=False kwarg. |
| 20 | `flash_attn.flash_attn_interface._flash_attn_varlen_backward_fake` | **MISSING** | `(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left, window_size_r...` | P3 skip unless torch.compile parity required. |
| 21 | `flash_attn.flash_attn_interface._flash_attn_varlen_forward` | **MISSING** | `(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slope...` | P2 port with upstream defaults (window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, return_softmax=False, block_table=None, leftpad_k=None, seqused_k=None, zero_tensors=False). |
| 22 | `flash_attn.flash_attn_interface._flash_attn_varlen_forward_fake` | **MISSING** | `(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slope...` | P3 skip unless torch.compile parity required. |
| 23 | `flash_attn.flash_attn_interface._get_block_size_n` | **MISSING** | `(device, head_dim, is_dropout, is_causal)` | P3 test/bench utility; copy or skip. Guarded by assert head_dim <= 256. |
| 24 | `flash_attn.flash_attn_interface._torch_custom_op_wrapper` | **MISSING** | `` | implement in fork (see detail). |
| 25 | `flash_attn.flash_attn_interface._torch_register_fake_wrapper` | **MISSING** | `` | implement in fork (see detail). |
| 26 | `flash_attn.flash_attn_interface._wrapped_flash_attn_backward` | **MISSING** | `` | implement in fork (see detail). |
| 27 | `flash_attn.flash_attn_interface._wrapped_flash_attn_forward` | **MISSING** | `` | implement in fork (see detail). |
| 28 | `flash_attn.flash_attn_interface._wrapped_flash_attn_varlen_backward` | **MISSING** | `` | implement in fork (see detail). |
| 29 | `flash_attn.flash_attn_interface._wrapped_flash_attn_varlen_forward` | **MISSING** | `` | implement in fork (see detail). |
| 30 | `flash_attn.flash_attn_interface.flash_attn_func` | **EXACT** | `(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)` | DONE (EXACT). Return semantics now match upstream: tuple only when return_attn_probs, S_dmask empty when dropout_p==0 (fixed 2026-10-19). Verify lse/dmask at runtime. |
| 31 | `flash_attn.flash_attn_interface.flash_attn_gpu` | **EXACT** | `` | RESOLVED (EXACT, 2026-10-19): the shim binds the extension module (`import flash_attn_2_cuda as flash_attn_gpu`, fallback `flash_attn_v100_cuda`), matching upstream's non-ROCm branch; checker accepts conditional-import equivalence. |
| 32 | `flash_attn.flash_attn_interface.flash_attn_kvpacked_func` | **EXACT** | `(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)` | DONE (EXACT, implemented 2026-10-19): k,v = kv.unbind(2) -> flash_attn_func; shape guard (B,S,2,Hk,D); GQA via H % Hk == 0. |
| 33 | `flash_attn.flash_attn_interface.flash_attn_qkvpacked_func` | **EXACT** | `(qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)` | DONE (EXACT, implemented 2026-10-19): q,k,v = qkv.unbind(2) -> flash_attn_func; shape guard (B,S,3,H,D). |
| 34 | `flash_attn.flash_attn_interface.flash_attn_varlen_func` | **EXACT** | `(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None...` | DONE (EXACT). Keep delegation; verify block_table + return_attn_probs semantics at runtime. |
| 35 | `flash_attn.flash_attn_interface.flash_attn_varlen_kvpacked_func` | **EXACT** | `(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, ...` | DONE (EXACT, implemented 2026-10-19): k,v = kv.unbind(1) -> flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,...); shape guard (Tk,2,Hk,D). |
| 36 | `flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func` | **EXACT** | `(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_p...` | DONE (EXACT, implemented 2026-10-19): q,k,v = qkv.unbind(1) -> flash_attn_varlen_func(..., cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, ...); shape guard (T,3,H,D). |
| 37 | `flash_attn.flash_attn_interface.flash_attn_with_kvcache` | **EXACT** | `(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None, softmax_scale=N...` | DONE (EXACT kwargs). Verify runtime: int cache_seqlens expansion, cache_batch_idx/cache_leftpad/block_table, rotary_interleaved, num_splits, return_softmax_lse dtype/shape (B,H,S) fp32. |
| 38 | `flash_attn.flash_attn_interface.maybe_contiguous` | **EXACT** | `(x)` | DONE (EXACT, 2026-10-19): defined in flash_attn_v100 and re-exported by the shim (upstream tools import it from flash_attn.flash_attn_interface). |
| 39 | `flash_attn.flash_attn_interface.noop_custom_op_wrapper` | **MISSING** | `(name, fn=None, *, mutates_args, device_types=None, schema=None)` | implement in fork (see detail). |
| 40 | `flash_attn.flash_attn_interface.noop_register_fake_wrapper` | **MISSING** | `(op, fn=None, *, lib=None, _stacklevel=1)` | implement in fork (see detail). |
| 41 | `flash_attn.flash_attn_interface.round_multiple` | **MISSING** | `(x, m)` | v100 does not define it either; 2-line pure-python helper - copy verbatim and add to the shim re-export list (P3). |
| 42 | `flash_attn.bert_padding.IndexFirstAxis` | **EXACT** | `class (bases=class, methods=forward/backward) - see spec` | no action (verified exact signature). |
| 43 | `flash_attn.bert_padding.IndexFirstAxisResidual` | **EXACT** | `class (bases=class, methods=forward/backward) - see spec` | no action (verified exact signature). |
| 44 | `flash_attn.bert_padding.IndexPutFirstAxis` | **EXACT** | `class (bases=class, methods=forward/backward) - see spec` | no action (verified exact signature). |
| 45 | `flash_attn.bert_padding.index_first_axis` | **EXACT** | `` | no action (verified exact signature). |
| 46 | `flash_attn.bert_padding.index_first_axis_residual` | **EXACT** | `` | no action (verified exact signature). |
| 47 | `flash_attn.bert_padding.index_put_first_axis` | **EXACT** | `` | no action (verified exact signature). |
| 48 | `flash_attn.bert_padding.pad_input` | **EXACT** | `(hidden_states, indices, batch, seqlen)` | DONE (EXACT). |
| 49 | `flash_attn.bert_padding.unpad_input` | **EXACT** | `(hidden_states, attention_mask, unused_mask=None)` | DONE (EXACT, incl. 5-tuple return with used_seqlens_in_batch). |
| 50 | `flash_attn.bert_padding.unpad_input_for_concatenated_sequences` | **EXACT** | `(hidden_states, attention_mask_in_length)` | DONE (EXACT, 4-tuple). |

## 6. AST parity checker output (raw, as produced by the versioned checker)

```
$ python3 parity_ast.py --upstream flash_attn-2.8.3/flash_attn \
      --repo <repo>/flash_attn --repo-pkg <repo>/flash_attn_v100 --json parity_report.json
(exit code 1 — 1 required failure pending: FlashAttnVarlenFunc param rename)

=== flash_attn/__init__.py ===
  [EXACT        ] flash_attn_func   <== REQUIRED
  [EXACT        ] flash_attn_kvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_qkvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_varlen_func   <== REQUIRED
  [EXACT        ] flash_attn_varlen_kvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_varlen_qkvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_with_kvcache   <== REQUIRED
  [EXTRA        ] _v100_version   (repo-only, kind=assign)
  [EXTRA        ] flash_attn_gpu   (repo-only, kind=assign)
  [EXTRA        ] flash_attn_varlen_gpu   (repo-only, kind=assign)
  [EXTRA        ] flash_attn_with_kvcache_gpu   (repo-only, kind=assign)

=== flash_attn/flash_attn_interface.py ===
  [EXACT        ] FlashAttnFunc   <== REQUIRED
  [MISSING      ] FlashAttnKVPackedFunc   (upstream kind=class)
  [MISSING      ] FlashAttnQKVPackedFunc   (upstream kind=class)
  [SIG-MISMATCH ] FlashAttnVarlenFunc   <== REQUIRED
                    upstream: None
                    repo:     None
                    diff:     forward: #15 return_softmax.name: 'return_softmax' vs 'return_attn_probs'
  [MISSING      ] FlashAttnVarlenKVPackedFunc   (upstream kind=class)
  [MISSING      ] FlashAttnVarlenQKVPackedFunc   (upstream kind=class)
  [MISSING      ] USE_TRITON_ROCM   (upstream kind=assign)
  [MISSING      ] _flash_attn_backward   upstream sig=(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None) -> torch.Tensor   (upstream kind=function)
  [MISSING      ] _flash_attn_backward_fake   upstream sig=(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None) -> torch.Tensor   (upstream kind=function)
  [MISSING      ] _flash_attn_forward   upstream sig=(q, k, v, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, return_softmax) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]   (upstream kind=function)
  [MISSING      ] _flash_attn_forward_fake   upstream sig=(q, k, v, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, return_softmax) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]   (upstream kind=function)
  [MISSING      ] _flash_attn_varlen_backward   upstream sig=(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None, zero_tensors=False) -> torch.Tensor   (upstream kind=function)
  [MISSING      ] _flash_attn_varlen_backward_fake   upstream sig=(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None, zero_tensors=False) -> torch.Tensor   (upstream kind=function)
  [MISSING      ] _flash_attn_varlen_forward   upstream sig=(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, return_softmax=False, block_table=None, leftpad_k=None, seqused_k=None, zero_tensors=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]   (upstream kind=function)
  [MISSING      ] _flash_attn_varlen_forward_fake   upstream sig=(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, return_softmax=False, block_table=None, leftpad_k=None, seqused_k=None, zero_tensors=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]   (upstream kind=function)
  [MISSING      ] _get_block_size_n   upstream sig=(device, head_dim, is_dropout, is_causal)   (upstream kind=function)
  [MISSING      ] _torch_custom_op_wrapper   (upstream kind=assign)
  [MISSING      ] _torch_register_fake_wrapper   (upstream kind=assign)
  [MISSING      ] _wrapped_flash_attn_backward   (upstream kind=assign)
  [MISSING      ] _wrapped_flash_attn_forward   (upstream kind=assign)
  [MISSING      ] _wrapped_flash_attn_varlen_backward   (upstream kind=assign)
  [MISSING      ] _wrapped_flash_attn_varlen_forward   (upstream kind=assign)
  [EXACT        ] flash_attn_func   <== REQUIRED
  [EXACT        ] flash_attn_gpu
  [EXACT        ] flash_attn_kvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_qkvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_varlen_func   <== REQUIRED
  [EXACT        ] flash_attn_varlen_kvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_varlen_qkvpacked_func   <== REQUIRED
  [EXACT        ] flash_attn_with_kvcache   <== REQUIRED
  [EXACT        ] maybe_contiguous
  [MISSING      ] noop_custom_op_wrapper   upstream sig=(name, fn=None, *, mutates_args, device_types=None, schema=None)   (upstream kind=function)
  [MISSING      ] noop_register_fake_wrapper   upstream sig=(op, fn=None, *, lib=None, _stacklevel=1)   (upstream kind=function)
  [MISSING      ] round_multiple   upstream sig=(x, m)   (upstream kind=function)
  [EXTRA        ] flash_attn_varlen_gpu   (repo-only, kind=assign)
  [EXTRA        ] flash_attn_with_kvcache_gpu   (repo-only, kind=assign)

=== flash_attn/bert_padding.py ===
  [EXACT        ] IndexFirstAxis   <== REQUIRED
  [EXACT        ] IndexFirstAxisResidual   <== REQUIRED
  [EXACT        ] IndexPutFirstAxis   <== REQUIRED
  [EXACT        ] index_first_axis   <== REQUIRED
  [EXACT        ] index_first_axis_residual   <== REQUIRED
  [EXACT        ] index_put_first_axis   <== REQUIRED
  [EXACT        ] pad_input   <== REQUIRED
  [EXACT        ] unpad_input   <== REQUIRED
  [EXACT        ] unpad_input_for_concatenated_sequences   <== REQUIRED

--- summary ---
exact=26 mismatch=1 missing=23 extra=6 required_failures=1
required failures: flash_attn/flash_attn_interface.py::FlashAttnVarlenFunc
wrote parity_report.json
```

Interpretation: 26 of the 50 compared entries are EXACT = the 7 package exports × 2 modules + `FlashAttnFunc` + `maybe_contiguous` + the `flash_attn_gpu` extension-module binding in the shim + all 9 `bert_padding` symbols. Of the 25 required symbols, 24 are EXACT; the only failure is `FlashAttnVarlenFunc` (forward arg #15 `return_attn_probs` vs `return_softmax`) — a one-word rename away from exit code 0. All 23 MISSING entries are non-required internals: 4 aux packed autograd classes, 8 `_flash_attn_*` entries (4 op functions + 4 fakes), 4 `_wrapped_*` op handles, 2 `_torch_*` wrapper assigns, 2 `noop_*` fallbacks, `USE_TRITON_ROCM`, `_get_block_size_n`, `round_multiple`. The 6 EXTRA entries: `_v100_version`, the top-level `flash_attn_gpu` function alias, `flash_attn_varlen_gpu` and `flash_attn_with_kvcache_gpu` (in both modules).
## 7. Semantic / behavioural differences worth runtime tests

These are *not* signature differences — they are places where the fork’s runtime behaviour can diverge from upstream. Full structured list with severities is in `03_api_spec.json → semantics_to_test`.

### 7.1 `return_attn_probs=True` with `dropout_p=0` (P1 — FIXED 2026-10-19, verify at runtime)

* Upstream: the python layer passes `return_softmax and dropout_p > 0` to the kernel, so with dropout 0 the kernel gets `return_softmax=False`; the *user-facing* function still returns the 3-tuple `(out, softmax_lse, S_dmask)` where `S_dmask` is an empty tensor and `softmax_lse` is valid.
* Fork before the fix: dense `FlashAttnFunc.forward` passed the user flag **raw**; `kernel/forward.cu:367` raised `return_softmax requires p_dropout > 0` → `RuntimeError` where upstream returns a tuple.
* Fork now: dense forward passes `return_softmax and dropout_p > 0.0` to `flash_attn_v100_cuda.fwd` and still returns `(out, lse_, dmask_)` whenever the user requested probs; with `dropout_p == 0` the dmask is a 0-element tensor — the upstream contract. The varlen path already behaved this way.
* Remaining: runtime verification on a GPU (3-tuple shape, fp32 lse, `dmask.numel() == 0`).

### 7.2 `deterministic=True` (P1, accepted divergence)

Upstream honours `deterministic` in the backward (dq accumulation buffer) and only raises for unsupported shapes; the fork warns `RuntimeWarning("Forward is always deterministic. Deterministic backward is not supported.")` and continues non-deterministically. Behavioural difference, no signature change; document in the known-differences list (and consider whether warning on the default path is desired — it is only triggered when the user opts in).

### 7.3 Capability gaps in the fork’s C++ (mostly unfixable by signatures — current status per row)

| gap | upstream | fork | note |
|---|---|---|---|
| dtype | fp16 **and bf16** | `TORCH_CHECK(q.dtype() == kFloat16)` in all 4 forward/backward entry points; no bf16 code anywhere | V100 has no bf16 hardware — bf16 users will get `RuntimeError: q must be fp16`. Must be documented; do not silently cast. |
| head dim | any dim ≤ 256 via `HEADDIM_SWITCH` buckets {32,64,96,128,192,256} (python pads to a multiple of 8 first) | kernel dispatch unchanged **{16,32,64,128,256}**, but python now pads up to the next bucket (`_round_head_dim`, 2026-10-19) | 96/160/104/192/224 now work; >256 raises `ValueError` (upstream raises `RuntimeError` from C++ — consider aligning the error type). |
| arch | `cc_major >= 8` (Ampere+) else RuntimeError | `props->major == 7 && minor == 0` (Volta only) | intended divergence; tests must run on V100. |
| kvcache extras | `num_splits` 0..128, softcap+window, softcap+ALiBi, paged KV + `cache_batch_idx` | `num_splits <= 1`; `Softcap + window` / `Softcap + ALiBi` raise; paged KV + `cache_batch_idx` raises | document; tests should assert fork error messages for those combos. |
| compaction of errors | `TORCH_CHECK` messages as upstream | different messages (e.g. “q must be fp16”, “Unsupported D”) | error *types* match (RuntimeError); tests should not string-match upstream messages. |

### 7.4 `flash_attn_gpu` kind mismatch (P2 decision)

Upstream `flash_attn.flash_attn_interface.flash_attn_gpu` **is the extension module**; the fork originally pointed it (and `flash_attn_gpu` in `flash_attn/__init__`, which upstream does not export at all) at `flash_attn_func`. Code that only calls `flash_attn_gpu(...)` (fork heritage) keeps working; code that calls `flash_attn_gpu.fwd/.varlen_fwd/.bwd/.fwd_kvcache(...)` broke. **Resolved 2026-10-19:** the shim now binds `flash_attn_gpu` to the extension module (`import flash_attn_2_cuda as flash_attn_gpu`, fallback `flash_attn_v100_cuda`), matching upstream’s non-ROCm branch; the top-level `flash_attn.flash_attn_gpu` callable alias remains a documented fork extra.

### 7.5 Smaller notes

* `cache_seqlens` as `int`: upstream expands to `torch.full((k_cache.shape[0],), …)`, fork to `(q.shape[0],)` — equal in normal usage.
* Contiguity: upstream C++ requires only `stride(-1)==1` and honours other strides; the fork copies to contiguous after permuting to `(B,H,S,D)` — same results, extra copy.
* Head-dim padding: the fork pads any dim ≤ 256 up to the next supported bucket {16,32,64,128,256} (`_round_head_dim`, added 2026-10-19) and slices the output back; upstream pads to a multiple of 8 and relies on `HEADDIM_SWITCH`. The fork returns a contiguous out, upstream may return a non-contiguous view (`out_padded[..., :head_size_og]`) — only observable via `is_contiguous()`/strides.
* lse: dense `(B,H,S)` fp32; varlen `(H, total_q)` fp32 — assert dtype/shape at runtime for both `return_attn_probs` and `return_softmax_lse`.
* `flash_attn.flash_attn_interface.maybe_contiguous` is now importable (re-exported from the shim, EXACT — used by e.g. mamba-style code); `round_multiple` (same upstream module) is still absent — add it if downstream imports it.
* `softcap > 0` + `dropout_p > 0` raises on both (same check); `softcap ≤ 0` deactivates.
* Window size: `(-1,-1)` infinite; upstream normalizes `window_left >= seqlen_k → -1` in C++.
* `USE_TRITON_ROCM` env flag exists upstream at import time; the fork does not define it — add for code that reads it.
* Empty inputs: the fork now short-circuits `B==0`/`M==0` (dense) and `T==0` (varlen) with zero-filled outputs and `+inf`-filled lse (`empty_inputs` item in the spec); upstream behavior unverified without a GPU.

## 8. Implementation recommendations (ordered)

**P1 — done in the 2026-10-19 working tree, except one rename (status as of this snapshot):**

1. ✅ The 4 packed wrappers are implemented in `flash_attn_v100/flash_attn_interface.py` (`qkv.unbind(2)` / `kv.unbind(2)` / varlen `unbind(1)` → delegate to the dense/varlen functions with identical kwargs; shape guards) and re-exported through `flash_attn/flash_attn_interface.py` + `flash_attn/__init__.py`.
2. ✅ `return_attn_probs`/`dropout_p==0` divergence fixed (§7.1).
3. ✅ `maybe_contiguous` and `FlashAttnFunc` re-exported from `flash_attn/flash_attn_interface.py`; `flash_attn_gpu` now binds the extension module (§7.4).
4. ⏳ **Remaining (the only required failure):** rename `FlashAttnVarlenFunc.forward` arg #15 `return_attn_probs` → `return_softmax` (and its internal caller), or wrap the class in the shim so `inspect.signature` matches upstream. After this the checker exits 0.
5. ✅ Optional stretch from the earlier pass: head-dim bucket padding (`_round_head_dim`) restores 96/160/192/224 parity without kernel work.

**P2 — full private/auxiliary parity:**

6. Port the 4 packed `torch.autograd.Function` classes (upstream `forward` arg lists in the spec).
7. Port `_flash_attn_forward`, `_flash_attn_varlen_forward`, `_flash_attn_backward`, `_flash_attn_varlen_backward` as plain functions calling `flash_attn_v100_cuda.{fwd,varlen_fwd,bwd,varlen_bwd}` with upstream argument order and defaults (the varlen ones include `leftpad_k`/`seqused_k`/`zero_tensors`). Optionally register them as `torch.library.custom_op("flash_attn::…")` when torch ≥ 2.4 to restore `torch.ops.flash_attn.*` and `torch.compile` fake support (`_…_fake` + `_wrapped_*` handles).
8. Add `round_multiple` (2-line pure-python helper) + shim re-export.

**P3 — cosmetics/test utilities:** `USE_TRITON_ROCM`, `_get_block_size_n`, `noop_custom_op_wrapper`, `noop_register_fake_wrapper` (only needed if code imports them), pure-python optional submodules (`layers.rotary`, `ops.activations`, `utils.benchmark/distributed/torch`).

**P4 — out of scope:** triton kernels, `ops/layer_norm|rms_norm|fused_dense` (separate extensions), model zoo, `cute`, FA1 blocksparse.

## 9. Adapter for repo tests (next phase)

`parity_ast.py` (stdlib-only) can be dropped into `tests/` unchanged and run in CI without a GPU; the runtime variant should compare `inspect.signature(<symbol>)` against `03_api_spec.json`:

```python
spec = json.load(open("research/03_api_spec.json"))
for s in spec["symbols"]:
    if not s["required"] or s["kind"] != "function":
        continue
    obj = eval(f"import {s['module']}; {s['module']}.{s['name']}")  # or importlib
    got = inspect.signature(obj).parameters
    exp = [(p["name"], p["kind"], p["default"]) for p in s["params"]]
    # compare names, kinds (POSITIONAL_OR_KEYWORD…), and repr-normalised defaults
```

Notes for that test: annotations are intentionally excluded from the verdicts (upstream annotates only some args); defaults must be compared as normalised strings (`None`, `True/False`, `0.0`, `(-1, -1)`); classes are compared on `forward`/`backward` parameter lists; keyword names matter — `FlashAttnVarlenFunc.forward` must use `return_softmax`, not the current `return_attn_probs` (the single failing required item; checker exit code 1 as of the snapshot). `03_api_spec.json → repo_snapshot` pins the file hashes the verdicts were computed against.

## 10. Summary (≤15 lines)

1. Upstream 2.8.3 obtained from PyPI sdist, sha256 `1e71dd64…`, BSD-3-Clause; AST-only analysis (no imports/GPU).
2. Deliverables: `research/03_api_parity.md`, `research/03_api_spec.json` (incl. `repo_snapshot` file hashes); checker at `/tmp/opencode/fa_upstream/parity_ast.py`.
3. Verdicts as of the 2026-10-19 ~14:56 repo snapshot: **26 EXACT, 23 MISSING, 1 SIG-MISMATCH, 6 EXTRA — 24/25 required EXACT, exit code 1.**
4. The single remaining required failure: `FlashAttnVarlenFunc.forward` arg #15 named `return_attn_probs` instead of upstream `return_softmax` — a one-word rename.
5. Present and exact: all 7 `flash_attn` exports (incl. the 4 packed functions), `FlashAttnFunc`, `maybe_contiguous`, the `flash_attn_gpu` extension-module binding, and all 9 `bert_padding` symbols.
6. Landed during the refresh: packed wrappers, dense `return_attn_probs`+`dropout_p=0` contract, empty-input short-circuit, head-dim bucket padding (`_round_head_dim`), `flash_attn_gpu`→extension-module binding.
7. Missing (all non-required): 4 aux packed classes, 8 `_flash_attn_*` entries (4 ops + 4 fakes), 4 `_wrapped_*` handles, 2 `_torch_*` assigns, 2 `noop_*` fallbacks, `USE_TRITON_ROCM`, `_get_block_size_n`, `round_multiple`.
8. Accepted divergence: `deterministic=True` warns and is disabled (V100 lacks deterministic backward).
9. Capability gaps (hardware/kernel): fp16-only (no bf16), Volta-only arch check, kvcache `num_splits ≤ 1` + softcap/window/ALiBi restrictions, >256 head dim raises `ValueError` vs upstream `RuntimeError`.
10. Legacy names: shim `flash_attn_gpu` is the extension module (upstream parity); `flash_attn_varlen_gpu`/`flash_attn_with_kvcache_gpu` (×2 modules) and the top-level `flash_attn_gpu` function alias remain fork extras.
11. Raw extension parity good: fork `.so` answers to both `flash_attn_v100_cuda` and `flash_attn_2_cuda` and exposes the same 5 functions (nit: `varlen_fwd` extra 22nd `num_splits` arg without default).
12. Checker improvements during the refresh: relative-import level preserved during parsing; conditional-import equivalence for `flash_attn_gpu`.
13. Extended scope: pure-python-portable = `layers/rotary` (torch fallback), `layers/patch_embed`, `ops/activations`, `utils/{benchmark,distributed,torch}`; skip triton/compiled-ext/model-zoo/cute areas.
14. Recommended order: varlen rename (finish P1) → aux classes + `_flash_attn_*` (P2) → helpers/optional modules (P3).
15. `03_api_spec.json` encodes every symbol (params/defaults/required flag + per-symbol recommendation), ready for an `inspect.signature` test.
