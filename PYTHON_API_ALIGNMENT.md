# Python API parity with official flash-attn 2.8.3

This V100 build keeps the sm_70 CUDA kernels from this repository but makes the
**pure-Python layer byte-identical to upstream flash-attn 2.8.3** (apart from two
private adapter lines in `flash_attn/__init__.py`). Existing code written for the
official package can use this wheel unchanged.

## What is upstream-verbatim

* `flash_attn/flash_attn_interface.py` — all public functions/classes, packed and
  varlen entry points, `return_attn_probs`, and the `torch.library.custom_op`
  registrations (`_flash_attn_forward`, `_flash_attn_varlen_forward`,
  `_flash_attn_backward`, `_flash_attn_varlen_backward`).
* `flash_attn/bert_padding.py`
* `flash_attn/__init__.py` (official exports; two private adapter lines added)
* Optional upstream Python packages: `flash_attn/modules`, `layers`, `ops`
  (including Triton kernels), `models`, `utils`, `losses`, plus the legacy
  `flash_attn_triton.py` / `flash_attn_triton_og.py`.

Excluded because they are not applicable to V100 or need unbuilt native ops:
`flash_attn/cute` (FlashAttention-3 / Hopper+), `flash_attn_triton_amd` (AMD
ROCm), and the blocksparse files.

## The only V100-specific adapter

`flash_attn/_v100_compat.py` is private and installs a proxy for the compiled
`flash_attn_2_cuda` module. The official Python files call the same extension
functions with the same signatures; the adapter only normalizes implementation
differences:

* unsupported head dimensions are zero-padded to 16/32/64/128/256 and sliced back
  (zero dimensions contribute zero to QK^T and P@V; the softmax scale still comes
  from the original head dimension);
* `q/k/v` views passed by the official packed entry points are made contiguous;
* V100's tight dropout-mask / `softmax_d` buffers are padded to upstream's
  128-padded shapes; varlen max sequence lengths are rounded for mask alignment.

No public names, signatures, defaults, or return layouts are changed.

## Verified integrations

* `diffusers` 0.41: no "flash_attn failed to import" fallback; its FLASH backend
  sees `_wrapped_flash_attn_forward/backward`.
* `transformers` 5.20: a small Llama model with
  `attn_implementation="flash_attention_2"` trains (loss/backward).
* `torch.compile(..., fullgraph=True)` around `flash_attn_func`.
* Dense, varlen, qkv/kv packed, varlen packed and KV-cache; head dims
  16/32/64/80/96/128/192; GQA; dropout + `return_attn_probs`; backward.
* `flash_attn.modules.mha.MHA` and `modules.block.Block` instantiate/run;
  Triton `ops.triton.layer_norm` and `layers.rotary.RotaryEmbedding` run on V100;
  `losses` imports; the reference `models` modules import when their optional
  dependencies (e.g. `sentencepiece`) are installed.

The repository's CPU acceptance suite (stubbed CUDA backend, no GPU) is fully
green and the API parity checker reports `RESULT PARITY PASS`:

```bash
python tests/run_all_cpu.py          # 8/8 PASS
python tests/parity_check.py --repo . --allow-known-gaps   # PARITY PASS
```

The three prebuilt wheels were installed and tested under
CUDA 12.4/torch 2.5.1, CUDA 12.6/torch 2.6.0 and CUDA 12.8/torch 2.8.0 on a
V100S-32GB.

## Known limitations

* **Non-causal ALiBi** still follows the V100 kernel's historical formula
  (`|i-j|`) rather than upstream's `|i + seqlen_k - seqlen_q - j|`; causal ALiBi
  matches upstream.  This is a kernel-level difference, not a Python one.
* `deterministic=True` backward is not implemented by the V100 kernel and raises a
  clear `RuntimeError` (the forward and non-deterministic backward are unchanged).
* Paged KV-cache **append** with a head dimension that had to be padded (e.g. 96)
  raises a clear `NotImplementedError`; paged attention without append works, and
  D=128 paged append works.
* `cache_batch_idx` incurs a cache-row gather/copy per call for correctness.
* Empty-input semantics match upstream where the compiled kernel can run: `B == 0`
  is rejected by the extension like upstream, empty varlen cases are forwarded,
  and an empty KV sequence (`N == 0`) returns upstream's zeros/`+inf` with a
  zero-gradient backward.  `M == 0` (an empty **query** sequence) is rejected with
  a clear `RuntimeError` because the V100 kernel cannot launch its backward and
  PyTorch creates the autograd node before the private adapter runs.
* `fused_dense_lib` and `dropout_layer_norm` are **separate, optional** native
  extensions in upstream 2.8.3: the root `setup.py` builds only
  `flash_attn_2_cuda`, and the official default wheel likewise does not contain
  these two.  Consequently `ops.fused_dense`, `ops.layer_norm`, `ops.rms_norm`,
  `ops.triton.mlp`, `ParallelMHA` and `FusedMLP` are unavailable unless the user
  separately builds the standalone `csrc/fused_dense_lib` / `csrc/layer_norm`
  extensions (their setup scripts do support sm_70).  Standard `MHA`/`Block`,
  Triton LayerNorm/RMSNorm, the model reference implementations and the
  cross-entropy loss work without them.
* FA3/CuTe, AMD Triton and blocksparse are not included.
* The public API is identical; the compiled V100 kernel is not the upstream
  sm80+ kernel and keeps its existing memory/performance characteristics.
