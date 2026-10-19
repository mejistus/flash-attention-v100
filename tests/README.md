# CPU-runnable test suite

These tests verify everything that can be checked **without a Volta GPU**:

| file | purpose |
|------|---------|
| `test_api_parity.py` | Signature/AST parity of all 7 official FlashAttention-2 entry points (pinned 2.8.x signatures + optional live diff against an official checkout). |
| `test_interface_behavior.py` | Behavioral tests of the Python interface layer against a recording stub backend: permutes, head-dim padding (`96 -> 128`, ...), softmax-scale defaults from the *original* head size, `return_attn_probs` semantics (empty `S_dmask` when `dropout_p == 0`), packed QKV/KV wrappers, varlen casting, kvcache argument marshalling, autograd wiring (incl. gradient arity) and empty-input early returns. |
| `test_bert_padding.py` | Numerical + gradient parity of `flash_attn.bert_padding` against the official implementation (and self-consistency of pad/unpad round trips). |
| `test_wrapper_cpu.py` | Wrapper-semantics tests against a computational mock (`mock_cuda`) that behaves like the compiled extension but computes with CPU torch: layout permutes, head-dim padding, GQA, window/softcap/alibi plumbing, autograd flow, varlen packing, kvcache in-place update / `cache_seqlens` conversion, numerics vs an independent `softmax`/`logsumexp` oracle. |
| `mock_cuda.py` | CPU mock of the compiled extension (reference math) used by `test_wrapper_cpu.py` and `parity_check.py --mock`. |
| `parity_check.py` | API drop-in parity harness — runtime / `--ast` (torch-less) / `--mock` modes: `__version__` stamps, dist-info metadata, import paths (`flash_attn`, `flash_attn_2_cuda`, ...), signatures vs `research/03_api_spec.json`, known-gap accounting. |
| `test_extension_smoke.py` | Post-build only: compiled `flash_attn_2_cuda` exposes the 5 ops, TorchBind registrations present. Skips cleanly pre-build. |
| `gpu_verify.py` | GPU-day end-to-end correctness suite (needs a real sm_70 GPU, skips otherwise): dense / varlen / kvcache forward, backward gradients, GQA, causal, sliding window, softcap, alibi, head-dim padding, lse. |
| `collect_ptxas_stats.py` | Parses nvcc/ptxas `-v` build logs into per-kernel register / shared-memory statistics (build evidence, no GPU needed). |

Runners:

* `run_all_cpu.py` — runs every `test_*.py` in this directory in its own process (quick CPU gate).
* `run_all.py` — full GPU-free acceptance runner: build evidence + artifacts + API parity +
  CPU mock tests + static SASS/ISA checks → machine-readable report under `tests/artifacts/`.

```bash
/mnt/envs/v100/bin/python tests/run_all_cpu.py     # quick CPU gate
/mnt/envs/v100/bin/python tests/run_all.py --repo .  # full acceptance (after a build)
```

Static / build helpers (repo `scripts/`):

* `scripts/verify_sass.sh` — cuobjdump/nvdisasm static check that the built `.so` embeds
  sm_70-only cubins and no post-Volta ISA opcodes (falls back to an arch-tag scan without
  `cuda-cuobjdump`/`cuda-nvdisasm`).
* `scripts/build_matrix.sh` — builds `default` / `MMA_NATIVE` / `MMA_884` scratch copies and
  collects per-mode ptxas + SASS evidence.
* `scripts/build_cuda126.sh` — pinned CUDA 12.6 + conda-gcc toolchain build recipe.

Useful environment variables:

* `FLASH_ATTENTION_OFFICIAL_SRC` — path to an official `flash_attn_interface.py` for the live signature diff.
* `FLASH_ATTENTION_OFFICIAL_BERT_PADDING` — path to the official `bert_padding.py` for numerical comparisons.

GPU-dependent end-to-end tests live in the repository root (`test.py`) and require a physical
V100 (sm_70); they are intentionally not part of this suite.
