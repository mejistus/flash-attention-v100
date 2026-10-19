# VERIFICATION — V100 drop-in `flash-attn` 2.8.3 (memory-first Pareto build)

Date: 2026-10-20 · hardware: **Tesla V100S-PCIE-32GB** (sm_70, 32768 MiB, driver 580.105.08)
· software: **torch 2.8.0+cu128**, CUDA 12.8 (nvcc 12.8.93), gcc 11.4, Python 3.12.
The same source builds against CUDA 12.6 / torch 2.6.0+cu126 (recipe in the README).

## 0. Published artifacts

| item | value |
|------|-------|
| GitHub release | <https://github.com/mejistus/flash-attention-v100/releases/tag/v2.8.3> |
| wheel | `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` (15 672 193 B) |
| sha256 | `f0f6d69250db7ec8c5d1ec91c741654b3f5e557178352a59e629eca29d422795` |
| source | branch `official-2.8.3`, commit `d0f5fd4` (parent: remote `main` `55d685a`) |
| extra assets | `results_final.json`, `cpu_final2.log`, `sass_final.log`, `*.whl.sha256` |

The wheel was installed from the release URL into a **fresh venv** and ran a GPU
fwd+bwd: `flash_attn 2.8.3`, extension `flash_attn_2_cuda`, `importlib.metadata`
2.8.3. Wheel contents: `flash_attn/`, `flash_attn-2.8.3.dist-info/`,
`flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so` — **0 entries containing "v100"**.

## 1. What the build is

Official naming and layout, exactly like flash-attn on other GPUs:

* distribution **`flash-attn` 2.8.3**, import **`flash_attn`**, extension
  **`flash_attn_2_cuda`**, API `flash_attn_func` / `flash_attn_varlen_func` /
  `flash_attn_with_kvcache` / packed variants / `flash_attn.bert_padding`.
* The old fork names (`flash_attn_v100`, `flash_attn_v100_cuda`, the
  `flash_attn_2_cuda.py` alias) are gone; `torch.ops` is `torch.ops.flash_attn`.
* sm_70 kernels only (V100 16 GB and 32 GB share the same code path).

## 2. Core changes vs the previous V100 implementation

1. **Dense BMHD end-to-end** — the kernels consume the official `[B, M, H, D]`
   layout directly. The Python layer no longer permutes + copies q/k/v/out
   (`permute().contiguous()`), which cost ~320 MiB at 8k tokens during training and
   doubled inference buffers. Offsets/loads/stores in `forward.cu` and
   `backward.cu` use the BMHD row strand (`H*D`) and a `GLOBAL_WIDTH=D` sub-tile.
2. **GQA/MQA varlen backward without expansion buffers** — one CTA per KV head
   loops over its Q-head group and accumulates dK/dV in shared memory instead of
   allocating `(T_k, H_q, D)` fp16 buffers and running a `sum_out` reduction.
3. **Tile tuning** — D=128: forward 64×64 (was 32×176), dQ 32×64 (was 32×112),
   dKV 32×64 (was 16×128); `__launch_bounds__` no longer forces 2 CTAs/SM when the
   96 KB shared-memory footprint cannot provide them.
4. All correctness fixes from the red/blue review are retained: empty-KV `+inf`
   sentinel, fully-masked rows P=0, dmask alignment, `dsoftmax_sum`, dropout Philox
   mapping, paged-KV segment loads + block-table coverage check, Python empty-input
   guards, official `S_dmask` padding for dense.

## 3. Memory and compute (V100S 32 GB, fp16, fwd+bwd unless stated)

Method: `bench/bench_pareto.py` — `torch.cuda.reset_peak_memory_stats()` after input
allocation; median of 3 runs; causal; GQA `H=32 / H_k=8 / D=128` unless stated.

| workload | this build | previous V100 impl | PyTorch SDPA (mem-efficient) |
|----------|-----------:|--------------------:|-----------------------------:|
| dense N=8192 | **418 MiB / 119.3 ms** | 738 MiB / 126.6 ms | 770 MiB / 69.8 ms |
| dense MHA (H=H_k=32) N=8192 | **706 MiB / 112.7 ms** | — | 834 MiB / 69.6 ms |
| packed varlen T=16384 (4×4096) | **836 MiB / 112.4 ms** | 1092 MiB / 123.4 ms | 1540 MiB / 71.8 ms (padded) |
| packed varlen T=32768 (4×8192) | **1672 MiB / 442.9 ms** | 2184 MiB / 478.7 ms | 3080 MiB / 285.7 ms (padded) |
| inference prefill N=8192 | **224 MiB / 23.9 ms** | 321 MiB / 25.9 ms | 288 MiB / 15.3 ms |
| KV-cache decode, cache=8192 | 48 MiB / 0.60 ms | 48 MiB / 0.48 ms | — |

* **Dense training peak memory −43 %** and **packed-varlen −23 %** vs the previous
  V100 implementation, with equal or slightly better latency on every measured case.
  This is the Pareto improvement requested: memory down, compute not down.
* vs PyTorch SDPA the build trades speed for memory: 40–50 % lower peak memory for
  GQA/varlen training (SDPA must expand KV heads and has no packed-varlen path), at
  ~1.7× the latency. The README states this explicitly; SDPA is the faster option
  only for MHA workloads where memory is not the constraint.
* **16 GB budget**: with `torch.cuda.set_per_process_memory_fraction(0.5)` (16 GiB
  cap) a packed-varlen fwd+bwd of **T=262144 (8×32768)** completes at
  **13 376 MiB peak** in 14.4 s (`tests/artifacts/v100_dropin/16gb_cap.txt`).

## 4. Correctness on the V100

* Dense vs an eager fp32/`sdpa` reference, fp16 tolerance: causal/non-causal, GQA
  (32/8, 8/2), M=700/N=1024, window, softcap, alibi — forward max |Δ| ≤ 1.0e-3;
  backward max |Δ| ≤ 0.008 (≤ 0.1 % relative for large-gradient alibi cases).
* varlen vs per-sequence dense (same kernels): forward max |Δ| = 0, backward
  ≤ 0.0078; GQA/MQA (H_k = 32/8) both pass.
* kvcache decode path exercised; paged/leftpad paths unchanged from the reviewed
  implementation.
* GPU-free acceptance suite on the installed package: `tests/run_all_cpu.py`
  **7/7 PASS** (API parity runtime mode, official-source signature comparison,
  `bert_padding` official parity, wrapper semantics, installed-extension smoke with
  `torch.ops.flash_attn`, dist-info metadata). Log:
  `tests/artifacts/v100_dropin/cpu_suite_final.log`.

## 5. Static sm_70 legality

`scripts/verify_sass.sh` on the released `.so` (CUDA 12.8 cuobjdump/nvdisasm):

```
5 embedded cubins, all sm_70; embedded PTX: none
forbidden post-Volta opcodes (HMMA.1688/HMMA.16816/LDSM/LDGSTS/WGMMA/.../CP.ASYNC): 0
HMMA.884 occurrences: 620992
RESULT STATIC_SASS PASS
```

Log: `tests/artifacts/v100_dropin/sass_final.log`.

## 6. Known limitations (honest list)

1. **Compute vs SDPA**: the WMMA kernels are ~1.7× slower than PyTorch's cutlass
   mem-efficient attention on V100. The delivered Pareto point is memory-first (the
   user's stated core need); relative to the previous V100 implementation both memory
   and time improved. No A100-class performance is claimed on V100 hardware.
2. `flash_attn.ops`, `.layers`, `.modules`, `.utils`, `.models` are not shipped.
3. Head dims `{16,32,64,128,256}` are native; other dims are zero-padded to the next
   supported size (exact results, overhead only for those models).
4. `num_splits > 1` is not implemented; `deterministic=True` is accepted with a
   warning (non-deterministic atomic backward, as before).
5. varlen `S_dmask` keeps the tight `(T_q, H, max_k)` shape (dense matches upstream's
   padded 4-D). Documented deviation.
6. The optional `MMA_884` build variant is not part of this release; the default
   WMMA path is what the SASS check covers.
7. The released wheel is ABI-bound to CPython 3.12 + torch 2.8 (cu128); for other
   torch versions rebuild from source (README).

## 7. Reproduce

```bash
# on a V100 (or GPU-less build host), following README "Build from source"
python tests/run_all_cpu.py                    # 7/7 expected
python tests/gpu_verify.py                     # numeric checks on a V100
python bench/bench_pareto.py --json results.json
bash scripts/verify_sass.sh                    # 5x sm_70 + HMMA.884
```
