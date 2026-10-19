# VERIFICATION — V100 FlashAttention-2 as a drop-in for upstream flash-attn 2.8.3

Branch: `v100/cuda126-dropin` · date: 2026-10-20 · all evidence is reproducible from
this repository (no GPU required for anything claimed here; no performance claim is
made).

## 0. TL;DR

| Claim | Result |
|-------|--------|
| Full CUDA 12.6 build (sm_70, torch 2.6.0+cu126) | **PASS** — 6/6 TUs, wheel built and installed via `uv` (`BUILD_EXIT=0`) |
| `import flash_attn` + upstream API surface | **PASS** — `__version__ == "2.8.3"`, 25 required symbols (17 exact / 0 mismatch / 0 missing / 8 present-nonfunc) |
| Runtime API parity vs upstream 2.8.3 source | **PASS** — `RESULT PARITY PASS (failed=0, warn=0, known_gaps=[])`; official-source signature comparison 23/23 |
| `import flash_attn_2_cuda` (upstream import name) | **PASS** — alias module, same module object as `flash_attn_v100_cuda`, works in either import order |
| `importlib.metadata.version("flash-attn")` | **PASS** — `2.8.3` (`top_level.txt == flash_attn`) |
| GPU-free acceptance suite | **PASS** — `tests/run_all_cpu.py`: **7/7 checks** (parity mode: *runtime, compiled extension importable*) |
| Static SASS legality | **PASS** — 5 embedded sm_70 cubins, 0 post-Volta opcodes, 0 embedded PTX, 657,824 × `HMMA.884` (Volta tensor cores actually used) |
| ptxas resource report | **PASS** — per-kernel registers/smem/spills captured (see §5; `tests/artifacts/ptxas/ptxas_stats.json`) |
| Adversarial iteration | **PASS** — red round 1 (`research/07`, `08`), blue round 1 (`09`), independent verifier (`10`), red round 2 (`11`); all S0/S1 items fixed, residuals documented in §7 |

## 1. Requirement → evidence map

| # | Requirement | Concrete evidence | Status |
|---|-------------|-------------------|--------|
| 1 | CUDA 12.6 compile success (conda nvcc allowed) | `tests/artifacts/build_cuda126/build.log`, `build_env.txt` (`nvcc 12.6.85`, `torch 2.6.0+cu126`), `BUILD_EXIT=0` | ✅ |
| 2 | conda env `v100`, packages via `uv`, torch pinned 2.6.0 | env at `/mnt/envs/v100` (persistent); `torch.__version__ == 2.6.0+cu126`, `torch.version.cuda == 12.6`; installs done with `uv pip` | ✅ |
| 3 | API-level seamless replacement of A100 FA2 | §4 (parity spec, official-source comparison, import surface, metadata, alias, `torch.ops`) | ✅ (API level; scope exclusions in §7) |
| 4 | Multiple subagents, ≥2 red/blue rounds, traceable find→fix→recheck | §6 (`research/07`–`11`; 12+ subagent sessions: recon, six-way research, 3 blue, verifier, red-2) | ✅ |
| 5 | GPU-free evidence chain (build log, import/API smoke, ptxas, SASS, before/after) | §3–§5, `tests/artifacts/**` | ✅ |

## 2. Environment (persistent, not under `/tmp`)

```
/mnt/envs/v100            conda env (python 3.12.15)
  bin/nvcc                CUDA 12.6.85 (cuda-nvcc conda package)
  bin/x86_64-conda-linux-gnu-gcc   gcc 12.4 (nvcc 12.6 rejects newer hosts)
  lib/python3.12/site-packages/torch 2.6.0+cu126
/mnt/.cache/uv            uv cache (persistent)
/mnt/envs/cuda-tools      cuobjdump/nvdisasm **12.6** (CUDA 13+ nvdisasm dropped sm_70)
/mnt/envs/upstream        upstream flash-attn-2.8.3 sdist (sha256 1e71dd64…70d) for
                          official-source comparisons
```

The environment, caches, tools and the upstream reference all live on persistent
storage; `/tmp` is only used for disposable logs.

## 3. Build and install

```bash
bash scripts/build_with_evidence.sh     # wraps scripts/build_cuda126.sh, archives evidence
```

* Toolchain: `CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux`, `CC/CXX` = conda gcc 12.4,
  `PATH` includes `$CONDA_PREFIX/nvvm/bin` (conda nvcc resolves `cicc`/`cudafe++`
  through `PATH` when invoked via the `targets/.../bin/nvcc` symlink).
* Full clean rebuild: **6m46s**, 6/6 translation units, `-gencode arch=compute_70,code=sm_70`
  (`tests/artifacts/build_cuda126/build.log`, `build_env.txt`).
* Wheel install: `uv pip install --no-build-isolation --no-deps .` → `Installed 1 package`
  + post-install upstream metadata (`OK: flash-attn 2.8.3 top_level='flash_attn\n'`).
* Installed artifacts (site-packages): `flash_attn/`, `flash_attn_v100/`,
  `flash_attn_2_cuda.py`, `flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so`,
  `flash_attn_v100-26.6.dist-info/`, `flash_attn-2.8.3.dist-info/`.

Repair history (kept in `build.log.*`, every failure → fix documented):
1. ambient `CONDA_PREFIX` pointed at base miniforge → env forced explicitly;
2. conda nvcc could not find `cicc`/`cudafe++` through the `targets` symlink → `PATH` fix
   in `scripts/build_cuda126.sh`;
3. the wheel carried two `.dist-info` trees (blue F-5 approach) → PEP 427 forbids it
   (uv: “Multiple .dist-info directories found”); metadata is now materialised
   post-install by `flash_attn_v100._upstream_metadata` (§4);
4. a stale `flash_attn_2_cuda.so` from an older build tree was packaged into the wheel →
   removed by `BuildAttention.build_extensions()` before packaging (the alias module
   replaces it).

## 4. API drop-in evidence

* **Parity spec** (`research/03_api_spec.json`, generated from upstream 2.8.3 sources):
  25 required symbols — **exact 17 / mismatch 0 / missing 0 / present-nonfunc 8**.
* **Runtime parity check**: `python tests/parity_check.py` →
  `RESULT PARITY PASS (failed=0, warn=0, known_gaps=[])`
  (`tests/artifacts/cpu_suite/parity_runtime.log`; run_all_cpu detects mode
  *runtime (compiled extension importable)*).
* **Official-source comparison**: `FLASH_ATTENTION_OFFICIAL_SRC=/mnt/envs/upstream/.../flash_attn_interface.py
  python tests/test_api_parity.py` → **23/23** (signatures, defaults, names).
* **`flash_attn.bert_padding` official parity**:
  `FLASH_ATTENTION_OFFICIAL_BERT_PADDING=/mnt/envs/upstream/.../bert_padding.py
  python tests/test_bert_padding.py` → **22/22** (exact official outputs + gradient parity).
* **Wrapper/interface semantics** (CPU mock of the extension): `test_wrapper_cpu` **29/29**,
  `test_reference_cpu` **29/29**, `test_interface_behavior` **113/113**.
* **Installed-extension smoke**: `test_extension_smoke` **11/11**, including
  `torch.ops.flash_attn_v100.{fwd,bwd,varlen_fwd,varlen_bwd,fwd_kvcache}` registration.
* **Import names** (verified from a neutral cwd, both orders):

  ```text
  import flash_attn                     -> 2.8.3   (site-packages/flash_attn/__init__.py)
  import flash_attn_v100_cuda as a
  import flash_attn_2_cuda as b         -> a is b True (same module object)
  importlib.metadata.version("flash-attn") -> 2.8.3, top_level.txt == "flash_attn"
  ops: fwd, bwd, varlen_fwd, varlen_bwd, fwd_kvcache
  ```

* **Why not a `flash_attn_2_cuda.so` symlink**: shipping the same `.so` under two module
  names makes any process that imports both names abort — the second `dlopen` re-runs the
  static initialiser and trips
  `Only a single TORCH_LIBRARY can be used to register the namespace flash_attn_v100`.
  Downstream code (vLLM) imports both names, so the symlink was a real drop-in blocker.
  `flash_attn_2_cuda.py` provides the same import name while loading the library once.

## 5. GPU-free static evidence

**SASS legality** (`scripts/verify_sass.sh`, tools pinned to CUDA 12.6):

```
5 embedded cubins, all sm_70; embedded PTX: none
forbidden post-Volta opcodes (HMMA.1688/HMMA.16816/LDSM/LDGSTS/WGMMA/…/CP.ASYNC): 0
HMMA.884 occurrences: 657824        FFMA: 108016
RESULT STATIC_SASS PASS
```

Evidence: `tests/artifacts/sass/verify.log`, `sass_summary.txt` (the raw dump was 1.3 GB
and is reproducible). `tests/artifacts/sass/so_sha256.txt` records the binary hash.

**ptxas resource report** (`scripts/capture_ptxas_stats.sh` replays the exact nvcc
commands with `-Xptxas -v`; torch swallows successful ninja output, hence the replay):

| Metric | Value |
|--------|-------|
| Entry functions | **1440**, all `sm_70` |
| Registers | **64** for 1434 kernels, 63 for 6 (cap enforced by `__launch_bounds__(…, 2)`) |
| Stack frame | up to 136 B (worst instantiation) |
| Spills | 81,908 B stores + 142,268 B loads across the binary; **1379/1440** kernels spill (≤ 148 B stores + 308 B loads per thread) |
| Static shared memory | 0 B (kernels take *dynamic* shared memory, configured at launch via `cudaFuncSetAttribute`, ≤ 96 KB, host-asserted) |

Representative instantiations (full table: `tests/artifacts/ptxas/ptxas_stats.json`):

| kernel (instantiations) | regs | stack | spill st+ld |
|-------------------------|------|-------|-------------|
| `flash_attention_forward_kernel<64,…>` (32) | 64 | 8 | 8 + 24 |
| `flash_attention_forward_kernel<128,…>` (32) | 64 | 24 | 24 + 52 |
| `flash_attention_forward_varlen_kernel<64,…>` (64) | 64 | 48 | 40 + 80 |
| `flash_attention_backward_kernel<128,…>` (32) | 64 | 24 | 28 + 40 |
| `flash_attention_kvcache_kernel<128,…>` (128) | 64 | 80 | 80 + 140 |

The 64-register cap comes from `__launch_bounds__(threads, 2)`; the resulting spills are
expected and are recorded here so a V100 run can correlate them with performance. No
performance claim is made.

Parse with `python tests/collect_ptxas_stats.py tests/artifacts/ptxas/ptxas_all.log
--json tests/artifacts/ptxas/ptxas_stats.json --expect-arch 70`.

## 6. Adversarial iteration (find → fix → recheck)

| Round | Artifact | Result |
|-------|----------|--------|
| Red 1 (API/build) | `research/07_redteam_round1.md` | F-1…F-12 (CUDA 12.6 build was falsified at the audited commit; F-2 empty-backward crash; F-3 `cache_seqlens=None`; F-4 torch pin; F-5 metadata; F-6 `S_dmask`; F-7 raw arity; F-8 empty-KV sentinel; F-9 coverage taxonomy; F-10 subpackages; F-11/F-12) |
| Red 1 (kernel) | `research/08_kernel_redteam.md` | P0-1 dropout RNG stride, P1-1 missing `dsoftmax_sum`, P1-2 all-masked row, P1-3 cross-page paged loads, G-1 Philox mapping, G-2 dmask alignment, G-3 `seqused_k`; P1-4 refuted, P2-4 deferred (MMA_884 variant) |
| Blue 1 | `research/09_blue_round1_fixes.md` | all fixes applied; orchestrator audit + post-build corrections (§5 of that report) |
| Independent verifier | `research/10_blue_verify.md` | D1–D8: G-3 dead guard, mock layout/stub shape/`NameError`, in-kernel empty-KV sentinel, F-5/F-7 runtime gaps, P1-3 residual |
| Red 2 | `research/11_redteam_round2.md` | R1–R11 attacked the blue fixes; disposition in §(e) of that report — R1–R5/R7–R10 fixed, R6/R11 accepted, plus the double-load blocker found during installation |

Key correctness changes landed (details in the reports): `softmax.h` (P1-2/G-1/G-2),
`backward*.cu` (`dsoftmax_sum`, varlen dropout RNG stride), `forward*.cu` (empty-KV sentinel
`+inf`, two-segment paged loading, host block-table coverage `TORCH_CHECK`),
`api.cpp` (raw `varlen_fwd` annotations + defaults, upstream window arg names),
Python layer (empty-input backward guards, dense `S_dmask` padding, `cache_seqlens=None`
semantics, `head_dim` rounding, kvcache asserts), packaging (post-install upstream
metadata, alias module, stale-`.so` removal).

## 7. Known limitations / residual risk

1. **No GPU on this machine**: numerics and performance are *not* validated here. Run
   `python tests/gpu_verify.py` on a V100. No performance claim is made.
2. **Upstream subpackages** `flash_attn.ops`, `flash_attn.layers`, `flash_attn.modules`,
   `flash_attn.utils`, `flash_attn.models` are not shipped (documented scope; see
   `research/06_prior_art.md`). `flash_attn.bert_padding` *is* shipped and official-parity
   verified.
3. **varlen `S_dmask` shape** stays `(T_Q, H, max_k)` (dense matches upstream's padded
   4-D shape). Documented deviation.
4. **`seqused_k` + gradients**: the varlen backward has no `seqused_k` (same as upstream);
   the Python guard cannot fire because the upstream-compatible wrapper does not expose
   `seqused_k` — it is documentation-only. Raw extension users passing `seqused_k` with
   gradients get upstream's behaviour (forward uses it, backward ignores it).
5. **P2-4 (`MMA_884` build option)**: the optional native-MMA path's mask-lane mapping is
   unverified without a GPU; the default build (WMMA) is what the SASS check covers.
6. **Paged varlen**: an under-sized block table is now rejected on the host; malformed
   tables can no longer silently attend to stale shared memory (kvcache sizes the table
   from `seqlen_k` by construction).

## 8. Reproduce (clean machine, persistent paths)

```bash
conda create -n v100 python=3.12 -y && conda activate v100
conda install -n v100 -y -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6
conda install -n v100 -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12
uv pip install --python "$CONDA_PREFIX/bin/python" \
    torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$CONDA_PREFIX/bin/python" einops ninja packaging psutil

bash scripts/build_with_evidence.sh          # build + install + evidence capture
FLASH_ATTENTION_OFFICIAL_SRC=/mnt/envs/upstream/flash_attn-2.8.3/flash_attn/flash_attn_interface.py \
FLASH_ATTENTION_OFFICIAL_BERT_PADDING=/mnt/envs/upstream/flash_attn-2.8.3/flash_attn/bert_padding.py \
    python tests/run_all_cpu.py              # 7/7 PASS

conda create -y -p /mnt/envs/cuda-tools -c nvidia -c conda-forge \
    cuda-nvdisasm=12.6 cuda-cuobjdump=12.6
bash scripts/verify_sass.sh                  # RESULT STATIC_SASS PASS
bash scripts/capture_ptxas_stats.sh          # ptxas -v stats
```

## 9. Evidence index

```
tests/artifacts/
  build_cuda126/{build.log, build_env.txt, build_driver_success.log, build.log.*}
  build_cuda126_ptxas/build.log            flags-pure verbose build (same codegen)
  compile_checks/{summary.txt, check_*.log} single-TU compile checks (6/6 EXIT=0)
  cpu_suite/*.log                          every suite, incl. official-source runs
  ptxas/{ptxas_all.log, ptxas_stats.json, cmd_*.sh}
  sass/{verify.log, verify_final.log, sass_summary.txt, so_sha256.txt}
research/01…06                            environment / architecture / API / build /
                                          verification-plan / prior-art research
research/07…11                            red1 / kernel-red1 / blue1 / verifier / red2
scripts/{build_cuda126.sh, build_with_evidence.sh, verify_sass.sh,
         capture_ptxas_stats.sh}
```
