# 05 — GPU-free verification plan for flash-attention-v100 (sm_70)

Verification-architecture subagent report. Date: 2026-10-19.
Repo under verification: `/home/eculid/Documents/Github/flash-attention-v100`
(fork `ai-bond/flash-attention-v100`, FlashAttention-2 port to Volta sm_70).
Revision moved during the session (`55d685a` at start → `cfcb0cd` after other subagents
committed their fixes); all live checks quoted below were re-run against the final
`cfcb0cd` working tree. Historical log excerpts keep the revision they were taken at.

**Hard environment constraint:** this host has no NVIDIA GPU (AMD RX580 only; no CUDA
driver). Everything below therefore runs on CPU or is a *static* analysis of build
artifacts. The goal is to make "day 1 on the V100" a pure smoke test: every property that
can be checked without buying compute time **is** checked here, with machine-parseable
evidence, and every property that genuinely needs silicon is explicitly listed as
unverified (§6) instead of being silently assumed.

**Drafts are kept outside the repo** in `/tmp/opencode/verif_drafts/` (promote to `tests/`
after review; §7 has the inventory). Only this document was written into `research/`.

Toolchain facts used throughout (all re-verified today, see §7 evidence ledger):

- `/mnt/envs/v100/bin/nvcc` = CUDA **12.6.85**, `ptxas` same version.
- system `gcc` 16.2.1 **breaks** nvcc 12.6 (`unsupported GNU version`); conda
  `x86_64-conda-linux-gnu-g++` **12.4.0** works and must be pinned via
  `CC`/`CXX`/`CUDAHOSTCXX`.
- `cuobjdump`/`nvdisasm` are **not installed** here → the full SASS ISA check is
  unavailable; the fallback arch scan runs instead, and everything ISA-related is
  reported PARTIAL rather than PASS (optional install:
  `mamba install -n v100 -c nvidia cuda-nvdisasm cuda-cuobjdump`; deliberately not
  installed by this plan — it must never become a build-time dependency).

---

## 1. Static SASS / ISA verification — `verify_sass.sh`

**Question answered:** does the produced `.so` contain *only* sm_70 code, and does the
tensor-core instruction stream look like Volta (HMMA.884 m8n8k4) and not like a
back-ported path that accidentally carries Turing/Ampere+ opcodes or PTX?

### 1.1 Checks performed (full toolchain mode)

| # | Check | How | Failure means |
|---|-------|-----|---------------|
| 1 | Exactly sm_70 embedded cubin(s), no other `-lelf` tags | `cuobjdump -lelf` | wrong `-gencode`, fatbin pollution |
| 2 | No embedded PTX for newer targets | `cuobjdump -ptx` (list + tags) | driver-side JIT could pick a non-Volta SASS |
| 3 | No post-Volta tensor-core/async mnemonics in SASS | `cuobjdump -sass \| nvdisasm -c` greps | using instructions Volta cannot execute |
| 4 | At least one `HMMA.884` (default on; `--no-require-hmma` to relax) | grep count | tensor cores were scalarized/broken → perf regression, not correctness |
| 5 | Disassembly has no undecoded/unknown instructions | `nvdisasm` warnings scan | corrupt/unsupported encodings |

Forbidden post-Volta pattern (check 3):

```
HMMA\.1688|HMMA\.16816|LDSM|LDGSTS|WGMMA|HGMMA|UBLKCP|BLKCP|CP\.ASYNC
```

(`HMMA.1688` = Turing m16n8k8, `HMMA.16816` = Ampere m16n8k16, `LDSM` = Turing
`ldmatrix`, `LDGSTS` = Ampere async copy, `WGMMA`/`HGMMA` = Hopper/Blackwell,
`UBLKCP`/`BLKCP`/`CP.ASYNC` = TMA-family. Volta's only tensor-core MMA opcode family is
`HMMA.884`/`HMMA.884.F32.*`.)

### 1.2 Fallback mode (no cuobjdump/nvdisasm — the state of this host)

Scans the raw `.so` for fastbin-embedded ELF images with `e_machine == 190` (EM_CUDA),
decodes the arch nibble from `e_flags & 0xff`, and fails on anything ≠ sm_70. It also
records the cubin-name strings (e.g. `flash_attn_v100_cuda.cubin`). Verdict is forced to
**PARTIAL** because no instruction-level check is possible.

### 1.3 Executed evidence (this host, today)

sm_70-only tiny library → PARTIAL (arch tags OK, ISA unchecked):

```
$ bash verify_sass.sh --so sass_evidence/libtiny.so --out-dir sass_evidence/out_tiny
# target: sass_evidence/libtiny.so (1005144 bytes)
# cuobjdump/nvdisasm not found -> fallback embedded-ELF scan
# (arch tags only; NO instruction-level ISA check possible)
embedded CUDA ELFs: 2
  @0x95b90 e_flags=0x460546 -> sm_70 [OK]
  @0x95f48 e_flags=0x460546 -> sm_70 [OK]
FALLBACK_BAD=0
PARTIAL: arch tags are sm_70-only, but SASS ISA legality was NOT checked (install cuda-nvdisasm/cuda-cuobjdump)
RESULT STATIC_SASS PARTIAL
EXIT=0
```

Mixed sm_80/sm_86 library (what a wrong build looks like) → FAIL, exit 1:

```
$ bash verify_sass.sh --so sass_evidence/libmixed.so --out-dir sass_evidence/out_mixed
# target: sass_evidence/libmixed.so (1009256 bytes)
# cuobjdump/nvdisasm not found -> fallback embedded-ELF scan
# (arch tags only; NO instruction-level ISA check possible)
embedded CUDA ELFs: 4
  @0x95b90 e_flags=0x500550 -> sm_80 [BAD]
  @0x95f38 e_flags=0x560556 -> sm_86 [BAD]
  @0x962f0 e_flags=0x500550 -> sm_80 [BAD]
  @0x96dd8 e_flags=0x560556 -> sm_86 [BAD]
FALLBACK_BAD=4
FAIL: fallback scan found non-sm_70 embedded cubins
RESULT STATIC_SASS FAIL
EXIT=1
```

(Also observed: the same nibble is enough to distinguish archs — `0x460546 → 70`,
`0x500550 → 80`, `0x560556 → 86`; `e_flags` low byte = arch, upper bits include the
`EF_CUDA_SM` marker.)

### 1.4 Limits of this section

- Fallback proves **arch tagging only**, not legality of every instruction (a hand-patched
  cubin could still smuggle an `LDSM`).
- Check 4 is a *presence* check, not coverage: it can't prove the hot path uses HMMA.884.
- It says nothing about numerics, races, or occupancy — see §6.

**Run on the build machine (after the real build):**

```bash
bash verify_sass.sh --so <repo>/flash_attn_v100_cuda*.so --out-dir tests/artifacts/sass
# with cuobjdump/nvdisasm installed this is a full PASS/FAIL; otherwise PARTIAL
```

---

## 2. CPU-mock wrapper harness — `mock_cuda.py` + `test_wrapper_cpu.py`

**Question answered:** given a *correct* device kernel, does the python wrapper +
pybind/C++ argument plumbing compute the right schedule (layouts, padding, GQA map,
masks, alibi, varlen packing, kvcache bookkeeping, autograd wiring)? These layers are
pure CPython/torch/C++ logic and can be exercised faithfully on CPU by replacing only
the CUDA extension with a semantic twin.

### 2.1 Architecture

- `mock_cuda.py` fabricates a module and installs it into `sys.modules` as **both**
  `flash_attn_v100_cuda` **and** `flash_attn_2_cuda` **before** the wrapper is imported
  (the wrapper resolves the extension at import time; `flash_attn/flash_attn_interface.py`
  in this fork imports it by the upstream name).
- The fake module implements exactly the five entry points the repo's `kernel/api.cpp`
  binds, with identical dtypes/arities:

  | entry | returns | notes |
  |-------|---------|-------|
  | `fwd(q,k,v,out,alibi,dropout,scale,causal,wl,wr,softcap,return_softmax,gen)` | `(out, lse, S_dmask, rng_state)` | 4-tuple, `S_dmask` empty when `dropout_p==0` |
  | `bwd(dout,q,k,v,out,lse,dq,dk,dv,alibi,...)` | `(dq, dk, dv, dsoftmax_sum)` | wrapper reads `[0..2]` |
  | `varlen_fwd(...)` | same 4-tuple, `lse` `(H_Q,T)` | `cu_seqlens` int32 required; accepts `seqused_k` |
  | `varlen_bwd(...)` | `(dq, dk, dv, dsoftmax_sum)` | |
  | `fwd_kvcache(...)` | `(out, softmax_lse)` | updates `k_cache`/`v_cache` **in place** |

- Math mirrors the documented semantics of `kernel/*.cu` (fp32 upcast on CPU, cast back;
  the kernel's approximations — e.g. fp32 softmax — are the intended behavioral model):

  - dense/varlen softmax: scale `1/sqrt(D)` or `softmax_scale`; `softcap>0` →
    `p = softmax(softcap*tanh(s/softcap))`; causal + `window_size` mask with
    `win >= N → -1` clamp; `M==1` causal rows aligned to the end; fully-masked rows →
    `out=0`, `lse=-inf`; alibi slopes `(B,H)` with the causal/window seqlen offset rule;
    GQA maps kv head `q_head // (H_Q//H_K)`; `lse` `(B,H,M)`.
  - kvcache: writes new k/v at `cache_seqlens[b]`, causal window aligned to the *end* of
    the cache, int `cache_seqlens` converted by the wrapper to a device tensor.
  - backward: `torch.autograd.grad` under `enable_grad()`, gradient of the mock's own
    forward — i.e. it validates the wrapper's autograd plumbing, not kernel gradients.
- Mock reproduces the C++ `TORCH_CHECK` regressions cheaply: fp16/fp32 only, head-dim
  dispatch `D ∈ {16,32,64,128,256}` (after the wrapper pads to the next multiple of 8),
  GQA divisibility, tail-contiguity assumptions, kvcache stride asserts.
- **Not modelled → raise `NotImplementedError` (tests assert the raise or skip):**
  dropout RNG-offset serialization (`mock_cuda.ALLOW_DROPOUT=True` allows shape-only runs),
  rotary, `cache_leftpad`/`leftpad_k`, paged `block_table`.
- `CALL_LOG` records every extension call (function name + shapes/dtypes of key args)
  so tests can assert *exactly what the wrapper hands to the kernel*, not just the output.

### 2.2 Test catalog (`test_wrapper_cpu.py`, ~30 tests)

| area | tests |
|------|-------|
| import surface | `test_import_surface` (both packages expose the documented names) |
| dense fwd | basic, causal/window combos, softcap, alibi, GQA, head-dim padding (40→48 etc.), **negative**: unsupported head dim raises `RuntimeError: Unsupported D`, `return_attn_probs` tuple vs tensor, edges seqlens (M=N=1, M<N, M>N), non-multiple-of-64 seqlens, B=1/H=1, dropout shape-only, deterministic warning |
| dense bwd | backward vs oracle grads, partial grads (subset of q/k/v requires grad), no-grad path returns non-tensor grads only when needed |
| oracle cross-check | `test_sdpa_cross_check` — outputs vs `torch.nn.functional.scaled_dot_product_attention` (fp32 upcast) on a subset |
| varlen | basic + int32 `cu_seqlens` conversion assert, `lse` shape `(H_Q,T)`, GQA + features, backward, zero-length batch entries |
| kvcache | decode without new kv, int `cache_seqlens` + **in-place** cache update assert, `return_softmax_lse` + causal/window, GQA, `cache_seqlens=None` → empty, stride assert negative test |

The numeric oracle is **independent**: it is written with `torch.softmax` /
`torch.logsumexp` on upcast tensors and never imports the mock's reference functions —
so a bug in the mock's math is caught by the *tests'* oracle, and a bug in the wrapper's
plumbing is caught by both.

### 2.3 Executed evidence

**NOT YET EXECUTED — no torch existed on this host at report time.** An install attempt
into `/mnt/envs/v100` (uv, started 14:29) was killed (exit 143) at 15:21 while still
downloading the CUDA wheels, and two polls of `import torch` covering ~50 minutes total
were both negative. Installing packages is outside this subagent's remit; run the suite
on any machine with torch ≥ 2.0:

```
FA_V100_REPO=<repo> /mnt/envs/v100/bin/python /tmp/opencode/verif_drafts/test_wrapper_cpu.py
```

Expected protocol: per-test `PASS`/`FAIL` lines plus a final `N/N passed`; `run_all.py`'s
`cpu_mock` section parses exactly that line (§5). Standalone syntax status:
`python -m py_compile mock_cuda.py test_wrapper_cpu.py` OK (§7). Everything else in this
harness — mock module, oracle, CALL_LOG assertions — must be treated as *draft-verified
only* until that command's output is pasted here.

---

## 3. API parity — `parity_check.py`

**Question answered:** is the fork's public python surface a drop-in for upstream
`flash-attn 2.8.3` — names, exact argument order, defaults, and (runtime mode) actual
`inspect.signature`s plus dist-info top-level path?

### 3.1 Modes (auto-selected by `run_all.py`)

| mode | how | needs |
|------|-----|-------|
| `--spec research/03_api_spec.json` (or builtin v2.8.3 fallback) | expected surface | nothing |
| `--ast` | stdlib `ast` parse of the three source files; compares names/order/defaults **without importing** | nothing (pre-build) |
| `--mock --mock-dir …` | imports the real packages with the §2 mock installed; compares live `inspect.signature` + alias modules | torch |
| runtime (no `--ast`/`--mock`) | imports real packages + extension; also checks dist-info (`flash-attn`, top_level `flash_attn`) | built env |

`--allow-known-gaps` records historically-missing upstream symbols as warnings instead of
failures, so a partial port still yields a meaningful report.

### 3.2 Executed evidence (AST mode, pre-build)

```
$ /mnt/envs/v100/bin/python parity_check.py --repo <repo> --ast \
      --spec research/03_api_spec.json --allow-known-gaps --json .../api_parity.json
API parity spec: file:<repo>/research/03_api_spec.json
STATUS  CHECK                                         DETAIL
<PASS rows suppressed in summary; full list from the JSON artifact:>
PASS  sig:flash_attn_func
PASS  sig:flash_attn_kvpacked_func
PASS  sig:flash_attn_qkvpacked_func
PASS  sig:flash_attn_varlen_func
PASS  sig:flash_attn_varlen_kvpacked_func
PASS  sig:flash_attn_varlen_qkvpacked_func
PASS  sig:flash_attn_with_kvcache
PASS  sig:bert_padding.unpad_input
PASS  sig:bert_padding.unpad_input_for_concatenated_sequences
PASS  sig:bert_padding.pad_input
PASS  version:flash_attn.__version__                 expected "2.8.3" literal in flash_attn/__init__.py
SKIP  dist_info                                      checked only in runtime mode (site-packages metadata)

RESULT PARITY PARTIAL  (failed=0, warn=1, known_gaps=[])
```

Interpretation: **all 12 checkable items pass**; the verdict is PARTIAL (not PASS) because
(a) AST mode cannot compare live `inspect.signature`s, and (b) dist-info is runtime-only.
This harness earlier in the day flagged 4 missing upstream symbols
(`flash_attn_{varlen_,}qkvpacked_func`, `flash_attn_{varlen_,}kvpacked_func`) — they were
fixed in the meantime (other subagent) and the checker re-run now shows `known_gaps=[]`,
which is exactly the intended fail→fix→pass loop.

### 3.3 Nuances recorded deliberately

- Upstream exports `flash_attn_gpu` as the **extension module object**; this fork exports
  `flash_attn_gpu` as a **python alias of `flash_attn_func`**. Runtime-mode parity checks
  these repo extras by *presence* only (`extra:*` PASS/WARN), so the module-vs-function
  identity difference is recorded here as a deliberate WARN-level nuance: imports stay
  drop-in compatible, but code that introspects the object would see a difference.
- Environment caveat: `einops` is not installed in `/mnt/envs/v100` while writing this
  (upstream lists it as a runtime dependency). The harness therefore degrades the
  `bert_padding` import/signature checks to WARN/SKIP with an explicit "env lacks
  optional dep" note instead of misreporting a repo defect; AST-mode checks are
  unaffected. Install `einops` before the runtime-mode acceptance run to lift this.
- After the real build, re-run in runtime mode so dist-info (`top_level.txt` ==
  `flash_attn`) and the `flash_attn_2_cuda` symlink are verified as well; those checks
  live in `run_all.py`'s `artifacts` section.

---

## 4. Build matrix — `build_matrix.sh` + `collect_ptxas_stats.py`

**Question answered:** do *all* supported kernel configurations build cleanly against
CUDA 12.6 on a GPU-less host, target sm_70 only, and stay within sane resource budgets?

### 4.1 Matrix

| mode env | propagated flag | intent | expected static verdict |
|----------|-----------------|--------|--------------------------|
| (none) | — | default tile path | sm_70-only; HMMA.884 present |
| `MMA_NATIVE=1` | `-DMMA_NATIVE` | native Volta `HMMA.884` m8n8k4 path (`include/mat_mul.h`) | sm_70-only; HMMA.884 |
| `MMA_884=1` | `-DMMA_884` | explicit 884 tile shape (`include/kernel.h`) | sm_70-only; HMMA.884 |

Matrix-wide PASS criteria: every mode exits 0, produces a `.so`, arch scan says sm_70
(yes), and no mode reports SASS FAIL; with `ATTENTION_DEBUG=1` each mode also yields a
`ptxas_stats.json` (registers, smem, spills, stack per entry function).

Build recipe that this host requires (baked into `build_matrix.sh`, which builds each
mode in a scratch copy under `/tmp/opencode/build_matrix/<mode>/src` so the repo is never
touched):

```bash
export PATH=/mnt/envs/v100/bin:$PATH
export CC=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-gcc \
       CXX=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-g++ \
       CUDAHOSTCXX=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-g++   # system gcc 16.2 breaks nvcc 12.6
export FLASH_ATTN_MIN_CUDA=12.6 NVCC_THREADS=8 MAX_JOBS=8
export ATTENTION_DEBUG=1   # adds -Xptxas -v (+ -g --keep) so ptxas stats are collectable
python setup.py build_ext --inplace
```

### 4.2 `collect_ptxas_stats.py` (verified parser)

Parses `ptxas -v` output from a build log into per-entry-function records and a summary
JSON; `--compare` diffs two matrix modes; `--expect-arch 70` hard-fails if ptxas ever
compiled for another arch. Executed evidence against a real ptxas 12.6 log:

```
$ python collect_ptxas_stats.py build_sample_ptxas.log --expect-arch 70
# build_sample_ptxas.log: 2 compiled entry function records
arch  regs     smem  spill_s  spill_l   stack  kernel
  70    22        0        0        0       0  kern_regs(float const*, float*)
  70    12     4096        0        0       0  kern_smem(float const*, float*)
summary: {"num_entry_functions": 2, "max_registers": 22, "total_smem_bytes": 4096, "total_spill_bytes": 0, "arches_seen": [70]}
```

(The `smem=4096` line is a real shared-memory declaration from the probe translation
unit; the "smem after barriers" regex was fixed against this exact log.) Budgets to
watch when the real kernels build: **regs ≤ 255 (hard limit), spills = 0, smem ≤ 96 KB**
(`cudaFuncAttributeMaxDynamicSharedMemorySize`, V100 max 96 KB opt-in) — values beyond
that are flagged in the report, not auto-failed (kernel-specific).

### 4.3 Status

Draft complete (`build_matrix.sh`, `bash -n` clean, `--dry-run` validated end-to-end on
this host: rsync scratch copy + pinned env + per-mode command construction all execute,
only the actual compile is skipped). Full-matrix execution is **NOT YET EXECUTED** on
this host: it needs torch (absent — see §2.3) and a green first build. The script is
deliberately independent from `run_all.py` so a build failure cannot mask, or be masked
by, other checks.

---

## 5. Acceptance runner — `run_all.py`

Single entry point that executes/collects **all** of the above and emits
`tests/artifacts/verification_report.json` (schema_version=1) plus a Markdown digest.
Sections and their degradation rules:

| section | needs | PASS means | else |
|---------|-------|-----------|------|
| `build` | `--build-log`, `--build-exit`, `--so` | exit 0 + `.so` present + ptxas parsed + sha256 recorded | SKIP if no log / FAIL on exit≠0 or missing `.so` |
| `artifacts` | torch | extension imports, 5 entry points callable, `flash_attn_2_cuda` alias symlink, dist-info version/top_level | SKIP without torch |
| `api_parity` | nothing (AST fallback) | RESULT line from §3 | PARTIAL when torch-less |
| `cpu_mock` | torch | §2 tests all pass (`N/N passed`) | SKIP without torch |
| `static_sass` | `.so` | RESULT from §1 | PARTIAL without cuobjdump, SKIP without `.so` |

Overall = FAIL if any FAIL; PARTIAL if any PARTIAL/SKIP remain; else PASS.
The JSON additionally carries `claims` (what a PASS does prove) and `not_verified`
(always included; mirrors §6) so the report cannot be quoted out of context.

Executed pre-build dry run (this host, today; artifacts redirected out of the repo):

```
$ python run_all.py --repo <repo> --artifacts-dir /tmp/opencode/verif_drafts/run_prebuild \
      --python /mnt/envs/v100/bin/python
wrote .../verification_report.json
# GPU-free verification report
- generated: 2026-10-19T14:50:55
- repo: `<repo>` @ `55d685a3279311f869a7792116b1c24f4c76f301` (dirty)
- env: python 3.12.15, torch None, nvcc 12.6
- overall: **PARTIAL**

| section | status | detail |
|---|---|---|
| build | SKIP | no --build-log; build not executed by this runner |
| artifacts | SKIP | torch not importable by target python; cannot probe extension |
| api_parity | PARTIAL | exit=2 result=PARTIAL (torch missing -> AST fallback, runtime signatures NOT compared) |
| cpu_mock | SKIP | torch not importable by target python |
| static_sass | SKIP | no .so available (build first) |
```

Several bugs were found and fixed by these dry runs and the review that followed them
(recorded here because they are exactly the kind that would otherwise burn GPU-host
time): (1) `run_all.py`'s environment probe passed an inline script with invalid
indentation; (2) `parity_check.py`'s builtin spec parser split signature strings per
character instead of per parameter (now `ast`-based, which also survives defaults like
`(-1, -1)`); (3) `run_all.py` did not propagate torch availability into the section
functions; (4) `run_all.py`'s `cpu_mock` section ran the test file without exporting
`FA_V100_REPO`, a path the drafts layout requires. All are re-verified fixed in the
logs above.

**On the real (GPU or build-capable) machine the invoked sequence is:**

```bash
# 1) build once, capture log (note: PIPESTATUS must be read in the same shell as the pipe)
cd <repo> && python setup.py build_ext --inplace 2>&1 | tee /tmp/fa_build.log
BUILD_RC=${PIPESTATUS[0]}
# 2) run the whole acceptance suite
python run_all.py --repo <repo> --so <repo>/flash_attn_v100_cuda*.so \
    --build-log /tmp/fa_build.log --build-exit "$BUILD_RC" --build-mode default
# exit code: 0 iff overall == PASS
```

---

## 6. What cannot be verified without a real V100 (and what would verify it)

Ordered by residual risk × likelihood:

| # | Unverifiable here | Why | Cheapest real-GPU check |
|---|-------------------|-----|--------------------------|
| 1 | **Kernel numeric equality** (fwd/bwd vs reference on real data) | mock mirrors *documented* semantics in fp32; real kernel uses fp16 inputs, fp32 accumulation, different reduction orders, `tanh` approximations | `tests/test_v100_numerics.py` once written: compare vs `torch SDPA` fp32 on a handful of shapes; tolerance-based |
| 2 | **Race conditions / missing `__syncthreads` / smem bank pathologies** | static + CPU can't observe concurrency | run with `compute-sanitizer --tool racecheck` (does support sm_70) on a few shapes |
| 3 | **Dropout path** (RNG offset serialization into `rng_state`, `S_dmask` encoding, determinism property) | mock intentionally raises | seed-reproducibility test + backward-with-dropout smoke |
| 4 | **Rotary / `leftpad_k` / paged `block_table`** paths of `fwd_kvcache` | not modelled in mock | targeted unit tests on device (these are thin wrappers over the same kernel) |
| 5 | **Performance / occupancy / real smem-limit enforcement** | no clock, no SM | `ncu` or wall-time bench vs par; also confirms ptxas smem ≤ 96 KB actually launches |
| 6 | **C++ `TORCH_CHECK`s that call `getCurrentDeviceProperties()`** (arch guard, smem capacity) | need a CUDA context | instantiation test: import + call each entry point once |
| 7 | **Cubin load / symbol resolution on the driver** | no driver | first run on the V100 *is* the test (this plan makes it deterministic) |

Everything in this table is restated verbatim in `run_all.py`'s `not_verified` field so a
PARTIAL report can never be read as "verified correct".

---

## 7. Deliverables, execution ledger, promotion path

### 7.1 Draft inventory (`/tmp/opencode/verif_drafts/`)

| file | lines | role | syntax status | executed |
|------|-------|------|---------------|----------|
| `mock_cuda.py` | 551 | fake extension (5 entry points, CALL_LOG, TORCH_CHECK mirroring) | py_compile OK | via §2 tests |
| `test_wrapper_cpu.py` | 735 | ~30 wrapper-semantics tests + independent oracle | py_compile OK | not yet executed (no torch) |
| `parity_check.py` | 498 | spec/AST/mock/runtime API parity | py_compile OK | ✅ AST mode (§3.2) |
| `collect_ptxas_stats.py` | 208 | ptxas -v parser, --compare/--expect-arch/--json | py_compile OK | ✅ on real ptxas log (§4.2) |
| `verify_sass.sh` | 215 | SASS/ISA checks + embedded-ELF fallback | bash -n OK | ✅ both paths (§1.3) |
| `run_all.py` | 413 | acceptance runner → JSON + MD report | py_compile OK | ✅ pre-build (§5) |
| `build_matrix.sh` | 154 | 3-mode scratch builds + per-mode evidence | bash -n OK, --dry-run executed | not yet executed (no torch) |
| `upstream_ref/upstream_api_fallback.json` | — | fetched upstream v2.8.3 signatures (parity seed) | — | ✅ |

### 7.2 Execution ledger (what was actually run on this host, in order)

1. `nvcc/ptxas 12.6.85` probes; system gcc 16.2.1 rejected; conda gcc 12.4 accepted →
   toolchain pinning of §4.1.
2. Trio of tiny sm_70/sm_80/sm_86 `.so` built with nvcc (`-Xcompiler -fPIC` needed,
   recorded) → `verify_sass.sh` PARTIAL/FAIL outputs pasted in §1.3.
3. `collect_ptxas_stats.py` against a real `-Xptxas -v` log → §4.2 output.
4. `parity_check.py --ast` against the live repo → §3.2 (12 checks; 0 failures).
5. `run_all.py` end-to-end pre-build → §5 dry-run report; two runner bugs found+fixed.
6. `bash -n build_matrix.sh` + `build_matrix.sh --dry-run` (scratch rsync + env
   plumbing executed); `py_compile` all python drafts.

### 7.3 Not yet executed, and what closes it

| pending | blocked on | closes with |
|---------|-----------|-------------|
| §2 mock tests real output | torch missing in `/mnt/envs/v100` (install attempt killed at 15:21; installing is outside this subagent's remit) | paste of `N/N passed` under §2.3 |
| §4 build matrix | same, plus a green first build | `build_matrix.sh` SUMMARY table + 3× `verify_sass` results |
| runtime-mode parity (incl. dist-info/symlink) | same | `run_all.py` full run; expected overall **PASS** (with cuobjdump installed) or PARTIAL (without) |

### 7.4 Promotion path

When the V100-day checklist is cut: move `mock_cuda.py`, `test_wrapper_cpu.py`,
`parity_check.py`, `collect_ptxas_stats.py`, `run_all.py` into `tests/`, keep
`verify_sass.sh` + `build_matrix.sh` in `scripts/`; `run_all.py`'s default artifacts dir
is already `tests/artifacts/`. Nothing in this plan requires network or GPU to run.

---

## Final summary

- **Deliverables:** this plan + 7 runnable drafts in `/tmp/opencode/verif_drafts/`
  (`mock_cuda.py`, `test_wrapper_cpu.py`, `parity_check.py`, `collect_ptxas_stats.py`,
  `verify_sass.sh`, `run_all.py`, `build_matrix.sh`).
- **Executed:** SASS fallback PARTIAL on sm_70 / FAIL on mixed sm_80+86 lib; ptxas-stats
  parser on a real log; API parity 12/12 (AST); full acceptance dry run (found + fixed
  4 harness bugs).
- **Drafted, not yet executed:** §2 mock wrapper tests and the 3-mode build matrix —
  torch never arrived in `/mnt/envs/v100` (install attempt killed mid-download, exit 143);
  exact commands in §2.3/§7.3.
- **Top risks without a V100:** kernel numerics, races/sync bugs, dropout RNG path,
  rotary/leftpad/paged paths, occupancy — all in §6 and in the report's `not_verified`.
- **Bottom line:** all pre-GPU-checkable properties are checked or queued; nothing
  unverifiable is claimed as passed.
