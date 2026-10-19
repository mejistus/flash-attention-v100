# 11 — Red-team round 2 (adversarial audit of the blue round-1 fixes)

Scope: uncommitted blue fixes on `v100/cuda126-dropin` in the 7 assigned areas.
Method: read-only source inspection + cheap falsification (AST, Python repros, naive-oracle
probes, suite runs). No nvcc, no pip, no edits outside this file; no commit.

**Timing note (important):** the tree changed *during* this audit (blue follow-up wrote
`setup.py` 18:58, `tests/*` 19:00, `kernel/forward_{varlen,kvcache}.cu` 19:03). This report
re-baselines against the current revision; `research/10_blue_verify.md` is partially stale
(its D2/D3/D4/D5 are already fixed — see §c).

## (a) Per-area verdict table (current revision)

| # | Area | Verdict |
|---|------|---------|
| 1 | Paged loaders (`forward_varlen.cu`, `forward_kvcache.cu`) | ⚠️ 1 residual (R2): early `break` leaves rows unwritten **and comments claim they are masked — they are not**. kvcache `break` unreachable by construction; varlen reachable with an under-sized table (no coverage check) |
| 2 | `include/softmax.h` (all_masked, dmask alignment, Philox) | ✅ Could not break (details §c) |
| 3 | `backward.cu` / `backward_varlen.cu` (softmax_d, GLOBAL_N=max_seqlen_k) | ✅ Could not break (details §c) |
| 4 | `kernel/api.cpp` (py::arg, num_splits default) | ✅ Statically correct; `window_left` naming note only (R10). Import-time proof needs build |
| 5 | `flash_attn_interface.py` (F-2 12/17, F-6 pad, G-3 guard) | ⚠️ F-2/F-6 verified, incl. own AST 12/17; G-3 still dead code (R6, matches 10/D1); varlen dmask deviation documented |
| 6 | `setup.py` F-5 / `pyproject.toml` F-4 | ⚠️ Old wheel-multi-dist-info defect is **gone**; v2 (post-install `_upstream_metadata`) leaves R4/R5 + stale docs R7 |
| 7 | `tests/mock_cuda.py` + `tests/run_all_cpu.py` | ❌ 3 findings: whole interface suite aborts (R1), mock oracle numerically wrong (R3), runner vacuous-PASS holes (R8) |

Fresh `tests/run_all_cpu.py` (current tree): **1 PASS, 1 PARTIAL, 1 SKIP, 4 FAIL** —
`test_api_parity` PASS; `parity_check` PARTIAL (only expected pre-build dist-info SKIP);
`test_extension_smoke` SKIP; bert 12/14; interface = NameError abort; reference 22/29; wrapper 7/29.

## (b) Findings, ranked

### R1 — S2 — `test_interface_behavior.py` aborts with `NameError`; F-5 wiring test never runs. **Confirmed**
`tests/test_interface_behavior.py:511` calls `test_setup_fake_dist_info_wiring()`, but the F-5
rewrite renamed the function to `test_upstream_metadata_wiring` (defined `:339`, **never
invoked**; grep shows no other reference). Running the suite: `NameError: name
'test_setup_fake_dist_info_wiring' is not defined` after `test_empty_backward_no_crash`, before
`test_packed_dense/packed_varlen/kvcache` and `R.finish` → rc=1, no summary; `run_all_cpu`
reports FAIL and **every interface check that did run is discarded**. Blue-introduced at 19:00.
Fix: call `test_upstream_metadata_wiring()` at `:511`.

### R2 — S2 — P1-3 residual: unwritten paged rows are *not* masked; in-code claim is false. **Confirmed (code); needs GPU for the numeric witness**
- `forward_varlen.cu:112-122` / `forward_kvcache.cu:111-121`: `for (seg…) { if (page_idx+seg >= block_table_pages) break; … }` — rows `[rows_done, valid_kv_rows)` are never written.
- Call sites pass the full `valid_kv_rows` to the loader (`:287-290` K, `:335-338` V in varlen; kvcache `:311/:359`), while `WMMA_GEMM_SCORES` masks only `global_n >= GLOBAL_N + VALID_N` (`include/mat_mul.h:87-92,98-107`) with the *same* `valid_kv_rows` (`:306-311`). Unwritten rows have index `< VALID_N` → `in_bounds` → softmaxed as real keys (stale smem on K, stale/undefined on V).
- The comments "rows stay masked" (`:113`) and "WMMA_GEMM_SCORES masks them" are therefore wrong.
- Reachability: varlen — user passes `max_seqlen_k`/`cu_seqlens_k` larger than `block_table.size(1)*page`; the host checks (`:536-546`) validate dtype/shape but **not table coverage**. kvcache — `max_seqlen_k = max_num_blocks_per_seq * page_block_size` (`:600`), so the break is unreachable there.
Fix sketch: make the loader return `rows_done` and mask with it (or zero-fill remaining smem rows), or add `TORCH_CHECK(max_num_blocks_per_seq * page_block_size >= max(seqlen_k))` for varlen.

### R3 — S2 — mock oracle computes unnormalized attention; F-3 test fails for mock reasons. **Confirmed (empirical)**
`tests/mock_cuda.py:161-172` returns `p = exp(s−m)` unnormalized, and `:212-213` does
`out = torch.matmul(p, vx.float())` without dividing by `psum`. Probe (dense, causal, seed 0):
mock vs naive softmax → out `max|d| = 2.35`, lse `0.0`; kvcache F-3 test setup → out differs
2.37, lse exact. Consequence: `test_wrapper_cpu.py` 7/29 (22 fails, most "values differ"),
including `test_kvcache_none_seqlens_attends_full_cache` (`:699-716`), whose wrapper-level
`out_none == out_full` assertion *does* pass — the failure is the mock's wrong `out`. So the CPU
suite currently supplies **no valid numeric evidence** for F-3/F-8/F-6 values; "test semantics
synced" (09 §3) overstates. Pre-existing at HEAD (`git show HEAD:tests/mock_cuda.py` same line),
exposed once the suite became runnable. Fix: `out = (p / psum[..., None]) @ v` (and re-check
gradients, which flow through the same core).

### R4 — S2 — F-5 v2: metadata is materialised only on the install branch; wheel path loses it. **Confirmed (static)**
`scripts/build_cuda126.sh:41-51`: the `--wheel` branch runs `uv build` and returns — only the
`uv pip install` branch runs `_upstream_metadata.py` + `--check`. `utils/docs/install_cuda126.md`
§3 advertises the wheel option, and PEP 427/517 has no post-install hook, so a wheel-only
install (or plain `uv pip install .`) yields no `flash-attn` dist-info →
`importlib.metadata.version("flash-attn")` raises for downstream probes. Fix sketch: state the
extra step in the wheel instructions (or emit a `flash_attn_v100` console entry point /
documented post-install command).

### R5 — S3 — the fake dist-info has no RECORD/WHEEL; uninstall/force-reinstall of `flash-attn` breaks under pip. **Confirmed (source + repro of the writer)**
Empirical: `_upstream_metadata.write_dist_info()` writes exactly `METADATA` + `top_level.txt`
(no RECORD/WHEEL; discovery works: name `flash-attn`, version `2.8.3`, top_level `flash_attn`;
idempotent). pip's uninstall path raises `UninstallMissingRecord` ("RECORD file not
found", `pip/_internal/exceptions.py:763-777`) for a dist whose `iter_declared_entries()`
returns None (`req_uninstall.py:58-77, 505-507`) → `pip uninstall flash-attn` /
`pip install --force-reinstall flash-attn` hard-fail; `uv uninstall flash_attn_v100` leaves the
fake tree orphaned (not in the fork's RECORD). Minimal fix: also write an empty `RECORD` (and
`WHEEL`) file, or document the intended recovery command in the install doc.

### R6 — S3 — G-3 guard still dead code. **Confirmed** (agrees with 10/D1)
`flash_attn_v100/flash_attn_interface.py:303` hard-sets `seqused_k = None`; the check at
`:314-321` can never fire, and no wrapper exposes `seqused_k` (upstream 2.8.3 varlen signature
has none). Keep-as-documentation only; delete or wire it.

### R7 — S3 — F-5 documentation stale in the blue report set. **Confirmed**
`research/09_blue_round1_fixes.md:23` and `research/10_blue_verify.md:29` still describe the
**removed** `InstallFakeDistInfo`/`install_lib` wheel-staging implementation
(`setup.py` no longer contains either symbol; its `:156-161` comment states the new design).
09:44 refers to an F-5 test that exists but is never invoked (R1).

### R8 — S3 — `run_all_cpu.py` still has vacuous-PASS paths; smoke parity is weak. **Confirmed**
`classify_test` (`:47-56`) reads only rc and the *skipped* bucket: (i) rc=0 with **no** summary
line → PASS (probe run: `no-summary rc=0 → PASS`); (ii) rc=0 with `0/10 passed, 10 failed,
0 skipped` → PASS (the `failed` group(3) is never read); (iii) `0/0 passed, 0 failed, 0 skipped`
→ PASS. Named weak checks: **`test_extension_smoke.py:29-31,44-50`** — post-build it asserts
only `callable(...)`/torch.ops registration, so a numerically wrong kernel passes the CPU
runner (rapport: the suite cannot catch a real regression post-build); `test_api_parity.py`
compares signatures only. `parity_check.py` is always run with `--allow-known-gaps` (`:63/66/69`)
which softens any failure whose name is a known-gap (`:473-477`). Fix: treat rc=0 with no/zero
summary and rc=0 with `failed>0` as PARTIAL/FAIL.

### R9 — S3 — `test_kvcache_stride_assert` fails for the wrong reason. **Confirmed**
`tests/test_wrapper_cpu.py:721` (`test_kvcache_stride_assert`): `k_cache.transpose(1,2)` keeps `stride(-1)==1`, so the
wrapper's tail-contiguity assert never trips; the mock then reads the (B,H,N,D) layout as
(B,N,H,D) and raises `RuntimeError: H_Q must be divisible by H_K` — not the expected
`AssertionError`. Test premise (or wrapper shape validation) needs fixing.

### R10 — S3 — cosmetic: `py::arg("window_left"/"window_right")` vs upstream C++ names `window_size_left/right`. **Confirmed**
`kernel/api.cpp:33,54`; upstream 2.8.3 has **no** `py::arg` annotations at all (grep over
`fa283_api.cpp` = 0 hits), and the wrapper calls the binding positionally, so no user-visible
break; the mock/stub mirror the fork names. No action needed beyond consistency.

### R11 — S3 — informational: `pyproject.toml` `torch==2.6.0` cannot select the cu126 variant under an isolated build. **Confirmed (reasoning)**
PyPI's default `torch==2.6.0` wheel is cu124; the setup gate (`setup.py:185-189`, min 12.6) then
raises a clear error. The documented flow uses `--no-build-isolation` everywhere, so the pin is
inert there. No silent path to a bad build found.

## (c) Attacked and could NOT break (check run)

- **softmax.h**: re-read `:98,104-107` (`all_masked` → e=0), `:140-156` (G-2 gate `(STRIDE&3)==0`
  **and** scalar fallback), `:168` tail, `:174-178`, `:310-325` (G-1 `r[(c0+k)&3]` equals fwd
  `(addr+k)&3` for 4-aligned cols). All `BLOCK_N` configs ≡0 mod 4 (`include/forward.h:9-22`) →
  the 8B-aligned branch is the taken one. NEG_INF is finite `-1e30`, so `== NEG_INF` comparisons
  are well-defined.
- **backward.cu**: `dsoftmax_sum` write `:255-259` executes in the dQ phase for every block of
  `grid_dq` covering all M rows (`:537`); alloc `:702`; wrapper reads only `grads[0..2]`
  (`interface.py:174-185`). **backward_varlen.cu**: `GLOBAL_N=max_seqlen_k` at both sites
  (`:260,:492`), matching fwd RNG base `H_Q*max_seqlen_k`; dLse covered by `:144,:287` over
  `grid_dq` of `max_seqlen_q` (`:584`).
- **F-2 arity**: own AST over `flash_attn_v100/flash_attn_interface.py` → `FlashAttnFunc.forward`
  13 args incl. `ctx` (=12), `FlashAttnVarlenFunc.forward` 18 incl. `ctx` (=17); guards at
  `:159` `(None,)*12` / `:355` `(None,)*17` exact.
- **F-6 dense**: `_pad_dmask_upstream` (`:20-29`, call `:148`) matches upstream's rounded
  allocation (`fa283_api.cpp:443-448`); `test_dense_dropout_shape_only` **passes** on the current
  tree (ran it).
- **F-6 varlen**: deviation is real (upstream allocates 4-D rounded `p`, `fa283_api.cpp:655`)
  but documented in the docstring (`:413-417`) and mirrored by the kernel's tight dmask
  (`forward_varlen.cu:632`) — accepted residual, not a break.
- **F-5 old defect fixed**: `setup.py` contains no `install_lib`/`InstallFakeDistInfo` (its own
  wiring test asserts this); the writer module works (write → `importlib.metadata` discovery →
  idempotent) — repro run with `/mnt/envs/v100/bin/python`, temp dir only.
- **api.cpp F-7**: 22 `py::arg` in both modules, order matches the C++ parameter list, trailing
  `num_splits = 0` preserves upstream's 21-positional calls; C++ default `num_splits=0`.
- **Paged segment composition**: `WMMA_GEMM_LOAD_TILE_ROW_OFFSET` keys the swizzle by
  `ROW_OFFSET+row` so page-segment writes compose for >2 pages (read both loaders and the
  helper; no counterexample beyond R2).
- **Runner taxonomy sanity**: `test_extension_smoke` is correctly `SKIP` pre-build (0/0 passed,
  1 skipped) and `parity_check` is `PARTIAL` for the expected pre-build `dist_info` SKIP only
  (`:393`).
- **10_blue_verify D2/D3/D4/D5 already fixed** in the current revision: mock kvcache/varlen
  references now permute correctly (probe: kvcache ref returns `(2,1,2,64)`/`(2,2,1)`);
  `N = M` at `test_wrapper_cpu.py:476`; stub returns `lse (B,H,T)` (`_stub_cuda_backend.py:82-84`);
  per-batch empty-KV early exits write `+inf` (`forward_varlen.cu:205-215`,
  `forward_kvcache.cu:219-229`).

## (d) Residual "needs GPU" list

1. R2 numeric witness: run varlen with a deliberately short `block_table` — expect silently
   wrong out instead of masked rows; decide check vs zero-fill.
2. F-6 padding values: zeros (fork) vs kernel-written upstream padding; varlen dmask shape
   deviation acknowledged.
3. F-8/F-2/F-3 end-to-end on the real extension (empty inputs, `cache_seqlens=None`, +inf lse).
4. F-7: import-time pybind annotation check and 21-positional raw call.
5. Dropout Philox fwd/bwd mask equality on hardware.
6. R5 with real pip/uv: wheel-only install metadata absence + `pip uninstall flash-attn`
   `UninstallMissingRecord` (both blocked here by the no-pip constraint; source-level confirmed).
7. P2-4 `MMA_884` lane mapping (deferred, unverifiable without GPU).

No S0/S1 defects found in the current revision. The old F-5 wheel-staging S1 hypothesis is
resolved by the 18:58 design change; R1–R3 are the items to fix before the build gate.

## (e) Disposition (orchestrator, after the build/install)

Triage against the current revision. No kernel algorithm changes were needed
beyond the host guard/comments below; evidence lives in `tests/artifacts/`,
`research/09` §5–6 and the git diff.

| ID | Disposition | Evidence |
|----|-------------|----------|
| R1 | Fixed — registry call renamed to `test_upstream_metadata_wiring`; the suite runs the new F-5 test | `test_interface_behavior`: 113/113 |
| R2 | Fixed — varlen host now rejects under-sized block tables (`TORCH_CHECK need_b <= block_table_pages`, `kernel/forward_varlen.cu`); the loader comments now state the real behaviour (unwritten rows are stale smem, consumers do **not** mask them); kvcache is unreachable by construction (host sizes `max_seqlen_k` from the table) and documented as such | `forward_varlen.cu`, `forward_kvcache.cu` |
| R3 | Fixed — `_softmax_from_scores` now normalises `p` before `p @ v`; this was the root cause of the 22 failing wrapper assertions | `tests/mock_cuda.py`; `test_wrapper_cpu`: 29/29 |
| R4 | Fixed — `--wheel` branch materialises `flash_attn-2.8.3.dist-info` next to the wheel (`_upstream_metadata.py --dest dist`) | `scripts/build_cuda126.sh` |
| R5 | Fixed — the writer emits `WHEEL`, `INSTALLER` and an (empty) `RECORD` in addition to `METADATA`/`top_level.txt` | `flash_attn_v100/_upstream_metadata.py` |
| R6 | Accepted as documentation-only — the guard cannot fire (wrapper hardcodes `seqused_k=None`) and is kept as an honest in-code note; upstream's varlen backward has no `seqused_k` either | `research/09` G-3, `research/10` D1 |
| R7 | Fixed — docs updated to the post-install metadata design | `research/09` §5–6, `utils/docs/install_cuda126.md` §3b, README |
| R8 | Fixed — `classify_test` FAILs scripts that exit 0 without a summary line; parity mode detection now imports `torch` before the extension (the false "extension not built" verdict is gone) | `tests/run_all_cpu.py`; final run 7/7, parity mode `runtime` |
| R9 | Fixed — the stride test now builds a genuinely last-dim-strided cache (`big[..., ::2]`), so the wrapper assert fires | `test_wrapper_cpu`: 29/29 |
| R10 | Fixed — raw binding keyword names aligned with upstream (`window_size_left`/`window_size_right`); the Python layer still calls positionally | `kernel/api.cpp` |
| R11 | Accepted — the documented recipe installs torch first and builds with `--no-build-isolation`; a `pyproject.toml` pin alone cannot select the cu126 wheel under isolation | install doc |

### Additional build-time finding (not in the red-team list)

* Shipping the same shared library under both module names
  (`flash_attn_v100_cuda*.so` **and** `flash_attn_2_cuda.so`) aborts any process that
  imports both: the second `dlopen` re-runs the static initialiser and trips
  `Only a single TORCH_LIBRARY can be used to register the namespace flash_attn_v100`.
  Downstream code imports both names (vLLM does), so this was a real drop-in blocker.
  Replaced by the top-level `flash_attn_2_cuda.py` alias module: it imports the one
  extension and replaces itself in `sys.modules`, so both names resolve to the same
  module object, in either import order, with a single library load.
