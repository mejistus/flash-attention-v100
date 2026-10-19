# 10 — Blue round 1: independent verification

Verifier: separate audit session. Only this file was created; no source edits, no build, no commit.
Tree: branch `v100/cuda126-dropin`, HEAD `cfcb0cd` + uncommitted blue edits (23 files, +952/−177).
Env: `/mnt/envs/v100` torch 2.6.0+cu126, `torch.cuda.is_available()=False`; CUDA extension **not built**.

Method: hunk-by-hunk `git diff` vs the fix sketches in `07`/`08`; pristine HEAD snapshot for
attribution (`git archive HEAD` → `/tmp/opencode/fa_head`); upstream FA 2.8.3 sources
(`fa283_api.cpp`, `fa283_block_info.h`, `fa283_fwd_kernel.h`) for semantics; `py_compile`;
CPU suites; algebraic re-derivation of the philox/RNG mappings. Suspicious-passes chased down.

## A. Per-fix verdicts (assigned list)

| ID | Where | Landed | Correct | Evidence / notes |
|----|-------|--------|---------|------------------|
| F-8 dense | `kernel/forward.cu` | yes | yes | `:405-410` `N==0` → `out.zero_()`, `softmax_lse.fill_(+inf)`, unconditional return |
| F-8 varlen | `kernel/forward_varlen.cu` | yes | yes (host) | `:643-650` zero+`+inf` **overrides** the `zero_tensors` pre-fill at `:637-639`; per-batch gap → D5 |
| F-8 kvcache | `kernel/forward_kvcache.cu` | yes | yes (host) | `:733-741` `T_Q==0 || max_seqlen_k==0` → zero + `+inf`; per-batch gap → D5 |
| F-3 | `kernel/forward_kvcache.cu` | yes | yes | `:190-194` `cache_seqlen = (seqlens!=nullptr) ? seqlens[b] : max_seqlen_k - leftpad`; matches upstream `fa283_api.cpp:1265` + `fa283_block_info.h:23` |
| F-7 | `kernel/api.cpp` | yes | yes (static) | `:25-35` and `:46-56`: 21 named `py::arg` + `num_splits=0`, same order as C++ (`forward_varlen.cu:476-500`, default `num_splits=0` L481); runtime import check needs build → D6 |
| G-1 | `include/softmax.h` | yes | yes | `:310-325` `c0=addr_plx&3`, `r[(c0+k)&3]`; equals fwd `(addr+k)&3` (`:114-122`) since `col` is 4-aligned by construction |
| P1-2 | `include/softmax.h` | yes | yes | `:98` `all_masked=(new_max==NEG_INF)`; `e=0` in main loop `:104-107` and tail `:168`; out stays 0 via normalization guard |
| G-2 | `include/softmax.h` | yes | yes | `:145-160` `st.global.v4.u16` gated on alignment; else 4× scalar `st.global.u16` |
| P0-1 | `kernel/backward_varlen.cu` | yes | yes | `:251-260` and `:483-492` both pass `max_seqlen_k` as `GLOBAL_N`; matches fwd RNG base `forward_varlen.cu:323-324` (`H_Q*max_seqlen_k`) |
| P1-1 | `kernel/backward.cu` | yes | yes | launcher param `:513`, ptr `:561`, call `:576`; `dsoftmax_sum` alloc `:702`, kernel write `:255-259` (`dLse_ptr[tid]=sRowDot[tid]`), returned `:730`; wrapper reads only `grads[0..2]` (`flash_attn_interface.py:174-185`) |
| F-2 | `flash_attn_v100/flash_attn_interface.py` | yes | yes | `:159` `(None,)*12`, `:355` `(None,)*17`; AST: `FlashAttnFunc.forward` 12 inputs, `FlashAttnVarlenFunc.forward` 17 — exact; empty-backward regression tests pass (no backend bwd recorded) |
| F-6 | `flash_attn_v100/flash_attn_interface.py` | yes | yes | `:15` `round_multiple`, `:20-29` `_pad_dmask_upstream`, call `:148`; interface test `dmask→(2,3,128,128)` PASS; re-export `flash_attn/flash_attn_interface.py:14,46`; varlen stays `(T_Q,H,max_k)` (documented deviation) |
| G-3 | `flash_attn_v100/flash_attn_interface.py` | yes | **no — dead code** | guard text `:307-321`, but `seqused_k = None` hardcoded `:303` → check can never fire → D1 |
| F-5 | `setup.py` | yes | static only | `:18` import, `:185` `self.install_lib`, `:192-202` `InstallFakeDistInfo(install_lib)`, `:260` cmdclass; no wheel build run (forbidden) → D6 |
| F-4 | `pyproject.toml` | yes | yes | `:7`/`:24` `torch==2.6.0` |
| F-9 | `tests/run_all_cpu.py` | yes | yes | `:34-35` `SUMMARY_RE`/`PARITY_RE`; HEAD runner (exit-code only) reported `PASS test_extension_smoke.py` although 0/0 ran; new runner prints `SKIP` + "SKIP is NOT a pass" (verified) |
| P1-3 | `forward_varlen.cu`, `forward_kvcache.cu` | yes | yes (structural; GPU pending) | `forward_varlen.cu:37-67,:91-123`, calls `:286,:334`; kvcache mirrors `:36-66,:90-122`, calls `:309,:357`; segment loop handles >2 pages; `ROW_OFFSET+row` composes because swizzle depends only on row&7; residual note D7 |
| tests/mock sync | `tests/mock_cuda.py` | partial | partial | F-3 `pos=N` `:297-299` + `T_new` guard `:288-291`; +inf `:311-313`; varlen dmask `:440`. Two items not synced → D2, D4 |

## B. py_compile (re-run during this audit)

`python -m py_compile` on `flash_attn/flash_attn_interface.py`, `flash_attn_v100/flash_attn_interface.py`,
`setup.py`, `tests/{run_all_cpu,parity_check,mock_cuda,_stub_cuda_backend,test_interface_behavior,test_wrapper_cpu}.py`
→ **all OK** (`PY_COMPILE_ALL_OK`).

## C. Test evidence (fresh run, dirty tree)

`tests/run_all_cpu.py` → **exit 1**; `-- 7 checks: 1 PASS, 1 PARTIAL, 1 SKIP, 4 FAIL`:

| suite | dirty result | HEAD baseline (fa_head) | attribution |
|-------|--------------|--------------------------|-------------|
| `test_api_parity.py` | PASS 16/16 (1 skip) | same | — |
| `parity_check.py` | PARTIAL (exit 2), `failed=0, warn=1: dist_info metadata not installed` | n/a | `--ast` mode gives same PARTIAL; plain runtime mode FAIL (exit 1) only because the extension is unbuilt (6 import/sig failures) — expected pre-build |
| `test_extension_smoke.py` | SKIP 0/0 (1 skip), exit 0 | old runner counted it `PASS` on exit 0 | blue F-9 fixed a wrong-reason pass |
| `test_interface_behavior.py` | 113/114, 1 fail (`kvcache lse (1,2,1)`) | crashed at `test_dense_empty`: `TypeError: _Result.ok() takes 2 positional arguments but 3 were given` | blue stub fix unblocked suite; 1 stub-semantics gap left (D4) |
| `test_wrapper_cpu.py` | 5/29 | 7/29 (list `/tmp/opencode/head_wrapper.txt`) | 2 blue test-level regressions (D2, D3) |
| `test_bert_padding.py` | 12/14 (2 value fails, 3 skips) | crashed in `check()` with the same `_Result.ok` TypeError | exposed, not caused (test+subject unchanged in diff) |
| `test_reference_cpu.py` | 22/29 (7 fails) | identical 7 failures | pre-existing (window/default-scale drift) |

Broad `values differ` mock failures are identical at HEAD (the suite was never runnable before torch
was installed). No remaining wrong-reason passes found; F-2 empty-backward checks genuinely run
(stub asserts 0 backend bwd calls).

## D. Missing / partial / suspicious findings

1. **G-3 dead guard** — `flash_attn_v100/flash_attn_interface.py:303` sets `seqused_k = None` immediately
   before the guard at `:314-321`, so the `NotImplementedError` can never trigger; the raw
   `flash_attn_v100_cuda.varlen_fwd` binding still accepts `seqused_k` with grad-requiring inputs and
   the bwd kernel ignores it → silently wrong grads remain possible for raw-extension users.
   The comment at `:311-313` admits "always None here"; `09_blue` marks this ✅ — misleading, fix or drop it.
2. **F-3 test cannot pass** — `tests/mock_cuda.py:315-316` feeds kvcache-layout `qb` `(1,T_Q,H_Q,D)`
   into `_scores_core`, which expects `(B,H_Q,M,D)` (`:178-181`) and fails `_check_gqa` (`:183`);
   probe: `q (2,1,2,64)`, `k (2,16,2,64)` → `RuntimeError: H_Q must be divisible by H_K` (H_Q read as 1).
   Same bug in the varlen reference `:252`. Pre-existing at HEAD, but it now blocks the renamed
   `test_kvcache_none_seqlens_attends_full_cache` (`test_wrapper_cpu.py:699-716`). Kernel fix itself is correct.
3. **F-6 test bug (blue-introduced)** — `tests/test_wrapper_cpu.py:466-480`: uses `N` at `:476` and `:479`
   but `N` is not defined in the function or module → `NameError: name 'N' is not defined`.
   At HEAD this test passed with the old shape assertion. Fix: `N = M` (or the real K length).
4. **Stub not fully synced** — `tests/_stub_cuda_backend.py:82-83` unpacks `B,H,M,_ = q.shape` for the
   kvcache layout `(B,T,H,D)`, returning lse `(1,1,2)` where `test_interface_behavior.py:463` expects
   `(1,2,1)`; this is the single interface failure (113/114). Blue added `ok(name, detail)` (`:117-119`,
   fixing the HEAD crash) but did not fix `fwd_kvcache` shapes.
5. **F-8 residual: per-batch empty KV in-kernel paths still write `NEG_INF`** —
   `forward_varlen.cu:206-212` (`block.block_min >= block.block_max || seqlen_k == 0`) and
   `forward_kvcache.cu:217-225` (`block_min >= block_max || total_seqlen_k == 0`); upstream early exit
   writes `INFINITY` (`fa283_fwd_kernel.h:125`). Host fast paths only cover whole-tensor emptiness.
   Follow-up recommended (upstream-visible convention).
6. **F-5 / F-7 not runtime-verified** — no wheel build and no nvcc by constraint. F-7 is static-only:
   a wrong arg name surfaces at extension import time; py::arg order matches the C++ params
   (`forward_varlen.cu:476-500`). Author's own report defers to a wheel build; same for F-5.
7. **P1-3 early break leaves rows unwritten** — `forward_varlen.cu:113` / `forward_kvcache.cu:112`
   `break` when the block table is exhausted early (stale smem rows, no OOB). Only reachable with a
   malformed `block_table`; blue documented it as a residual; TORCH_CHECK or zero-fill suggested.
8. **Attribution vs `09_blue_round1_fixes.md`** — all fix-level claims matched source except G-3
   (dead code, see D1) and the "test semantics synced" claim, which held only for mock F-3/F-8/dmask;
   stub kvcache shapes and the mock `_scores_core` layout remain mismatched. Broad suite failures exist
   identically at HEAD and had never passed in this environment.

## E. Bottom line

All assigned fix IDs are present in the working tree and statically consistent with upstream 2.8.3
and with the fwd/bwd RNG algebra (dense, varlen, kvcache, paged loaders, softmax/dropout, backward
dLse). Not verifiable here and still blocking closure: CUDA build + GPU numeric parity for the kernel
edits. Before the next GPU gate, fix D1 (dead guard), D2/D4 (mock/stub semantics), D3 (NameError test),
and decide D5 (per-batch `+inf`).
