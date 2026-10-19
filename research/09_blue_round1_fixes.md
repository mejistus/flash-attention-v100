# 09 — Blue-team round 1: fix audit (orchestrator review)

Context: three blue-team subagents were tasked with applying the fixes sketched in
`07_redteam_round1.md` (F-*) and `08_kernel_redteam.md` (P*/G*). All three sessions died
mid-run due to a provider-side IP block; they left **no reports**, but their edits landed
on disk (uncommitted). This document is the orchestrator's line-by-line audit of those
edits against the red-team sketches, plus the static checks performed at audit time.

Audit method:
1. `git diff` per file, checked hunk-by-hunk against the fix sketches in 07/08.
2. AST counting for the F-2 `(None,)*N` guards vs actual `forward()` arities.
3. `py_compile` on every edited Python file.
4. Fresh single-TU `nvcc` compile checks (sm_70, CUDA 12.6.85, installed torch 2.6.0
   headers) after the post-reboot environment rebuild.

## 1. Fix verdicts (all verified by source inspection)

| ID | File(s) | Fix applied | Evidence | Verdict |
|----|---------|-------------|----------|---------|
| F-2 | `flash_attn_v100/flash_attn_interface.py` | Empty-input backward guard returns `(None,)*12` (dense) / `(None,)*17` (varlen) | L159, L355; AST: `FlashAttnFunc.forward` has exactly 12 inputs, `FlashAttnVarlenFunc.forward` exactly 17 — counts match | ✅ applied, counts exact |
| F-3 | `kernel/forward_kvcache.cu` | `cache_seqlens == nullptr` → attend full cache (`cache_seqlen = max_seqlen_k - leftpad`) instead of returning zeros | L193–195 (`cache_seqlen = (cache_seqlens != nullptr) ? ... : (...)`), `total_seqlen_k = cache_seqlen + T_NEW` L195 | ✅ applied |
| F-4 | `pyproject.toml` | Build requires pinned `torch==2.6.0` | diff | ✅ applied |
| F-5 | `setup.py` | Fake `flash_attn-2.8.3.dist-info` now written inside PEP 517 wheel staging via `install_lib` override; legacy `install` path kept | `_write_fake_upstream_dist_info` L158; `InstallFakeDistInfo(install_lib)` L192; `dst_root = getattr(self, "install_lib", None)` L185 | ✅ applied (runtime proof pending wheel build) |
| F-6 | `flash_attn_v100/flash_attn_interface.py` | Dense `S_dmask` padded to upstream `(B, H, round128(M), round128(N))` via `_pad_dmask_upstream` (zero padding, documented) | L15 `round_multiple`, L20 `_pad_dmask_upstream`, L29 pad computation, L148 call site | ✅ dense applied; varlen `(T_Q,H,max_k)` deviation documented in docstring (decided: keep) |
| F-7 | `kernel/api.cpp` | `varlen_fwd` raw binding: all args annotated `py::arg(...)`, `num_splits` default 0 | L22–35 and L45–56 both modules | ✅ applied |
| F-8 | `kernel/forward.cu` (dense), `kernel/forward_varlen.cu`, `kernel/forward_kvcache.cu` | Empty-KV lse sentinel `-inf` → `+inf` (upstream-compatible), zeroed out | `forward.cu` L407–410 (`softmax_lse.fill_(+inf)` on N==0 fast path); varlen/kvcache same | ✅ applied |
| F-9 | `tests/run_all_cpu.py` | Coverage taxonomy PASS/PARTIAL/SKIP/FAIL with summary-parsing regex; no more overstated coverage | `SUMMARY_RE` L34 matches `_stub_cuda_backend.py` L132 output format `== title: p/t passed, f failed, s skipped ==` | ✅ applied, format cross-checked |
| P0-1 | `kernel/backward_varlen.cu` | Dropout RNG stride: `GLOBAL_N` uses `max_seqlen_k` at **both** dQ and dKV sites (matching fwd RNG key space) | L251–252 comment + call L260; L483–484 + call L492 | ✅ applied, both sites |
| P1-1 | `kernel/backward.cu` | `dsoftmax_sum` now written: `softmax_d` kernel param added, `dLse` written in reduction | L38 param, L122 `dLse_ptr = softmax_d + ...`, L513/561/576 launcher wiring | ✅ applied |
| P1-2 | `include/softmax.h` | Fully-masked rows now produce `P = 0` (not P=1/mean(V)): `all_masked` flag forces `e = 0` in both main and tail loops | L95–107 (main), L168 (tail); row keeps Out=0 after normalization guard | ✅ applied |
| P1-3 | `kernel/forward_varlen.cu`, `kernel/forward_kvcache.cu` | Paged KV tiles that cross a physical page boundary are loaded page-segment by page-segment (`WMMA_GEMM_LOAD_TILE_ROW_OFFSET` / `WMMA_GEMM_LOAD_PAGED_TILE`), swizzle keyed by tile-global row so segments compose | varlen L37/L91/L113 + call sites L286/L334; kvcache L36/L90/L112 + L309/L357 | ✅ applied (design reviewed; GPU-verification still pending — no GPU) |
| G-1 | `include/softmax.h` | Backward dropout Philox component selection now `r[(addr_plx+k)&3]` instead of mismatched mapping (fwd/bwd RNG stream alignment) | L114–122 (main), L174–178 (tail) | ✅ applied |
| G-2 | `include/softmax.h` | dmask overflow-path global write downgraded to alignment-safe 4×`st.global.u16` (matches row base 8B alignment) | L152–155 | ✅ applied |
| G-3 | `flash_attn_v100/flash_attn_interface.py` | `seqused_k` + grad-requiring training raises a documented `NotImplementedError`. **Post-audit correction (research/10 D1): dead code — the wrapper hardcodes `seqused_k = None` (L303), so the guard can never fire; retained as in-code documentation only. The kernel's backward ignoring `seqused_k` mirrors upstream (upstream's varlen backward has no `seqused_k` parameter either).** | L303, L314–321 | ⚠️ documentation-only |
| P1-4 | — | Refuted by red team (docstring-only issue) | no code change needed | ✅ no-op |
| P2-4 | — | Confirmed only for `MMA_884` builds; default build correct; NOT fixed this round (cannot verify without GPU) | documented in 08 | ⏸ deferred |

Support files synced with the above (semantics of the CPU mock must match the kernels):

- `tests/mock_cuda.py`: kvcache `cache_seqlens=None` → full cache; empty-KV lse `+inf`;
  varlen dmask documented shape `(T_Q,H,max_seqlen_k)`; header comment updated.
- `tests/_stub_cuda_backend.py`: `ok(name, detail)` for evidence-carrying PASS lines.
- `tests/test_interface_behavior.py`, `tests/test_wrapper_cpu.py`: F-2 empty-backward
  regression tests, F-6 shape assertions, F-5 static dist-info test, empty-KV
  expectations.
- `tests/parity_check.py`: `repo_extras` accepts dict entries from the spec (were bare
  strings → silently degraded before).
- `flash_attn/flash_attn_interface.py`: re-exports `round_multiple` added by F-6.

## 2. Static checks at audit time

| Check | Result |
|-------|--------|
| `py_compile` on 9 edited Python files | ✅ all OK |
| Single-TU `nvcc` compile (sm_70, CUDA 12.6.85, -O0 --threads 2) — api.cpp | ✅ EXIT=0 |
| Single-TU compile — forward.cu | ✅ EXIT=0 |
| Single-TU compile — backward.cu / backward_varlen.cu / forward_kvcache.cu / forward_varlen.cu | ⏳ in progress at report time (batch `/tmp/opencode/blue_r1/summary.txt`) |
| F-2 guard arity vs AST | ✅ 12/17 exact |
| F-9 regex vs stub summary format | ✅ match |

## 3. Residual gaps / notes (carried into red round 2)

1. **P1-3 early-break semantics**: `WMMA_GEMM_LOAD_PAGED_TILE` breaks out when the page
   index exceeds `block_table_pages`, leaving tile rows [rows_done, valid_kv_rows)
   unwritten (stale smem). This is only reachable with malformed block tables (the
   launcher clamps `valid_kv_rows` by `seqlen_k`; well-formed tables always cover it).
   No OOB access either way. Worth a `TORCH_CHECK` follow-up or zero-fill if we want
   belt-and-braces.
2. **P2-4 (MMA_884 mask lane mapping)**: deferred by decision (unverifiable without
   GPU; default build path unaffected).
3. **F-6 varlen dmask shape**: intentionally left at `(T_Q,H,max_seqlen_k)`; documented
   in the wrapper docstring as a known deviation from upstream `(round128 T_Q, ...)`.
4. **F-10** (missing upstream subpackages) remains a documented, intentional gap;
   `flash_attn.bert_padding` etc. are out of the drop-in surface promised for the
   quantized/rope path.

## 4. Next gates (for the record)

- Full CUDA 12.6 build (`scripts/build_cuda126.sh`, MAX_JOBS=8/NVCC_THREADS=4) — evidence
  to `tests/artifacts/build_cuda126/`.
- Python CPU suite (`tests/run_all_cpu.py`, `tests/parity_check.py`) on the built wheel.
- SASS/PTX static inspection (`scripts/verify_sass.sh` with cuobjdump/nvdisasm).
- Independent verifier report: `research/10_blue_verify.md` (separate subagent).

## 5. Post-build corrections (2026-10-19 evening)

Triggered by the first full build attempt (all 6 kernel TUs compiled clean, wheel step
failed) plus the independent verifier report (research/10):

1. **F-5 reworked — the blue wheel was invalid.** Embedding `flash_attn-2.8.3.dist-info`
   inside the wheel violates PEP 427 (exactly one `.dist-info` per wheel); uv rejected
   it: "Multiple .dist-info directories found: flash_attn-2.8.3, flash_attn_v100-26.6".
   The fake-dist-info cmdclasses were removed from `setup.py`; the upstream metadata is
   now materialised post-install by `flash_attn_v100/_upstream_metadata.py` (invoked by
   `scripts/build_cuda126.sh`, `--check` mode for verification).
   `tests/test_interface_behavior.py` was updated to the new mechanism.
2. **D5 applied (upstream parity)** — per-batch empty-KV early exits now write `+inf`
   like upstream (`flash_fwd_kernel.h:125`): `forward_varlen.cu` / `forward_kvcache.cu`
   in-kernel branches (host fast paths already wrote `+inf`); mock varlen reference synced.
3. **Test-side fixes (research/10 D2/D3/D4)** — `mock_cuda._scores_core` call sites now
   convert to BHSD (`varlen_attention_reference`, `kvcache_attention_reference`);
   `_stub_cuda_backend.fwd_kvcache` returns lse `(B,H,T)`; `test_wrapper_cpu` NameError
   `N` in the F-6 test fixed.
4. **D1 accepted as documentation-only** (G-3 row above); mirrors upstream's own varlen
   backward limitation (no `seqused_k` parameter upstream either).
5. **Build recipe fix** — conda `nvcc` invoked through `targets/x86_64-linux/bin/nvcc`
   (a symlink) cannot locate `cicc`/`cudafe++` via argv[0]-relative search; the script
   now prepends `$ENV_PREFIX/nvvm/bin` and `$ENV_PREFIX/bin` to `PATH`.
