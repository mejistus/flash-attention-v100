# 07 — Red-team round 1: audit of `fdf8631` ("CUDA 12.6 build support + FlashAttention-2 API drop-in parity + CPU test suite")

Reviewer: RED TEAM round 1 (adversarial). Date: 2026-10-19.
Target: branch `v100/cuda126-dropin`, commit `fdf8631` (vs parent `55d685a`), plus its follow-up state.
Reviewed snapshot: **working tree dirty** — `README.md`, `flash_attn/flash_attn_interface.py`, `flash_attn_v100/flash_attn_interface.py`, `research/03_*`, `research/05_*` modified; HEAD has since moved to `cfcb0cd` ("fix softcap path: `__tanhf`..."); `utils/docs/install_cuda126.md` and `tests/artifacts/` untracked. All line numbers below refer to that snapshot unless marked "upstream".
Method: static analysis only. Torch is **not importable** on this machine (env under construction; `/mnt/envs/v100` off-limits per task), so no runtime/GPU claim is made. "Upstream" = genuine flash-attn 2.8.3 sources at `/tmp/opencode/fa_upstream/flash_attn-2.8.3`. Parity judged against upstream sources, NOT against `research/03_api_spec.json` (same workstream artifact).

## 1. Executive ranked summary

| # | Sev | ID | One-line |
|---|-----|----|----------|
| 1 | High | F-1 | The commit's headline "CUDA 12.6 build support" is **not reproducible at `fdf8631`**: `include/mat_mul.h:116,145` call `__tanhf`, which does not exist in the CUDA 12.6 headers. Fixed only in the next commit `cfcb0cd`. |
| 2 | High | F-2 | `backward()` unconditionally unpacks `ctx.saved_tensors`, but all four "empty input" early returns skip `save_for_backward` → **guaranteed crash** (`ValueError`) for `requires_grad` inputs on empty paths. Not tested. |
| 3 | High | F-3 | `cache_seqlens=None` means "attend the full cache" upstream; the fork silently returns all-zeros / `lse=-inf` (`kernel/forward_kvcache.cu:85`). The repo's own test enshrines the divergence. |
| 4 | High | F-4 | `pyproject.toml` build isolation resolves **unpinned `torch`** while runtime pins `torch==2.6.0` → ABI-mismatch wheels for plain `pip install .` / `uv build` (docs/scripts mitigate with `--no-build-isolation`). |
| 5 | Med-High | F-5 | Fake `flash_attn-2.8.3.dist-info` is written only by the legacy `install` cmdclass, which PEP 517 wheel/uv installs never run → `importlib.metadata.version("flash-attn")` fails after the advertised install path. research/04's `install_lib` fix was drafted but not applied. |
| 6 | Med | F-6 | `return_attn_probs=True` returns `S_dmask` with fork shapes `(B,H,M,N)` / `(T_Q,H,Q,max_k)` instead of upstream's padded `(B,H,round128(M),round128(N))` / 4-D varlen tensor. |
| 7 | Med | F-7 | Raw extension `flash_attn_2_cuda.varlen_fwd` requires 22 positional args (pybind has no default for `num_splits`); upstream is 21 → raw drop-in callers get `TypeError`. |
| 8 | Med | F-8 | C++ empty-KV sentinel is `-inf`, upstream and the fork's own Python shim use `+inf`; raw-ext / `torch.ops` users see the wrong sentinel, and the kvcache test asserts `-inf`. |
| 9 | Med | F-9 | "CPU test suite" coverage is overstated: runner skips `parity_check.py`; extension smoke test trivially passes when the extension is absent; empty-path backward, paged KV, RoPE and varlen-dropout bwd are untested; several tests assert fork behavior, not upstream behavior. |
| 10 | Med | F-10 | `flash_attn.layers/modules/ops/utils/models` are absent (parity_ast `missing=22`); downstream (vLLM/transformers) silently degrades — acknowledged in research/06 but untouched by a commit claiming drop-in parity. |
| 11 | Info | F-11 | Commit-message audit: arity fix claims are accurate; "drop-in parity" is overstated (F-3/F-6/F-7/F-10); "verified" compile claim was premature (F-1). |
| 12 | High* | F-12 | Cross-ref (not introduced by this commit, not mitigated either): top real-kernel risks P0-1/G-1 (dropout RNG stride/component mismatch → silent wrong gradients in the **main dense API**), P1-2 (fully-masked rows → `mean(V)`), P1-3 (paged cross-page reads) remain; `gpu_verify.py` does not cover varlen-bwd dropout or paged KV. Details: research/02 §6, research/08. |

*F-12 severity is High for the product; it is out of the commit's diff but directly contradicts the "parity/verification" story.

## 2. Findings

### F-1 — "CUDA 12.6 build support" is falsified at the audited commit
- **Severity**: High (claim-level; fixed at HEAD, not fixed at `fdf8631`)
- **File:line**: `include/mat_mul.h:116` and `:145` at `fdf8631`; fix in `cfcb0cd`.
- **Claim**: commit `fdf8631` says "Build (CUDA 12.6, torch 2.6.0, no-GPU)" and implies the tree builds with CUDA 12.6.
- **Evidence**:
  - `git show fdf8631:include/mat_mul.h | grep tanhf` → `__fmul_rn(SOFTCAP, __tanhf(...))` at lines 116 and 145.
  - `grep -rl "__tanhf" /mnt/envs/v100/targets/x86_64-linux/include/` → no matches (CUDA 12.6 headers define `tanhf`, not `__tanhf`).
  - Follow-up commit `cfcb0cd` (message): "fix softcap path: **__tanhf is not a CUDA device intrinsic (CUDA 12.6)**", replacing it with `tanhf` in both `WMMA_GEMM_SCORES` masking paths. `git log -S __tanhf -- include/mat_mul.h` shows `fdf8631` inherited the call and `cfcb0cd` removed it.
  - Untracked preflight artifacts `tests/artifacts/preflight_compile/` (timestamp 15:23, i.e. after `cfcb0cd` at 15:16) cover only 5 of 6 TUs (`api`, `backward`, `backward_varlen`, `forward_kvcache`, `forward_varlen` → `EXIT=0`); there is **no `forward.*` artifact for `fdf8631`** and no full-link/import proof anywhere.
- **Fix**: none required at HEAD; for process: attach per-TU compile logs to the commit that claims them, and never self-verify "build support" in the same commit that first touches the build config (here the verification arrived one commit later).
- **Verify**: `git checkout fdf8631 && nvcc -arch=sm_70 -c kernel/forward.cu` (expected error on `__tanhf`); at `cfcb0cd` the same command succeeds and `forward.status` should exist next to the other TUs.

### F-2 — Backward crashes after any empty-input forward with `requires_grad`
- **Severity**: High (live crash; easy repro; hidden by tests)
- **File:line**: `flash_attn_v100/flash_attn_interface.py:69-85` (dense B==0/M==0), `:78-85` (N==0), `:115` (save), `:129-130` (unpack); varlen `:237-255`, `:288`, `:304-305`.
- **Claim**: the "empty input early returns" are parity improvements; combined with the unchanged backward they produce a Python-level crash for any differentiable empty call.
- **Evidence**:
  - All four early-return branches `return` before `ctx.save_for_backward` (lines 75/76, 84/85, 243/244, 254/255). `backward` then does `q_, k_, v_, out_, lse_, rng_state = ctx.saved_tensors` (dense :130) / 8-tuple (varlen :305) → `ValueError: not enough values to unpack (expected 6/8, got 0)`.
  - Upstream has no Python early return; it always runs `ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)` when `is_grad` (`/tmp/opencode/fa_upstream/.../flash_attn_interface.py:847-848`), and the C++ zeroes/`+inf`-fills (upstream `flash_api.cpp:501-503`), so an upstream backward after empty forward at least reaches the extension.
  - Tests never combine the two: `tests/test_interface_behavior.py:118-143` (`test_dense_empty`) and `:240-263` (varlen empty) call no `.backward()`; backward tests use non-empty inputs (`:175-197`, `:265-283`).
- **Fix**: minimal and safe — in each early-return branch, if `is_grad`, save well-shaped placeholders (`ctx.save_for_backward(torch.empty(0...), ...)`-style consistent with the branch) and set all `ctx.*` attrs; and/or start `backward` with `if not ctx.saved_tensors: return (None,)*12` (dense) / `(None,)*17` (varlen). The guard alone fixes the crash; the save keeps `torch.autograd` metadata sane.
- **Verify**: CPU/stub test (works without GPU once torch is installed): `q = torch.randn(0,4,2,8, requires_grad=True); flash_attn_func(q,q,q).sum().backward()` and the `k` shaped `(2,0,3,8)` and varlen `max_seqlen_k==0` variants must not raise; add to `test_interface_behavior.py` next to the existing empty tests. No GPU needed thanks to the recording stub.

### F-3 — `cache_seqlens=None` silently returns zeros instead of attending the full cache
- **Severity**: High (silent wrong output for a legitimate upstream call pattern)
- **File:line**: `flash_attn_v100/flash_attn_interface.py:399,423-428` (None passed through); `kernel/forward_kvcache.cu:85` (`cache_seqlens != nullptr ? ... : 0`).
- **Claim**: drop-in parity for `flash_attn_with_kvcache`.
- **Evidence**:
  - Upstream semantics when `seqlens_k` is absent: `flash_api.cpp:1265` computes `seqlen_k = kcache.size(1)` (non-paged), and the kernel's `BlockInfo` uses `params.seqlen_k` as `seqlen_k_cache` when `cu_seqlens_k == nullptr` (`upstream csrc/flash_attn/src/block_info.h:23`). I.e. **full cache is attended**.
  - Fork: `cache_seqlen = 0` → `total_seqlen_k = cache_seqlen + T_NEW` (`forward_kvcache.cu:85-89`); with `k=v=None` the result is an all-zero output and `lse = -inf`.
  - The repo locks this in: `tests/test_wrapper_cpu.py:691-699` (`test_kvcache_none_seqlens_is_empty`) asserts zeros and `lse<0`; `tests/mock_cuda.py:288` encodes `cache_seqlen=0`.
- **Fix**: `const int cache_seqlen = cache_seqlens ? cache_seqlens[cache_bidx] : (k_cache.size(1) - leftpad);` and update the test to assert full-cache attention (compare against the same case with explicit `cache_seqlens=torch.tensor([N])`). If the fork deliberately wants "None = empty", it must warn/raise in the Python wrapper and drop the "drop-in" wording.
- **Verify**: needs GPU to check attention content; the shape-level check (non-zero output for random cache + random q) can be done via the stub only after mirroring the new semantics in the mock.

### F-4 — Build isolation uses unpinned `torch` → ABI-mismatch wheels
- **Severity**: High for packaging; Medium in practice (shipped scripts use `--no-build-isolation`)
- **File:line**: `pyproject.toml:3-11` (`requires=["torch", ...]`) vs `:23-31` (`dependencies=["torch==2.6.0", ...]`); mitigations `scripts/build_cuda126.sh:41`, `utils/docs/install_cuda126.md:55,63`.
- **Claim**: "pin torch==2.6.0" (commit message) makes the build reproducible.
- **Evidence**: PEP 517/uv default isolation ignores `dependencies` and installs the newest `torch` into the build venv; torch C++ ABI is not stable across minor versions, so `import flash_attn_v100_cuda` against a 2.6.0 runtime can fail with undefined symbols. The shipped script `uv pip install --no-build-isolation --no-deps .` avoids this, so the hazard is for plain `pip install .` / bare `uv build` / CI defaults.
- **Fix**: pin build requires to `torch==2.6.0` (mirror `dependencies`), or relax the runtime pin and add an import-time `torch.__version__` check; keep `--no-build-isolation` in docs (already there).
- **Verify**: build a wheel in a scratch venv with isolation ON against a different torch, then import it in a 2.6.0 env.

### F-5 — Fake `flash_attn-2.8.3.dist-info` is not installed by PEP 517 wheels
- **Severity**: Med-High (breaks the metadata half of the "installs as flash-attn 2.8.3" story)
- **File:line**: `setup.py:157-167` (`InstallAttention.run`, writes `site.getsitepackages()[0]/flash_attn-2.8.3.dist-info`), `:222` (`cmdclass` includes `'install'`); `pyproject.toml:15` (`name = "flash_attn_v100"`).
- **Claim**: `flash_attn_i`/tools probing `flash-attn` metadata see version 2.8.3.
- **Evidence**: `pip install .` and `uv pip install .` build a wheel via `setuptools.build_meta`; the legacy `install` cmdclass is never invoked, so the dist-info is not part of the wheel. After the ADVERTISED install path (`scripts/build_cuda126.sh:41`), `importlib.metadata.version("flash-attn")` raises `PackageNotFoundError` and `pip show flash-attn` is empty. research/04 §3/B7 (lines ~205-235) already drafted the `install_lib` fix; `git status` shows `setup.py` untouched → **draft not applied**.
- **Fix**: apply the B7 draft (write the dist-info under `self.install_lib` inside an overridden `run()` so it lands in the wheel), then assert it in the build test.
- **Verify**: `uv build --wheel`, install into a scratch venv, `python -c "from importlib.metadata import version; print(version('flash-attn'))"` → must print `2.8.3` (cannot run here: env off-limits).

### F-6 — `S_dmask` shapes diverge from upstream for `return_attn_probs=True`
- **Severity**: Medium (consumer breakage under a "drop-in" banner)
- **File:line**: dense `kernel/forward.cu:397-401` (`{B,H_Q,M,N}`); varlen `kernel/forward_varlen.cu:527-531` (`{T_Q,H_Q,max_seqlen_k}`); upstream `flash_attn/flash_attn_interface.py:130` (`(batch,num_heads,round_multiple(seqlen_q,128),round_multiple(seqlen_k,128))`) and `:223-226` (4-D varlen `(batch,H,round128(max_q),round128(max_k))`). Tests: `tests/test_interface_behavior.py:101-112` assert the fork's `(2,3,5,5)`.
- **Claim**: `return_attn_probs` matches official FA2.
- **Evidence**: allocations above; both are gated `return_softmax && p_dropout > 0` (fork) vs `return_softmax and dropout_p > 0` (upstream `:845`) — the gating parity is verified good; only the returned shapes/rounding differ. Upstream pads to 128; the fork does not, and varlen is 3-D. Any code indexing `S_dmask[..., -1]`-style or expecting 4-D varlen breaks silently or errors.
- **Fix**: allocate padded shapes in both `.cu` files and return them (upstream also pads `seqlen_q`), or explicitly document the deviation and add a wrapper-level warning; stop the test from asserting the divergent shape as "parity".
- **Verify**: shape assertions can be checked after the fix without GPU; value-level comparison needs GPU.

### F-7 — Raw extension `varlen_fwd` arity differs from upstream
- **Severity**: Medium (raw-extension users only; the repo's own shim is unaffected)
- **File:line**: `kernel/api.cpp:21` (`m.def("varlen_fwd", &flash_attention_varlen_forward, ...)` with no `py::arg` defaults); `kernel/forward_varlen.cu:368-390` (`int num_splits = 0` is a **C++** default, which pybind11 does not honor without `py::arg("num_splits")=0`); upstream `flash_api.cpp:514-537` (21 params) and `flash_attn_interface.py:165-186` (21-arg call).
- **Claim**: "the fork's extension answers to both names" / raw drop-in via `flash_attn_2_cuda`.
- **Evidence**: `fwd` (13 args), `bwd` (19), `varlen_bwd` (24), `fwd_kvcache` (20) arities match upstream call sites (verified by counting the fork wrapper's call vs upstream call). Only `varlen_fwd` has an extra required argument (`num_splits`, 22nd). Note: this corrects the round-1 hypothesis that plain `fwd` differs.
- **Fix**: `m.def("varlen_fwd", ..., py::arg("num_splits") = 0)` (and/or keyword binding). One line in `kernel/api.cpp`.
- **Verify**: `python -c "import torch, flash_attn_2_cuda; print(flash_attn_2_cuda.varlen_fwd)"` then call with 21 upstream-style args (needs torch).

### F-8 — Empty-KV sentinel inconsistency: C++ `-inf` vs upstream/fork-Python `+inf`
- **Severity**: Medium (raw-ext/`torch.ops`; masked for public Python API)
- **File:line**: `kernel/forward.cu:405-409` (`softmax_lse.fill_(-inf)` when `N==0`); `kernel/forward_varlen.cu:534-537` (fills only when `zero_tensors`); upstream `flash_api.cpp:501-503` (`fill_(...infinity...)` = `+inf`); fork Python `flash_attn_v100/flash_attn_interface.py:78-85, 246-255` (+inf); test `tests/test_wrapper_cpu.py:696-698` asserts `lse<0`.
- **Claim**: "official seqlen_k==0 semantics: zeroed out, lse=+inf" (commit message) applies to the implementation.
- **Evidence**: the public Python path early-returns before the extension (hence public API is correct), but bare `flash_attn_v100_cuda.fwd(... N==0 ...)`, `flash_attn_2_cuda.fwd(...)`, and `torch.ops.flash_attn_v100.fwd` see `-inf`; for varlen, when `zero_tensors=false` the raw empty-KV result is uninitialized (nondeterministic garbage). The kvcache test bakes the wrong sign in.
- **Fix**: fill `+inf` in both C++ empty branches and make varlen zero/inf-fill unconditional (independent of `zero_tensors`), matching upstream's early-out; update the kvcache test to `lse>0`.
- **Verify**: raw-ext test with empty KV checks `isinf(lse) & (lse>0)`; static grep for the sentinel lines.

### F-9 — CPU acceptance suite overstates its coverage
- **Severity**: Medium (quality of the deliverable; several false-pass mechanisms)
- **File:line**: `tests/run_all_cpu.py:15-18,22-28`; `tests/test_extension_smoke.py:14-24` + `tests/_stub_cuda_backend.py:126-136`; `tests/test_interface_behavior.py:118-143,240-263`; `tests/test_wrapper_cpu.py:691-699`; `tests/mock_cuda.py:288`; `tests/gpu_verify.py:252-279,315`.
- **Claim**: "repository ships a GPU-free test suite covering FA2 API parity, wrapper semantics..." (README working-tree) and the commit's "Tests (GPU-free)" block.
- **Evidence**:
  - `run_all_cpu.py` runs only `test_*.py`; the strongest upstream-equivalence checker `tests/parity_check.py` is excluded (it is only reachable via `run_all.py --repo`, which reports SKIP/PARTIAL when torch is missing, `run_all.py:253-260`).
  - `test_extension_smoke.py` exits 0 with a SKIP when the extension will not import → appears as `PASS` in the runner summary; it cannot gate the build.
  - Empty-path tests exercise forward only (F-2); the kvcache test encodes fork-only semantics (F-3); dmask tests assert fork shapes (F-6).
  - `gpu_verify.py` covers dense forward/backward for a handful of shapes; dropout is checked for finiteness only (`:274-278`) and there is **no varlen-bwd dropout case and no paged/block_table case** (the near-miss `(32,128,448,4,2)` at `:315` would catch P1-2 for the dense causal case).
- **Fix**: make `run_all_cpu.py` also run `parity_check.py` (AST fallback when torch is missing); give the runner a distinct SKIP bucket instead of PASS for the smoke test; add the F-2 empty-backward tests; flip F-3/F-6 expectations to upstream-derived ones.
- **Verify**: run the suite in an env *without* torch and confirm the summary distinguishes SKIP from PASS; gift-wrap CI around the exit codes.

### F-10 — Missing upstream subpackages/symbols (documented, unresolved)
- **Severity**: Medium (silent downstream degradation)
- **File:line**: `flash_attn/` contains only `__init__.py`, `flash_attn_interface.py`, `bert_padding.py` (+`__pycache__`); no `layers/ modules/ ops/ utils/ models/`. Parity evidence: `parity_ast.py` → `exact=28 mismatch=0 missing=22 extra=6`; research/06 prior-art lines 94, 250 (vLLM/transformers probe `flash_attn.ops.*` and silently fall back; issue #29).
- **Claim**: "FlashAttention-2 API drop-in parity".
- **Evidence**: directory listing + parity checker output; `_flash_attn_forward`/`_flash_attn_backward` etc. also absent from the shim namespace (MISSING list).
- **Fix**: either ship minimal shims for the probed modules (rotary ops are pure Python upstream — cheap) or a `flash_attn/ops/__init__.py` that raises an explanatory ImportError; at minimum keep it loud in docs since "drop-in" is claimed.
- **Verify**: `python -c "import flash_attn.ops"`, `from flash_attn import layers` etc.; rerun parity_ast and shrink the MISSING list.

### F-11 — Commit-message claim audit
- **Severity**: Info
- **File:line**: `git show fdf8631` message; `parity_ast.py` / `parity_check_orch.py --ast` outputs; the findings above.
- **Claim-by-claim**:
  - "fix autograd arity bug (13→12, 18→17)": **accurate** — AST shows dense forward 12 params ↔ backward returns 12; varlen 17 ↔ 17; both parity checkers report `mismatch=0` on the required surface.
  - "build ... no-GPU": **falsified at this commit** (F-1); true only after `cfcb0cd`.
  - "drop-in parity": **overstated** (F-3, F-6, F-7, F-10; plus raw `-inf` F-8).
  - "CPU test suite": **present but with the gaps in F-9**.
  - "bert_padding.unpad_input returns used_seqlens_in_batch as 5th value": **accurate** (upstream 5-tuple parity verified).
- **Fix**: none code-wise; future commit messages should scope "parity" to a named reachable surface and "verified" to artifacts attached in-commit.
- **Verify**: re-run the two parity checkers after any further shim edit.

### F-12 — Cross-ref: top real-kernel risk is untouched and uncovered
- **Severity**: High (product risk; outside the commit diff)
- **File:line**: research/02 §6 P0-1 (`kernel/forward_varlen.cu:228-233` vs `kernel/backward_varlen.cu:254-260,485-491`), P1-2 (`include/softmax.h:92-104`; test skip at `test.py:144-145`), P1-3 (`kernel/forward_kvcache.cu:199-225`, `%256` gate at `:486`); corroboration and new dense-path dropout findings in research/08 (G-1: bwd philox components hardcoded → wrong dropout mask whenever `(row*N)%4 != 0`, i.e. **dense `flash_attn_func(dropout>0)` with `seqlen_k % 4 != 0`**; G-2 dmask 8B alignment; G-3 bwd ignores `seqused_k`).
- **Claim**: the change set improves parity and verification.
- **Evidence**: none of these paths is exercised by any CPU test (they are kernel-internal) and `gpu_verify.py` covers only dense fwd/bwd, a few shapes, dropout-finiteness (no varlen bwd, no paged, no RoPE position checks). P0-1/G-1 produce **silently wrong gradients**, the worst failure class.
- **Fix**: at least fix the two-line RNG stride/component mismatch before advertising training support; add `gpu_verify` cases: varlen dropout bwd with ragged seqlens (fixed generator seed + dmask-based reference), dense dropout with `N%4!=0`, and paged KV with shuffled `block_table`.
- **Verify**: needs a real V100; the fixes themselves are static/verifiable by the reference construction in research/08 §2.

## 3. Verified good (adversarial checks that failed to find problems)

1. **Autograd arity fix is real**: dense `(12,12)`, varlen `(17,17)`; `parity_ast.py` → `exact=28 mismatch=0 missing=22 extra=6 required_failures=0`; `parity_check_orch.py --ast` → `exact=17 mismatch=0 missing=0` for the required set.
2. **`bert_padding.py`**: `unpad_input` 5th return value (`used_seqlens_in_batch`) matches upstream; file is near-identical to upstream (docstring-only diffs). `test_bert_padding.py` is a real oracle.
3. **`return_softmax` gating**: `return_softmax and dropout_p > 0.0` in both fork paths (`flash_attn_interface.py:109,282`) matches upstream (`flash_attn_interface.py:845`); dropout==0 → empty dmask matches upstream.
4. **Empty-KV public API semantics**: dense `N==0` → zeros + `+inf` lse, varlen `max_seqlen_k==0` → zeros + `+inf`; matches upstream `flash_api.cpp:501-503` and the documented comment. (The C++ raw layer does not — F-8.)
5. **Intractable-looking checks that pass**: `cache_seqlens` int expansion to `(B,) int32`, `maybe_contiguous` handling, kvcache stride asserts — all match upstream conventions.
6. **D=16 dispatch**: `include/forward.h` config table and all five kernel dispatch switches include `case 16`; consistent with the V100 claims (verified in earlier session by grep).
7. **`run_all.py` honesty**: reports SKIP/PARTIAL and states "AST fallback, runtime signatures NOT compared" when torch is missing (`run_all.py:253-260`). The overstatement lives in `run_all_cpu.py`/README, not here (F-9).
8. **Module-naming design**: `flash_attn/flash_attn_interface.py:23-27` prefers `flash_attn_2_cuda` (official name) with `flash_attn_v100_cuda` fallback; the double `PYBIND11_MODULE` in one TU is the intended mechanism for the symlink alias. Plausible and consistent; duplicate-load/TORCH_LIBRARY double registration could not be observed (see §4) but dlopen's dev/inode dedup makes it likely fine.
9. **`round_multiple` shim helper** (working tree) matches upstream's helper semantics (`upstream:49`).
10. **Preflight compile (untracked, post-`cfcb0cd`)**: 5/6 TUs compile `EXIT=0` with nvcc 12.6/sm_70. Good evidence the build works after F-1's fix; it is *not* evidence for `fdf8631`.

## 4. Recommended minimal fixes (in order)

1. **Guard empty-path backward** (F-2): short-circuit in both `backward`s when `ctx.saved_tensors` is empty; add the 3 missing CPU tests. ~10 lines + tests.
2. **Fix kvcache `None` semantics** (F-3): one ternary in `kernel/forward_kvcache.cu:85` + test update. Decide and document if deliberate.
3. **Pin build `torch`** (F-4): `pyproject.toml:7` → `torch==2.6.0` (or remove runtime pin). One line.
4. **Apply research/04 B7** (F-5): `install_lib`-scoped dist-info write; verify via wheel install.
5. **`py::arg("num_splits") = 0`** in `kernel/api.cpp` (F-7). One line.
6. **Unify empty-KV sentinel to `+inf` in C++** (F-8) and make varlen fill unconditional. ~4 lines.
7. **`S_dmask` padded shapes** (F-6) or an explicit documented deviation; stop asserting fork shapes as parity.
8. **Test-runner truthfulness** (F-9): run `parity_check.py` from `run_all_cpu.py`; separate SKIP from PASS for the smoke test.
9. **Loud `flash_attn.ops` gap** (F-10): package-level ImportError hook or minimal shims; update README's "drop-in" wording meanwhile.

## 5. What could NOT be verified without GPU/torch (and current tree caveats)

- **No runtime whatsoever**: torch is not importable here and `/mnt/envs/v100` is off-limits (install in progress). Therefore: stub/mock test execution, raw-extension import, pybind argument acceptance (F-7), wheel metadata (F-5), duplicate-module/TORCH_LIBRARY behavior, and any value-level comparison against upstream are static-only.
- **No GPU**: every kernel-numeric claim (P0-1/G-1 dropout RNG, P1-2 masked rows, P1-3 paged reads, dmask values, kvcache attention content in F-3) is untouched by this audit; cross-check research/02 and research/08 for the static derivations.
- **No full build/link**: only the untracked per-TU preflight logs exist (5/6 TUs, post-`cfcb0cd`). No `.so`, no import test. The `fdf8631` compile claim (F-1) is falsified only by the follow-up commit message + header grep, not by a reproduced build of `fdf8631` itself (out of scope of read-only).
- **Moving tree**: other agents edited `README.md`, `flash_attn*/flash_attn_interface.py`, `research/03*`, `research/05*` during this review, and HEAD advanced to `cfcb0cd`. Re-verify line numbers before applying any fix; the reviewed snapshot is `cfcb0cd` + the listed dirty files.

Line count: ~13 findings, 10 verified-good, 9 fixes. This report intentionally contains no claims about kernel numerics beyond what research/02 and research/08 already prove statically.
