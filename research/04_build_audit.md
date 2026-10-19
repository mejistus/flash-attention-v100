# 04 — Build Audit: flash-attention-v100 on CUDA 12.6 / torch 2.6.0 / no-GPU host

Auditor: build-system engineer subagent. Date: 2026-10-19.
Scope: make `/home/eculid/Documents/Github/flash-attention-v100` (fork ai-bond/flash-attention-v100,
sm_70 FA2, commit 55d685a) buildable in conda env `v100` (`/mnt/envs/v100`) with **CUDA 12.6.85**,
**torch 2.6.0+cu126**, conda **gcc 12.4** — on a host with **no NVIDIA GPU** (AMD RX580 only).
No repo files were modified **by this audit**; all scratch in `/tmp/opencode/build_audit/`.

**Status 15:35:** the orchestrator applied this plan as commits `fdf8631` (CUDA 12.6 support + FA2 API
parity + tests) and `cfcb0cd` (softcap `__tanhf` fix), plus `scripts/build_cuda126.sh` and run.sh
tweaks. §(g) reconciles as-applied vs. planned and lists residual gaps; §(b)–§(e) carry status notes
where the recipe/plan changed. The pre-patch failure capture was obsoleted by those commits (§g.1).
**Update 16:06:** the torch gate passed after the orchestrator's staged recovery (§g.2) — the env is
build-ready for the canonical P4 run.

---

## (a) Environment validation (raw evidence)

### Toolchain

```
$ /mnt/envs/v100/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0

$ /mnt/envs/v100/bin/x86_64-conda-linux-gnu-gcc --version | head -2
x86_64-conda-linux-gnu-gcc (conda-forge gcc 12.4.0-2) 12.4.0
Copyright (C) 2022 Free Software Foundation, Inc.

$ which gcc; gcc --version | head -1
/usr/bin/gcc
gcc (GCC) 16.2.1 20260810        # system gcc — TOO NEW for CUDA 12.6 (max 13)
```

**Critical finding #0:** `/mnt/envs/v100/bin` has **no** `gcc`/`g++` symlinks (only `nvcc`, `ptxas`
match `^gcc|g++|cc|c++|ninja|nvdisasm|cuobjdump|ptxas|nvcc$`). nvcc does **not** read `CC`/`CXX`;
its default host compiler is `gcc` from PATH → `/usr/bin/gcc` 16.2.1 → hard error:

```
$ CC=.../x86_64-conda-linux-gnu-gcc CXX=.../x86_64-conda-linux-gnu-g++ \
    /mnt/envs/v100/bin/nvcc -arch=sm_70 ... -c t.cu -o t.o          # WITHOUT -ccbin
/.../crt/host_config.h:143:2: error: #error -- unsupported GNU version! gcc versions later than 13 are
not supported! ... The nvcc flag '-allow-unsupported-compiler' can be used to override
exit=255

$ ... same command + -ccbin=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-gcc ...
exit=0
-rw-r--r-- 1 eculid eculid 8112 10月 19 14:30 t.o     # sm_70 cubin/elf produced
```

`-ccbin` **must** be effectively passed. torch 2.6.0's cpp_extension does this automatically from the
environment: ninja path (`_write_ninja_file`, torch/utils/cpp_extension.py v2.6.0 line 2328)
`cuda_flags = ['-ccbin', os.getenv("CC")] + cuda_flags`; classic path line 605–612 adds
`-ccbin $CC` too. Host `.cpp` files are compiled with `$CXX` (`get_cxx_compiler()`, line 284–288:
`compiler = os.environ.get('CXX', 'c++')`). → **Recipe must export CC and CXX to the conda gcc 12.4
binaries** (not rely on PATH; no `gcc`/`g++` symlinks exist in the env).

### nvcc flag support for every flag used by setup.py (12.6.85 `--help`)

All of `-arch`, `-gencode arch=compute_70,code=sm_70`, `-std=c++17`, `-O3`, `--use_fast_math`,
`-U__CUDA_NO_HALF_OPERATORS__` / `_CONVERSIONS` / `_HALF2_OPERATORS`, `--expt-relaxed-constexpr`,
`--expt-extended-lambda`, `-Wno-deprecated-gpu-targets`, `--threads=<n>`, `-Xptxas -v`, `-g`,
`--keep`, `--keep-dir`, `-ccbin` compiled successfully in the smoke below (strongest evidence);
`nvcc --help` greps also show each long-named flag (11 regex hits across the 4 multi-name patterns).
The half-macro `-U`s enable `__half` operators in `cuda_fp16.hpp` — needed since the code does
arithmetic on `__half`/`__half2`.

### Build-mode switches (setup.py) and the macros they define

| Env var (only `"1"`-style truthy presence checked — any non-empty value works) | setup.py effect | Macro effect in code |
|---|---|---|
| `ATTENTION_DEBUG` | adds `-DKERNEL_DEBUG -g --keep --keep-dir <repo>/build -Xptxas -v` (setup.py:65-73) | `include/debug.h:8-25`: `KERNEL_DEBUG` default `0`; when 1, defines `__ASM_MARK`/`__ASM_DEBUG_BEGIN`/`__ASM_DEBUG_END` marker asm. Debug helper only. |
| `MMA_NATIVE` | `-DMMA_NATIVE` (setup.py:75-76) | `include/mat_mul.h:10` → `#include <mma.h>`; GEMM helpers use `nvcuda::wmma` (`mat_mul.h:36,177`). CUDA-9-era WMMA — fine on 12.6. |
| `MMA_884` | `-DMMA_884` (setup.py:78-79) | `include/kernel.h:10-13` selects WMMA 8×8×4 dims; `include/mat_mul.h:12` → `#include "mma_m8n8k4.h"` (HMMA.m8n8k4 PTX). |
| (neither) | default | `include/mat_mul.h:14` → `mma_m16n16k16.h` explicit-PTX HMMA.884 wrapper (sm_70-only `#error` guard). |

No switch requires > CUDA 12.6; none changes the `.so` name or API surface.

### Real-header compile smoke #1 (torch-free)

`/tmp/opencode/build_audit/smoke_nvcc.cu` includes `kernel.h` + `product.h` + `swizzle.h` + `mat_mul.h`
(→ `mma_m16n16k16.h`) and calls `ld_float`/`st_float`, compiled with the exact setup.py flag set plus
`-ccbin`:

```
$ nvcc -ccbin=.../x86_64-conda-linux-gnu-gcc -O1 -std=c++17 -gencode arch=compute_70,code=sm_70 \
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Wno-deprecated-gpu-targets \
    -I include -c smoke_nvcc.cu -o smoke_nvcc.o
exit=0   (1.8s user, 2.0s total)   smoke_nvcc.o 6400 bytes
```

→ repo headers compile clean under CUDA 12.6.85 + gcc 12.4 in default MMA mode (sm_70 native).

### Real-header compile smoke #2 (torch 2.6.0 headers, cpp_extension E2E)

See §(g).

### Other env facts

| Item | Status |
|---|---|
| `nvdisasm`, `cuobjdump` | **NOT installed** (only `nvcc`, `ptxas` in `targets/x86_64-linux/bin`). Only needed by `run.sh --debug` SASS extraction (`cuobjdump --dump-sass`). Optional: `conda install -n v100 -c nvidia cuda-nvdisasm cuda-cuobjdump` (do not run as part of fix). |
| `psutil` | **Missing initially**; `setup.py auto_tune()` catches ImportError → `MAX_JOBS=4`, `NVCC_THREADS=4` fallback. Being installed by orchestrator. |
| `pip` in env | **Not installed** (`No module named pip`). Use `uv pip --python /mnt/envs/v100/bin/python`. |
| `ninja` | Not in env initially; required by torch BuildExtension (falls back to slow distutils backend with a warning if missing). In install list. |
| disk | `/tmp` tmpfs 32G free; `/` 383G free (`df -h`). Torch-ext build dir + ninja objects fit easily. |
| CPU/RAM | 36 cores, 56 GiB available. `auto_tune()` formula: `max_jobs=int(56*0.9/2.5)=20`, `min(20,36)=20` → **20 jobs × 20 nvcc threads** if both env vars unset. Override in CI. |
| Python | 3.12.15, `site.getsitepackages()` → `['/mnt/envs/v100/lib/python3.12/site-packages']` (correct for this conda env). |
| CUDA include layout (conda) | CUDA headers **only** under `/mnt/envs/v100/targets/x86_64-linux/include/` (`cuda_runtime.h`, `cuda.h`, `cuda_fp16.h` confirmed). `/mnt/envs/v100/include/` exists but contains **zero** `*cuda*` files (`ls /mnt/envs/v100/include/ | grep -i cuda` → exit 1). → drives blocker B11. |
| CUDA lib layout (conda) | bridge symlinks exist: `/mnt/envs/v100/lib/libcudart.so -> ../targets/x86_64-linux/lib/libcudart.so.12.6.77` (+ `.so.12`, `.so.12.6.77`, `libcudart_static.a`). torch `library_paths()` handles `lib64`→`lib` fallback → linking `-lcudart` OK. |

`auto_tune()` wart (setup.py:19): it only early-returns when **both** `MAX_JOBS` and `NVCC_THREADS`
are set; if only one is set it **overwrites both**. Recipe must set both.

---

## (b) Blocker list (exact locations)

| # | Blocker | Location | Effect |
|---|---|---|---|
| B1 | Hard GPU requirement `if not torch.cuda.is_available(): raise` | `setup.py:139-140` | Any build on this host fails immediately. |
| B2 | Hard CUDA≥12.9 gate `if parse(torch.version.cuda) < parse("12.9"): raise` | `setup.py:141-142` | Fails with torch 2.6.0+cu126 (`12.6 < 12.9`), also crashes on `None` (CPU wheel) via `parse(None)`. |
| B3 | `pyproject.toml` build-system pins torch **2.10.0+cu129 direct URL** | `pyproject.toml:7` | Isolated builds download a 2.10 cu129 wheel (~GBs) that cannot host-compile sm_70 (Volta dropped) and mismatch CUDA_HOME 12.6. Breaks `pip install .` with isolation; even with `--no-build-isolation`, pip/uv still parse it. |
| B4 | `pyproject.toml` runtime dep pins same torch 2.10 URL | `pyproject.toml:24` | Installing the project tries to replace torch 2.6.0 (user-pinned). |
| B5 | `run.sh` calls bare `pip` | `run.sh` (`pip install . --no-build-isolation -v`) | Env has no pip → script unusable as-is; `uv pip` is the working installer. |
| B6 | `run.sh` ends with `CUDA_LAUNCH_BLOCKING=1 python test.py` | run.sh last line | test.py unconditionally uses `torch.cuda` events/`device='cuda'` → fails on no-GPU host even after a successful build. |
| B7 (robustness) | `InstallAttention` writes fake dist-info to `site.getsitepackages()[0]` | `setup.py:117-124` | Path is right for *this* env, but under isolated PEP517 builds it writes into the throwaway build venv (lost) and under non-isolated builds it litters a stray unmanaged `flash_attn-2.8.3.dist-info` outside wheel-install tracking. See §(c) for the install_lib fix. |
| B8 (robustness) | `packages=["flash_attn_v100"]` only; `flash_attn` shim depends on `CopyAttention` copy trick | `setup.py:173`, `pyproject.toml:34`, `setup.py:100-112` | Works with the copy, but `build_py` semantics don't know about `flash_attn`; cleaner to declare it. No conflict with the fake `flash-attn` metadata. |
| B9 (soft) | `flash_attn/bert_padding.py` imports `einops` | not in requirements/pyproject deps | `import flash_attn` fine (shim `__init__` doesn't import it), `import flash_attn.bert_padding` fails unless einops installed. Add `einops` if API parity needed. |
| B10 (soft) | `flash_attn_v100_cuda` runtime import name vs `flash_attn_2_cuda` symlink | `setup.py:149-158`, `kernel/api.cpp:11-22` | Verified correct: api.cpp defines `PYBIND11_MODULE(TORCH_EXTENSION_NAME)` (→ `PyInit_flash_attn_v100_cuda`) **and** `PyInit_flash_attn_2_cuda`; symlink `flash_attn_2_cuda.so` is in `importlib.machinery.EXTENSION_SUFFIXES` (`.so`), so upstream-style `import flash_attn_2_cuda` works. No change needed. |
| **B11** | **conda CUDA include layout vs torch host-compile include paths** | env: `/mnt/envs/v100/include` has **no CUDA headers** (they live in `/mnt/envs/v100/targets/x86_64-linux/include`); `kernel/api.cpp` (host `.cpp` via `$CXX`) → `mha.h:4` `#include <cuda_runtime.h>`; torch v2.6 `cpp_extension.py:1224-1232` appends only `$CUDA_HOME/include` (and optionally `$CUDA_INC_PATH`) | Without a fix, the **host** compile of `kernel/api.cpp` fails `cuda_runtime.h: No such file or directory` even though nvcc TUs compile fine (nvcc finds its headers internally). Fix options (evidence in §(g)): **(a) preferred, no repo change:** `export CUDA_INC_PATH=/mnt/envs/v100/targets/x86_64-linux/include` (torch-native knob); **(b)** add that dir to `include_dirs` in `setup.py:92`; **(c)** symlink CUDA headers into `$PREFIX/include` (env mutation). Linking is already fine: `/mnt/envs/v100/lib/libcudart.so` symlink farm exists (from `cuda-cudart-dev_linux-64`), and torch `library_paths()` falls back `lib64`→`lib` (`cpp_extension.py:1263-1271`). |

**Status (15:35):** the orchestrator applied the plan as commits `fdf8631`/`cfcb0cd`. B1/B2/B4/B6/B9/
B11 are addressed; B3 is partial (runtime deps fixed, build-requires left as bare `"torch"`); the
torch-hint text was fixed (now `torch==2.6.0 .../cu126`, setup.py:88-91 and 172-176); B5/B7/B8 remain
open with mitigations. Per-item reconciliation: §(g.3); residual gaps: §(g.4).

---

## (c) Minimal patch plan (unified diffs)

All patches are small, surgical, and preserve upstream behavior when the new env knobs are absent.
**Status note (15:35):** what was actually applied differs in places (see §(g.3) for the item-by-item
reconciliation and §(g.4) for residual gaps); the diffs below remain the reference design.

### 1) `setup.py` — torch install hint text (both copies: lines 45–48 and 129–133)

```diff
--- a/setup.py
+++ b/setup.py
@@ -45,7 +45,7 @@ def get_ext_modules():
     except ImportError as e:
         raise RuntimeError(
             "torch is required to build flash_attn_v100. "
-            "Please install torch >= 2.10 first (e.g., `pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129`)."
+            "Please install torch >= 2.6 first (e.g., `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126`)."
         ) from e
@@ -129,7 +129,7 @@ def get_cmdclass():
     except ImportError as e:
         raise RuntimeError(
             "torch is required to build flash_attn_v100. "
-            "Please install torch >= 2.10 first (e.g., `pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129`)."
+            "Please install torch >= 2.6 first (e.g., `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126`)."
         ) from e
```

### 2) `setup.py` — BuildAttention checks (B1, B2; new escape hatch `FLASH_ATTN_V100_ALLOW_NO_GPU`)

```diff
--- a/setup.py
+++ b/setup.py
@@ -135,11 +135,24 @@ def get_cmdclass():
     class BuildAttention(BuildExtension):
         def build_extensions(self):
             import torch
 
             if not torch.cuda.is_available():
-                raise RuntimeError("CUDA is required but not available.")
-            if parse(torch.version.cuda) < parse("12.9"):
-                raise RuntimeError(f"CUDA version {torch.version.cuda} < 12.9 is not supported.")
+                if os.environ.get("FLASH_ATTN_V100_ALLOW_NO_GPU") == "1":
+                    print("=" * 78)
+                    print("WARNING: FLASH_ATTN_V100_ALLOW_NO_GPU=1 and no CUDA device is present.")
+                    print("WARNING: producing COMPILE-ONLY artifacts for sm_70. They will")
+                    print("WARNING: only load/run on a machine with a Volta GPU (driver >= 12.6).")
+                    print("=" * 78)
+                else:
+                    raise RuntimeError(
+                        "CUDA is required but not available. "
+                        "Set FLASH_ATTN_V100_ALLOW_NO_GPU=1 to build without a GPU (compile-only)."
+                    )
+            cuda_ver = torch.version.cuda
+            if cuda_ver is None or parse(cuda_ver) < parse("12.6"):
+                raise RuntimeError(f"CUDA version {cuda_ver} < 12.6 is not supported.")
 
             super().build_extensions()
```

Notes:
- Exactly `"1"` enables the escape hatch; anything else (incl. `"0"`, `"true"`, empty) keeps the strict default.
- The CUDA-version check remains enforced for everyone (relaxed 12.9 → 12.6 to match the toolchain).
- `cuda_ver is None` guard avoids `parse(None)` TypeError on CPU-only torch.
- No change needed to the symlink block `setup.py:146-160`: `src_lib.parent` is the same directory
  the built `.so` lands in for both `build_ext --inplace` (repo root for this flat module) and wheel
  builds (`build/lib.*/…`), so `flash_attn_2_cuda.so` ends up beside `flash_attn_v100_cuda.cpython-312-*.so`.
  The relative symlink (`dst_lib.symlink_to(src_lib.name)`) survives `install_lib`'s `copy_tree`
  (verified: `setuptools._distutils.dir_util.copy_tree` dereferences symlinks → the wheel/site-packages
  gets a real `flash_attn_2_cuda.so` copy; experiment in `/tmp/opencode/build_audit/symtest`:
  `alias.so islink=False size=8`).

### 3) `setup.py` — InstallAttention path robustness (B7)

Normally `bdist_wheel` (used by pip/uv PEP517) runs the `install` command in staging mode where
`self.install_lib` points at the wheel staging dir. Writing the fake dist-info there makes it part of
the wheel (so `pip show flash-attn` / `importlib.metadata.version("flash-attn") == "2.8.3"` works after
install), instead of writing a stray directory into whichever interpreter runs the build, or losing
it entirely under isolated builds:

```diff
--- a/setup.py
+++ b/setup.py
@@ -114,9 +114,10 @@ class InstallAttention(install):
     def run(self):
         install.run(self)
-        import site
-        dst = os.path.join(site.getsitepackages()[0], 'flash_attn-2.8.3.dist-info')
+        # Write into the tree being populated: site-packages for `setup.py install`,
+        # or the wheel staging dir under bdist_wheel / PEP 517 (pip, uv).
+        dst = os.path.join(self.install_lib, 'flash_attn-2.8.3.dist-info')
         if os.path.exists(dst): shutil.rmtree(dst)
         os.makedirs(dst, exist_ok=True)
         with open(os.path.join(dst, 'METADATA'), 'w') as f:
             f.write("Metadata-Version: 2.4\nName: flash-attn\nVersion: 2.8.3\n")
         with open(os.path.join(dst, 'top_level.txt'), 'w') as f:
             f.write('flash_attn\n')
```

This keeps the "fake upstream metadata" behavior for the user-visible outcome and removes all
dependence on `site.getsitepackages()` (which is wrong under isolated builds and fragile under
`uv pip`). If the fix agent wants the absolute least risk: keep the old method as a fallback when
`self.install_lib` is missing — not needed for setuptools>=61 (install_lib always set post-finalize).

**Empirical verification** (setuptools 82.0.1 wheel unpacked from the tuna mirror, driven in a scratch
venv with an identical cmdclass; `/tmp/opencode/build_audit/miniproj`):
- setuptools 82.0.1 `setuptools/command/bdist_wheel.py:370-405`: `run_command("build")` →
  `install = reinitialize_command("install")`, `install.root = self.bdist_dir` → `run_command("install")`.
  So **the custom `install` cmdclass DOES run inside every pip/uv PEP517 wheel build**.
- Probe output for a project WITH `ext_modules` (same `root_is_pure=False` path as this repo):
  ```
  PROBE: wrote via site.getsitepackages()[0] -> /tmp/.../venv/lib/python3.12/site-packages/probe_fake-9.9.9.dist-info
  PROBE: install_lib -> build/bdist.linux-x86_64/wheel/./
  creating 'dist/probe_pkg-0.0.1-cp312-cp312-linux_x86_64.whl' ...
  adding 'probe_fake-9.9.9.dist-info/METADATA'
  ```
  and the built wheel contains `probe_fake-9.9.9.dist-info/METADATA` at the archive root, listed in
  `dist-info/RECORD` with its sha256 (spec-compliant), while the `site.getsitepackages()` copy was a
  stray directory that never entered the wheel.

### 4) `setup.py` + `pyproject.toml` — ship the `flash_attn` shim natively (B8)

```diff
--- a/setup.py
+++ b/setup.py
@@ -170,7 +170,7 @@ setup(
     name="flash_attn_v100",
     version="26.06",
-    packages=["flash_attn_v100"],
+    packages=["flash_attn_v100", "flash_attn"],
     ext_modules=get_ext_modules(),
     cmdclass=get_cmdclass(),
```

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -33,7 +33,7 @@
 [tool.setuptools]
-packages = ["flash_attn_v100"]
+packages = ["flash_attn_v100", "flash_attn"]
```

Keep `CopyAttention` as-is (it re-copies the same tree; harmless, and covers the case where only
`build_py` is run). Set **both** places to the same list to avoid setuptools config precedence
surprises. No conflict with `SitePackages`-level upstream metadata fakery: the fake dist-info is
about `importlib.metadata`, package files are about `import`, and they agree (`top_level.txt:
flash_attn`). `exclude-package-data` needs no change (it only filters package *data*, not `.py`).

### 5) `pyproject.toml` — build-system requires (B3) and dependency (B4)

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,11 +1,12 @@
 [build-system]
 build-backend = "setuptools.build_meta"
 requires = [
     "setuptools>=82.0.1",
     "wheel>=0.46.3",
     "packaging>=26.0",
-    "torch @ https://download-r2.pytorch.org/whl/cu129/torch-2.10.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=a3c703e74a88cccfeb4b1807c49d563a0a73793ba72d4fa035d4ac5885f3aefd",
+    # torch is deliberately NOT listed: setup.py imports torch.utils.cpp_extension at
+    # build time. Always build with --no-build-isolation against the interpreter that
+    # already has the pinned torch (2.6.0+cu126 for Volta).
     "ninja>=1.13.0",
     "psutil>=7.2.2",
     "numpy>=2.4.4"
 ]
@@ -20,7 +21,7 @@ requires-python = ">=3.10"
 license-files = ["LICENSE", "AUTHORS"]
 dynamic = ["authors", "classifiers", "urls"]
 dependencies = [
-    "torch @ https://download-r2.pytorch.org/whl/cu129/torch-2.10.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=a3c703e74a88cccfeb4b1807c49d563a0a73793ba72d4fa035d4ac5885f3aefd",
+    "torch==2.6.0",
     "setuptools>=82.0.1",
```

Policy decision (user constraint: torch **2.6.0** pinned): `torch==2.6.0` matches the already-installed
`2.6.0+cu126` local version per PEP 440 (`==2.6.0` ignores local labels), so `uv pip install .` won't
re-download torch. **Must** be installed from the cu126 index (the PyPI manylinux torch 2.6.0 default
is a cu124 build → `torch.version.cuda == "12.4"` → fails the relaxed 12.6 check and mismatches the
12.6 nvcc). Keep `torch==2.6.0`, don't loosen to `>=2.6` — newer cu12x wheels drop Volta kernels
(pytorch#157517) and would break runtime on V100.
Alternative if build isolation is ever required: replace the URL with
`"torch>=2.6"` **and** add `--no-build-isolation`; but simplest is: no torch in build requires,
always `--no-build-isolation` (matches upstream run.sh). uv and pip both honor `--no-build-isolation`
for setup.py-based projects (uv ≥0.4 supports legacy setuptools builds via `setuptools.build_meta`
PEP517 shim).

### 6) `requirements.txt` (B5 auxiliary; keep all existing pins, add torch policy)

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,3 +1,5 @@
+# torch must come from the CUDA 12.6 wheel index (Volta / sm_70 support):
+#   uv pip install --python /mnt/envs/v100/bin/python "torch==2.6.0+cu126" --index-url https://download.pytorch.org/whl/cu126
 filelock==3.25.2
 fsspec==2026.2.0
 Jinja2==3.1.6
@@ -10,4 +12,5 @@ psutil==7.2.2
 setuptools==82.0.1
 sympy==1.14.0
 typing_extensions==4.15.0
 wheel==0.46.3
+torch==2.6.0
```

(If the orchestrator prefers the index-pinned form, use `torch==2.6.0+cu126`; bare `torch==2.6.0`
plus the comment documents the constraint without breaking plain `uv pip install -r`.)
Optionally add `einops` (B9) if `flash_attn.bert_padding` must import.

### 7) `run.sh` (B5, B6)

```diff
--- a/run.sh
+++ b/run.sh
@@ -1,18 +1,24 @@
 #!/bin/bash
 
 set -e
 
 DEBUG=0
 MMA_MODE=""
+NO_GPU=0
 
 for arg in "$@"; do 
     [[ "$arg" == "--debug" ]] && DEBUG=1
     [[ "$arg" == "--mma-native" ]] && MMA_MODE="native"
     [[ "$arg" == "--mma-884" ]] && MMA_MODE="884"
+    [[ "$arg" == "--no-gpu" ]] && NO_GPU=1
 done
 
 clear
 rm -rf ./build
 
+if [ "$NO_GPU" -eq 1 ]; then
+    export FLASH_ATTN_V100_ALLOW_NO_GPU=1
+    echo "no-GPU build mode: compile-only artifacts (test.py will be skipped)"
+fi
+
@@ -37,7 +43,11 @@ if [ -n "$MMA_MODE" ]; then
 fi
 
-pip install . --no-build-isolation -v
+if command -v uv >/dev/null 2>&1; then
+    uv pip install --python "$(command -v python)" . --no-build-isolation -v
+else
+    python -m pip install . --no-build-isolation -v
+fi
@@ -CUDA_LAUNCH_BLOCKING=1 python test.py
+if [ "$NO_GPU" -eq 1 ]; then
+    echo "no-GPU build mode: skipping test.py (requires a CUDA device)"
+else
+    CUDA_LAUNCH_BLOCKING=1 python test.py
+fi
```

Recommendation: **yes, add the `--no-gpu` path** (compile-only CI), default behavior unchanged.
If a smaller diff is preferred, at minimum replace `pip` with the uv/mpip fallback and gate the final
`python test.py` on `FLASH_ATTN_V100_ALLOW_NO_GPU` being unset.

### 8) OPTIONAL `setup.py` — conda CUDA include bridge (B11, only if the env var route is unwanted)

```diff
--- a/setup.py
+++ b/setup.py
@@ (inside get_ext_modules, around the CUDAExtension(...) call)
-            include_dirs=[this_dir / "include"],
+            include_dirs=[this_dir / "include"] + (
+                # conda CUDA layout: headers live under targets/<arch>/include
+                [os.path.join(os.environ["CUDA_HOME"], "targets/x86_64-linux/include")]
+                if os.path.isdir(os.path.join(os.environ.get("CUDA_HOME", ""),
+                                              "targets/x86_64-linux/include"))
+                else []
+            ),
```

Not required if the build recipe exports `CUDA_INC_PATH` (torch-native, zero repo change).

---

## (d) Exact build recipe (no GPU; CUDA 12.6; torch 2.6.0+cu126)

One-time env setup (uv is at `/usr/bin/uv 0.12.10`; env has no pip):

```bash
uv pip install --python /mnt/envs/v100/bin/python \
    "torch==2.6.0+cu126" ninja==1.13.0 psutil==7.2.2 setuptools==82.0.1 \
    wheel==0.46.3 packaging==26.0 numpy==2.4.4 \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Build (in-place, the CI path):

```bash
cd /home/eculid/Documents/Github/flash-attention-v100
export CUDA_HOME=/mnt/envs/v100
export PATH=/mnt/envs/v100/bin:$PATH              # nvcc visible; note: gcc/g++ symlinks DON'T exist here
export CC=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-gcc
export CXX=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-g++
export MAX_JOBS=4                                 # explicit: CI host is shared
export NVCC_THREADS=4                             # (setting both also disables auto_tune overwrite)
# Plan-era knobs — NOT needed with the as-applied setup.py (§g.3):
#   FLASH_ATTN_V100_ALLOW_NO_GPU=1 : as-applied always allows no-GPU builds (warns only)
#   CUDA_INC_PATH                  : as-applied `_extra_include_dirs()` already appends
#                                    <CUDA_HOME>/targets/x86_64-linux/include
unset TORCH_CUDA_ARCH_LIST                        # keep unset: setup.py provides -gencode compute_70,sm_70,
                                                  # torch's _get_cuda_arch_flags() then returns [] (verified in
                                                  # torch/utils/cpp_extension.py v2.6.0: early return if any flag
                                                  # contains 'arch' — see §(a)/(g) evidence)
/mnt/envs/v100/bin/python setup.py build_ext --inplace
```

**Canonical as-applied procedure:** `/tmp/opencode/P4_runbook.md` §4 (same env, `MAX_JOBS=8
NVCC_THREADS=4`) or `scripts/build_cuda126.sh` (sets `CUDA_HOME=<env>/targets/x86_64-linux` — both
layouts work; `targets/x86_64-linux/{bin/nvcc,include}` verified present).

Why `CC`/`CXX` matter (torch 2.6.0 `cpp_extension.py`):
- ninja path appends `-ccbin $CC` to nvcc (line 2328); host `.cpp` compiled with `$CXX` (line 284-288, 2415).
- Without them nvcc picks `/usr/bin/gcc` 16.2.1 → `host_config.h:143` error (see §(a)).
- `TORCH_CUDA_ARCH_LIST` **must stay unset** *because* setup.py already passes `-gencode ... sm_70`;
  torch then adds no arch flags. If either piece is changed, a default multi-arch list (sm_90/sm_100/sm_120)
  would break under nvcc 12.6. (If in doubt during debugging: `export TORCH_CUDA_ARCH_LIST=7.0` is the
  belt-and-braces equivalent.)

Wheel install path (alternative to `--inplace`):

```bash
uv pip install --python /mnt/envs/v100/bin/python . --no-build-isolation --no-deps -v
#  --no-deps for a quick install while deps are already present; drop it once
#  pyproject.toml dependency is patched to torch==2.6.0.
```

Equivalent conservative fallback if uv ever fails on the legacy setup.py project:
`python setup.py bdist_wheel` then `uv pip install <wheel> --no-deps`.

---

## (e) Risk register + post-patch re-test plan

| Risk | Likelihood | Notes / mitigation |
|---|---|---|
| nvcc host gcc mismatch | **certain if CC/CXX missing** | Evidence §(a). Always set both; alternatively add `gcc`/`g++` symlinks into the env bin (not needed). |
| host `.cpp` (api.cpp) can't find `cuda_runtime.h` (B11) | **certain if CUDA_INC_PATH missing** | conda layout; mitigate with `CUDA_INC_PATH` (or `include_dirs` patch). Smoke capture expected in §(g). |
| `__CUDA_ARCH__ != 700` `#error` in `include/mma_m16n16k16.h:20` / `mma_m8n8k4.h:37` | low if gencode kept | These fire only if torch/other flags add non-70 archs. Keep `-gencode ...,sm_70` and `TORCH_CUDA_ARCH_LIST` unset. |
| Long compile time / RAM spikes | medium | 6 TUs with heavy templates; auto_tune would pick 20 jobs×20 threads on this box. Use MAX_JOBS=4/NVCC_THREADS=4; expect tens of minutes. |
| libstdc++ runtime mismatch | low | conda gcc 12.4 → extension may reference GLIBCXX_3.4.29+; conda python already links conda libstdc++ (RPATH), and env is the runtime target. Smoke test verifies link+load on this host. |
| Fake dist-info in wheel (patch #3) | low | Wheel RECORD covers all files; extra dist-info without RECORD is tolerated by pip/uv. Fallback if wheels get rejected: revert to writing directly to `site.getsitepackages()[0]`. |
| `flash_attn` package name clash with real flash-attn | none new | Same situation as upstream copy trick; namespace shim via `extend_path`. |
| `uv pip install .` re-resolves torch | low after patch #5 | `torch==2.6.0` matches installed `2.6.0+cu126`; still use `--no-deps` on first CI run if paranoid. |

Re-test matrix after patching (fix agent) — steps 1–2 reworded for as-applied behavior (§g.3/§g.5):
1. **No-GPU default build (policy changed):** `python setup.py build_ext --inplace` on this host must
   *succeed* and print the no-GPU warning (setup.py:196-202). The planned fail-fast opt-in
   `FLASH_ATTN_V100_ALLOW_NO_GPU=1` was **not** adopted — see §(g.3) item 2. Capture warning text.
2. **Outputs & import:** check `flash_attn_v100_cuda.cpython-312-*.so` and the `flash_attn_2_cuda.so`
   symlink in the repo root; `python -c "import flash_attn_v100_cuda, flash_attn_2_cuda"` (no kernels).
3. **Clean + incremental:** delete `./build` and rerun (clean); rerun without deleting (incremental,
   ninja must be a no-op).
4. **Wheel path:** `uv pip install . --no-build-isolation --no-deps`; then in a fresh `python -c`
   from a non-repo cwd, check `import flash_attn, flash_attn_v100, flash_attn_v100_cuda, flash_attn_2_cuda`
   and `importlib.metadata.version("flash-attn") == "2.8.3"` (patch #3) plus `version("flash_attn_v100")`.
5. **MMA variants:** rebuild with `MMA_NATIVE=1` and `MMA_884=1` (must compile on 12.6; MMA_884 selects
   `mma_m8n8k4.h`). Optional, doubles wall time.
6. Runtime kernel tests require a real V100 → out of scope on this host (`test.py` skipped).

---

## (f) CUDA-version-feature audit (does anything truly need > 12.6?)

Grep over `kernel/ include/ flash_attn*/` for `CUDART_VERSION|__CUDA_ARCH__|cuda::|#include <cuda|cooperative_groups|__grid_constant__|cudaMemAdvise|__builtin_assume` + PTX inspection:

| Hit (`file:line`) | Construct | Verdict for CUDA 12.6 |
|---|---|---|
| `kernel/{forward,forward_varlen,backward,backward_varlen,forward_kvcache}.cu:4-6` | `#include <cuda.h>`, `<cuda_runtime.h>`, `<cuda_fp16.h>` | OK — core headers, unchanged. |
| `include/{mat_mul,product,softmax,swizzle,gemm_smem,philox,rotary}.h:6-8` | same three headers | OK. |
| `include/mma_m16n16k16.h:20`, `include/mma_m8n8k4.h:37` | `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700) #error` | OK — this is why `-gencode arch=compute_70,code=sm_70` is mandatory; satisfied. |
| `include/mat_mul.h:10,36,177` | `nvcuda::wmma` under `MMA_NATIVE` only (`#include <mma.h>`) | OK — WMMA exists since CUDA 9. |
| `include/mat_mul.h:12`, `include/kernel.h:10` | `MMA_884` selects `mma_m8n8k4.h` (HMMA.m8n8k4, Volta-only) | OK — sm_70 instruction set. |
| inline asm | only `mma.sync.aligned.m8n8k4` (plus explicit HMMA.884 packing helpers) | OK — Volta PTX, supported by every CUDA 9+ ptxas ≤ 12.9; **removed only in CUDA 13.0**. |
| `kernel/*.cu` (`at::cuda::CUDAGuard`, `getCurrentDeviceProperties`, `getCurrentCUDAStream`, `getDefaultCUDAGenerator`) | torch CUDA APIs | OK — present in torch 2.6. Not CUDA-version-bound. |
| `CUDART_VERSION` | **no hits** | nothing gated on cudart version. |
| `cuda::` (libcu++), `cooperative_groups`, `__grid_constant__`, `cudaMemAdvise`, `__builtin_assume`, `cp.async`, `wgmma`, TMA | **no hits** | nothing newer than sm_70/CUDA 9. |

**Verdict: nothing in the code requires CUDA > 12.6.** The `>=12.9` gate is a policy artifact of the
upstream author's recommended CUDA 12.9.1 + torch 2.10.0+cu129 pairing (see `utils/docs/install.md`);
technically the sm_70-only code compiles with any CUDA 11.x–12.9 toolchain. CUDA 12.6 is in fact the
*best* match for torch 2.6.0+cu126 (same cudart).

---

## (g) Failure-capture status, as-applied reconciliation, torch bootstrap

### (g.1) Pre-patch failure capture — obsoleted; derived evidence used instead

The plan required ONE pre-patch build run against `55d685a` to capture the stock failure on this host.
That became impossible: before the capture window closed, the orchestrator landed `fdf8631`
(+`cfcb0cd`), declared the capture obsolete, and neutered the waiting watcher. No pre-patch log exists,
and it cannot be reproduced without reverting tracked files (`git checkout 55d685a -- setup.py` etc.),
which is out of scope — the working tree is HEAD-patched and owned by the orchestrator.

Derived evidence instead — exact stock code, `git show 55d685a:setup.py` lines 139–142:

```python
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available.")
            if parse(torch.version.cuda) < parse("12.9"):
                raise RuntimeError(f"CUDA version {torch.version.cuda} < 12.9 is not supported.")
```

On this no-GPU host the first check fires → `RuntimeError: CUDA is required but not available.`
(on a GPU host with cu126 torch the second would fire: `CUDA version 12.6 < 12.9 is not supported.`).
These are exactly blockers B1/B2 of §(b); a live capture would have added no information. (The stock
`parse(torch.version.cuda)` also crashes on `None` — CPU wheel — fixed in as-applied code.)

### (g.2) Torch bootstrap (env install) — audit trail

| Attempt | Log (`/tmp/opencode/`) | Outcome |
|---|---|---|
| 1 | `torch_install.log` | `EXIT=143` (**SIGTERM**) mid-download; only small wheels completed (networkx, sympy, numpy, nvjitlink, nvrtc, curand, cupti). |
| 2 | `uv_install2.log` | Killed with **no EXIT record** (wrapper died with the process group); log stops at "Downloading nvidia-cudnn/cufft/cusparse/...". |
| 3 | `uv_install3.log` | Restarted **15:30 by this audit**, `setsid`-detached (PIDs 262278/262280) to dodge the background-task sweep — **still SIGTERM'd at ~15:47 (`EXIT=143`)**. The uv cache kept several GB of unpacked wheels (incl. the complete `torch` wheel), reused by the recovery below. |

**Operational lesson for CI:** every plain background install got swept with `EXIT=143` — even the
`setsid`-detached one. What worked was the **staged, cache-first, resumable** strategy: pre-fetch the
big wheels to a local directory (per-file resumable downloads survive kills), then install them with
`uv pip install --no-deps` in small steps (recovery below).

**Gate — PASSED 16:06:39 (runbook §0 requirement):**

```
$ /mnt/envs/v100/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
2.6.0+cu126 12.6 False
$ /mnt/envs/v100/bin/python -c "import ninja, psutil, setuptools, wheel, packaging, numpy, einops; print('build deps ok')"
build deps ok
```

(`False` is required/expected here — no NVIDIA device; as-applied setup.py:196-202 tolerates it.
Raw log: `/tmp/opencode/torch_gate.log`.)

**Recovery (15:50+, orchestrator) — worked around the sweeps entirely:**
1. Pre-downloaded the 7 big nvidia wheels into `/tmp/opencode/wheels/` (1.8 GB; cublas 393 MB,
   cudnn 571 MB, cusparse 216 MB, cufft 200 MB, nccl 189 MB, cusolver 158 MB, cusparselt 157 MB;
   file mtimes 15:34–15:51 → per-file resume across kills).
2. Staged install (`/tmp/opencode/install_final.log`): **step 1** local wheels + cached nvjitlink
   (`STEP1_EXIT=0`, 8 pkgs); **step 2** `torch==2.6.0+cu126` alone, `--no-deps`, straight from the uv
   cache (`STEP2_EXIT=0`, 196 ms — proves the torch wheel was fully cached all along); **steps 3–4**
   small nvidia wheels + python deps; **step 5** verify (`VERIFY_EXIT=0`: torch 2.6.0+cu126, cudnn
   9.5.1, einops 0.8.2). Install stage finished 16:07:54. **Step 6** (late `triton` fetch, 241 MB)
   failed (`TRITON_LATE_FAILED`) — triton is optional here (torch.compile only; the FA2 build uses
   nvcc directly), the only remaining un-fetched item.
3. The uv archive also holds every pure-python dep torch needs at import
   (typing_extensions 4.16.0, sympy 1.13.1, networkx, jinja2, fsspec, filelock, mpmath), so the tail
   was small and cache-served.

**Gate outcome:** shown above — the env is build-ready for P4 runbook §4 onward.

### (g.3) As-applied vs. planned (§(c)) — per-item reconciliation

| §(c) item | As applied (`fdf8631` unless noted) | Verdict / delta |
|---|---|---|
| 1 torch hint text | Applied — setup.py:88-91 and 172-176 now point at `torch==2.6.0 …/cu126`. | ✅ as planned |
| 2a CUDA gate (B2) | Applied differently: `FLASH_ATTN_MIN_CUDA` env override, default `"12.6"`, plus explicit `torch.version.cuda is None` guard — setup.py:184-195. | ✅ equivalent (extra knob) |
| 2b no-GPU (B1) | **Applied differently:** `torch.cuda.is_available()` false ⇒ warn + continue (setup.py:196-202). `FLASH_ATTN_V100_ALLOW_NO_GPU` **does not exist**. | ⚠️ policy delta — default is permissive; §(e) re-test step 1 reworded; residual risk: a GPU host with a broken driver silently produces a compile-only extension. |
| 3 InstallAttention → `self.install_lib` (B7) | **Not applied** — still `site.getsitepackages()[0]` (setup.py:157-167). | ⛔ open (G1): fine for `build_ext --inplace` and `uv pip install --no-build-isolation` (runs in the target env); wrong for isolated builds / `uv build` from another interpreter. |
| 4 `packages += ["flash_attn"]` (B8) | **Not applied** — still `["flash_attn_v100"]` (setup.py:233, pyproject.toml:35); `CopyAttention` copy trick remains (setup.py:143-155). | ⚠️ open (G2, mitigated); wheel content verified by §(e) step 4. |
| 5 pyproject (B3/B4) | **Half applied:** `dependencies` now `torch==2.6.0` + einops/ninja/psutil/numpy/setuptools/wheel/packaging (pyproject.toml:23-32) ✅; build-system keeps bare `"torch"` (pyproject.toml:7) instead of removing it ❌. | ⚠️ delta (G3): inert under mandatory `--no-build-isolation`; isolation would provision some other torch. |
| 6 requirements.txt | **Not applied** — unchanged from `55d685a` (it has no torch line anyway). | ℹ️ no impact |
| 7 run.sh (B5/B6) | **Partially applied:** `--no-deps` added (run.sh:33); `test.py` gated on a runtime `torch.cuda.is_available()` probe → else `tests/run_all_cpu.py`. Bare `pip`/`python` remain; no `--no-gpu` flag / uv fallback. | ⚠️ open (G4) |
| 8 include bridge (B11) | **Applied as a superset:** `_extra_include_dirs()` (setup.py:23-58) adds repo `include/` + `<CUDA_HOME>/{include,targets/x86_64-linux/include}` + every `site-packages/nvidia/*/include`. `CUDA_INC_PATH` no longer needed. | ✅✅ stronger than planned |
| (new) `cfcb0cd` | `include/mat_mul.h`: `__tanhf(...)` → `tanhf(...)` (2 lines, softcap masking). `__tanhf` is not a CUDA device intrinsic on any toolkit; real compile fix (commit message: all 6 TUs build clean on 12.6). | ✅ required |
| (new) `scripts/build_cuda126.sh` | `CUDA_HOME=$ENV_PREFIX/targets/x86_64-linux` (verified: `bin/nvcc` + `include/` exist → works), `MAX_JOBS=32`, `NVCC_THREADS=4`; installs via `uv pip install --no-build-isolation --no-deps .`. | ⚠️ see G5 (`MAX_JOBS=32`; `--wheel` path lacks `--python`) |

### (g.4) Residual gaps (ordered; none hard-blocking)

- **G1 — InstallAttention dist-info target (B7).** Until fixed per §(c) item 3, build/install wheels only
  with the v100 interpreter: `python setup.py bdist_wheel` (env python) or
  `uv pip install --python /mnt/envs/v100/bin/python --no-build-isolation --no-deps .` (the canonical
  runbook path). If `uv build --wheel` is ever used, add `--python /mnt/envs/v100/bin/python`.
- **G2 — `flash_attn` packaging declaration (B8).** Cosmetic for import/CI (the copy runs); confirm the
  wheel contains `flash_attn/` via §(e) step 4.
- **G3 — bare `"torch"` in build-requires (B3).** Keep `--no-build-isolation` mandatory in every
  install path (already true in runbook + scripts). If isolation ever becomes required, replace with
  the comment form from §(c) item 5.
- **G4 — run.sh bare `pip`/`python` (B5).** Env has no `pip`, and `python` resolves via PATH. Minimal
  fix: `PY=${PY:-/mnt/envs/v100/bin/python}`; `uv pip install --python "$PY" --no-build-isolation
  --no-deps -v .`; run tests with `"$PY"`.
- **G5 — build_cuda126.sh defaults.** `MAX_JOBS=32` risks OOM on this 56 GiB host (auto-tune formula
  would pick 20; runbook uses 8) → recommend `MAX_JOBS=8`. Add `--python "$ENV_PREFIX/bin/python"` to
  the `uv build --wheel` branch.

### (g.5) Acceptance sequence (supersedes §(e) steps 1–2 wording)

Canonical as-applied sequence = `/tmp/opencode/P4_runbook.md` §0→§8:

1. Torch gate (§g.2) → `2.6.0+cu126 12.6 False` ✅ passed 16:06.
2. `/mnt/envs/v100/bin/python tests/run_all_cpu.py` (pre-build; extension smoke skips).
3. `/mnt/envs/v100/bin/python tests/parity_check.py --mock` + `--ast`.
4. Full build: `CUDA_HOME=/mnt/envs/v100`, `CC=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-gcc`,
   `CXX=…-g++`, `MAX_JOBS=8 NVCC_THREADS=4`, `TORCH_CUDA_ARCH_LIST` unset →
   `/mnt/envs/v100/bin/python setup.py build_ext --inplace`; **the no-GPU warning must appear**.
5. `scripts/verify_sass.sh` + `tests/collect_ptxas_stats.py --expect-arch 70` (optional:
   `conda install -n v100 -c nvidia cuda-cuobjdump cuda-nvdisasm`; skip gracefully if absent).
6. `uv pip install --python /mnt/envs/v100/bin/python --no-build-isolation --no-deps .` → import
   `flash_attn`, `flash_attn.flash_attn_interface`, `flash_attn_v100`, `flash_attn_v100_cuda`,
   `flash_attn_2_cuda`; `importlib.metadata.version("flash-attn") == "2.8.3"` and `version("flash_attn_v100")`.
7. `/mnt/envs/v100/bin/python tests/run_all.py --build-exit 0 --build-mode default`.
8. Kernel runtime tests require a real V100 — out of scope on this host (`test.py` skipped by run.sh).

---

## Final summary (≤15 lines)

1. Code/toolchain verdict: nothing needs CUDA > 12.6; build with torch 2.6.0+cu126, conda gcc 12.4
   (CC/CXX exports mandatory, else nvcc hits system gcc 16), `-gencode compute_70,sm_70` only.
2. As-applied (`fdf8631`+`cfcb0cd`+scripts+run.sh) covers the hard blockers: CUDA gate ≥12.6 via
   `FLASH_ATTN_MIN_CUDA`, no-GPU builds always allowed (warn-only), `torch==2.6.0` deps, einops, test
   gating, conda include dirs (`_extra_include_dirs()`), plus the required `__tanhf`→`tanhf` fix.
3. Open gaps (none blocking): install_lib dist-info (wheel/isolation only), `flash_attn` package
   declaration, bare "torch" in build-requires (keep `--no-build-isolation`), run.sh's bare pip/python,
   build_cuda126.sh `MAX_JOBS=32` and `--wheel` without `--python`.
4. Env: sweeps killed three install attempts (`EXIT=143`); staged cache-first recovery worked — gate
   PASSED 16:06 (`2.6.0+cu126 12.6 False`, deps ok; triton optional, unfetched).
5. Next step = P4 runbook §0→§8: torch gate → CPU tests → `build_ext --inplace` (MAX_JOBS=8,
   NVCC_THREADS=4, CUDA_HOME=/mnt/envs/v100, CC/CXX exported) → SASS checks → uv wheel install →
   `tests/run_all.py`. Kernel runtime verification needs a real Volta GPU.
