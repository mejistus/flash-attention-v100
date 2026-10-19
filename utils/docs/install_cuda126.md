[<- Main](../../README.md)

CUDA 12.6 / PyTorch 2.6.0 build (conda compilers, no system CUDA required)
-------------

This route builds the official-named distribution (`flash-attn` 2.8.3, import
`flash_attn`, extension `flash_attn_2_cuda`) with the CUDA 12.6 toolchain from
conda-forge / NVIDIA channels (`-c nvidia`) and PyTorch 2.6.0+cu126 from the PyTorch
index. It was verified end-to-end on a GPU-less machine and on a Tesla V100S-32GB:
all six translation units compile with `nvcc 12.6.85`, `-arch=sm_70`, the wheel
installs, the extension imports (including `torch.ops`), and the acceptance suite
passes. The V100 release wheel in this repository was built with torch 2.8.0+cu128 /
CUDA 12.8 (same sm_70 kernels); use the recipe that matches your torch version.

### 1. Create the conda environment

```bash
conda create -n v100 python=3.12 -y
conda activate v100
# CUDA 12.6 compiler + runtime headers (no system-wide CUDA install needed)
conda install -n v100 -y -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6
# host compiler: nvcc 12.6 rejects gcc > 13; the conda env ships gcc 12
conda install -n v100 -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12
```

Env layout produced by the `cuda-nvcc` conda package:

| what | path |
|------|------|
| nvcc | `$CONDA_PREFIX/bin/nvcc` (wrapper for `$CONDA_PREFIX/targets/x86_64-linux/bin/nvcc`) |
| CUDA headers | `$CONDA_PREFIX/targets/x86_64-linux/include` |
| libcudart | `$CONDA_PREFIX/targets/x86_64-linux/lib` |
| host gcc/g++ 12 | `$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-{gcc,g++}` |
| nvcc components (`cicc`, `cudafe++`) | `$CONDA_PREFIX/nvvm/bin`, `$CONDA_PREFIX/bin` |

### 2. Install PyTorch and Python deps with uv

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" \
    torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$CONDA_PREFIX/bin/python" einops ninja packaging psutil
```

`torch==2.6.0+cu126` is the newest PyTorch built against CUDA 12.6 with sm_70 kernels.
For torch 2.8 (the version used for the published V100 wheel) use the cu128 wheel and
CUDA 12.8 (`cuda-nvcc=12.8`); the extension is ABI-bound to the torch version it is
compiled against.

### 3. Build the extension

```bash
# from the repository root:
bash scripts/build_cuda126.sh
```

The script sets everything the build needs:

```bash
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"   # conda toolkit layout
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/bin:$PATH"
uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation --no-deps .
```

Notes:

* `setup.py` adds the CUDA headers plus the cuBLAS/cuSPARSE/cuSOLVER headers shipped
  inside PyTorch's `nvidia-*-cu12` wheels, and tolerates a missing GPU at build time
  (`FLASH_ATTN_MIN_CUDA`, default `12.6`).
* `--no-deps` keeps `pip` from re-resolving `torch`; `--no-build-isolation` uses the
  toolchain you just installed.
* The build instantiates ~1.4k sm_70 kernels; use `MAX_JOBS`/`NVCC_THREADS` to control
  parallelism (6 TUs; `NVCC_THREADS=16` on a large machine).
* conda `nvcc` is invoked through `targets/x86_64-linux/bin/nvcc` (a symlink) and
  resolves `cicc`/`cudafe++` through `PATH`, hence the `nvvm/bin` entry above.
* To create a wheel instead: `bash scripts/build_cuda126.sh --wheel` (output in `dist/`).

### 3b. What gets installed (drop-in details)

* `flash_attn_2_cuda*.so` — the compiled extension (sm_70), imported as
  `flash_attn_2_cuda` (official name; the previous `flash_attn_v100_cuda` name and the
  separate alias module are gone).
* `flash_attn/` — the implementation package (`__version__ == "2.8.3"`):
  `flash_attn.flash_attn_interface`, `flash_attn.bert_padding`, and the official API
  exports.
* `flash_attn-2.8.3.dist-info` — the wheel's own metadata (exactly one `.dist-info`
  per wheel, PEP 427). No post-install metadata step is needed any more;
  `importlib.metadata.version("flash-attn") == "2.8.3"` directly.

### 4. Verify

```bash
# import smoke test (no GPU needed)
python - <<'PY'
import flash_attn, flash_attn_2_cuda
import importlib.metadata as md
print(flash_attn.__version__, md.version("flash-attn"))   # 2.8.3 2.8.3
print(flash_attn_2_cuda.fwd)                              # extension function
PY

# GPU-free acceptance suite (API parity, wrapper semantics vs reference math,
# bert_padding official-source parity, installed-extension smoke + torch.ops)
python tests/run_all_cpu.py

# static sm_70 SASS check (one-time tool install, persistent location):
#   conda create -y -p /mnt/envs/cuda-tools -c nvidia -c conda-forge \
#       cuda-nvdisasm=12.6 cuda-cuobjdump=12.6
#   (CUDA 13+ nvdisasm dropped sm_70 support, hence the 12.6 pin)
bash scripts/verify_sass.sh

# full acceptance report (JSON evidence)
python tests/run_all.py --repo . --python "$CONDA_PREFIX/bin/python"

# on a machine WITH a V100: numeric verification against an eager reference
python tests/gpu_verify.py
```

### Known limitations

* fp16 only at the kernel level; head dims `{16,32,64,128,256}` natively; other head
  dims are zero-padded to the next supported size by the Python layer (exact results,
  small memory/compute overhead).
* `flash_attn_with_kvcache` paged KV: tiles that cross a physical page boundary are
  loaded page-segment by page-segment, so arbitrary page sizes work; the host rejects
  block tables that do not cover `seqlen_k + leftpad`. `num_splits > 1` is not
  implemented (single-pass path). `rotary_cos/sin`, `cache_batch_idx` and
  `cache_leftpad` are implemented by the kernel; the CPU mock only models the
  non-rotary, non-paged subset.
* `deterministic=True` is accepted with a warning (the Volta backward uses the
  non-deterministic atomic path).
* `softcap`, `window_size`, `alibi_slopes` are supported on dense/varlen/kvcache
  forward paths and on backward.
* Compute vs PyTorch SDPA (mem-efficient): this build is memory-first. On V100 it is
  significantly faster than the previous V100 implementation and uses ~40–50 % less
  memory than SDPA for GQA/varlen training, but SDPA's cutlass kernels are faster on
  MHA. See the README table and `bench/bench_pareto.py`.
* `flash_attn.ops`, `layers`, `modules`, `utils`, `models` are not shipped.
