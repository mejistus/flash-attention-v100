# FlashAttention-2 for NVIDIA Volta (V100 16 GB / 32 GB)

A drop-in build of FlashAttention-2 for Volta GPUs. It installs as the official
distribution name **`flash-attn` 2.8.3**, imports as **`flash_attn`**, and ships the
compiled extension as **`flash_attn_2_cuda`** — exactly like the official package on
newer GPUs, with no architecture suffix in the library or API.

* Built and validated on a **Tesla V100S-PCIE-32GB** (sm_70, 32768 MiB) with
  **torch 2.8.0+cu128** and **CUDA 12.8** (nvcc 12.8.93); prebuilt wheels are also
  published for **CUDA 12.6 / torch 2.6.0+cu126** and **CUDA 12.4 / torch
  2.5.1+cu124** (table below).
* Works within a 16 GB budget too (V100 16 GB shares the same sm_70 code path; all
  buffers are sized by the tensors you pass — see the 16 GB note below).
* Memory-first Pareto profile: vs the previous V100 implementation, training peak
  memory drops by **~43 %** (dense) / **~23 %** (packed varlen) while compute stays
  the same or improves slightly. Compared with PyTorch SDPA it trades some speed for
  substantially lower memory (tables below).
* **Official Python API parity**: the pure-Python layer is upstream flash-attn
  2.8.3 verbatim (a private V100 adapter). Existing code written for the
  official package — including `diffusers` and `transformers` FlashAttention-2
  integrations — runs unchanged. See
  [PYTHON_API_ALIGNMENT.md](PYTHON_API_ALIGNMENT.md).

## Install

Prebuilt wheels for CPython 3.12 / Linux x86_64 / sm_70 (the distribution name is
still **`flash-attn`**; imports are unchanged):

| Wheel | PyTorch | CUDA | sha256 |
|-------|---------|------|--------|
| [flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl](https://github.com/mejistus/flash-attention-v100/releases/download/v2.8.3-v100-api/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl) | 2.8.0+cu128 | 12.8 | `628fe534e4d930ee7e1800393ba8ad6b4e5042b914e2a35703942c339542673f` |
| [flash_attn-2.8.3+cu126.torch2.6.0-cp312-cp312-linux_x86_64.whl](https://github.com/mejistus/flash-attention-v100/releases/download/v2.8.3-v100-api/flash_attn-2.8.3%2Bcu126.torch2.6.0-cp312-cp312-linux_x86_64.whl) | 2.6.0+cu126 | 12.6 | `aa3960b691d13b5c6e84b293d15fcd4c9d951fa35021b9704f35cd66bfa6fea3` |
| [flash_attn-2.8.3+cu124.torch2.5.1-cp312-cp312-linux_x86_64.whl](https://github.com/mejistus/flash-attention-v100/releases/download/v2.8.3-v100-api/flash_attn-2.8.3%2Bcu124.torch2.5.1-cp312-cp312-linux_x86_64.whl) | 2.5.1+cu124 | 12.4 | `60d727910330c9692fd30ea38df296e93e2121811fd7049a779b3f4fc501f7f4` |

`torch 1.6.0` in the original request is interpreted as a typo for **2.6.0**:
torch 1.6 (2020) is incompatible with CUDA 12.x, Python 3.12 and the current
PyTorch extension APIs.

```bash
# standard torch 2.8 + CUDA 12.8 environment, for example
pip install https://github.com/mejistus/flash-attention-v100/releases/download/v2.8.3-v100-api/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
# sha256: 628fe534e4d930ee7e1800393ba8ad6b4e5042b914e2a35703942c339542673f
```

Release page: <https://github.com/mejistus/flash-attention-v100/releases/tag/v2.8.3-v100-api>

The **API-aligned rebuild** is published under the `v2.8.3-v100-api` tag; the original `v2.8.3` assets remain available for the earlier build.

Then, exactly like the official package:

```python
import torch, flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
import flash_attn_2_cuda            # the compiled extension
print(flash_attn.__version__)       # 2.8.3
print(flash_attn_2_cuda.fwd)        # extension function
```

The wheels are ABI-bound to the torch version they were compiled against; pick the
row matching your PyTorch/CUDA environment. For other torch versions, build from
source (below); torch extensions are ABI-bound to the torch version they are
compiled against.

## Python API parity (official 2.8.3)

`flash_attn/flash_attn_interface.py`, `bert_padding.py` and `__init__.py` are
upstream flash-attn 2.8.3 verbatim; the only private addition is
`flash_attn/_v100_compat.py`, imported and applied by two lines in `__init__.py`. It adapts only the
compiled V100 extension (zero-padding unsupported head dimensions, contiguous
packed views, upstream 128-padded mask/softmax_d shapes); public names,
signatures, defaults and return layouts are unchanged.

This means the official `_flash_attn_forward/backward` custom ops exist, packed and
varlen entry points match, and `diffusers` / `transformers` / `torch.compile` see a
normal official flash-attn 2.8.3 package. Optional upstream Python packages
(`modules`, `layers`, `ops/triton`, `models`, `utils`, `losses`) are included;
FA3/CuTe, AMD Triton and blocksparse are excluded. The separate
`fused_dense_lib`/`dropout_layer_norm` native extensions are optional in upstream
2.8.3 and are not part of its default wheel either.

See [PYTHON_API_ALIGNMENT.md](PYTHON_API_ALIGNMENT.md) for the exact scope,
verification (CPU suite 8/8 PASS, parity PASS; diffusers, transformers and
torch.compile) and limitations.

## Build from source (V100 / sm_70, verified recipe)

The build runs on a GPU-less machine too; a GPU is only needed to run.

```bash
# 1. conda environment (persistent path; not /tmp)
conda create -n v100 python=3.12 -y
conda activate v100
conda install -n v100 -y -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6
conda install -n v100 -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12

# 2. PyTorch + build deps (uv is recommended)
uv pip install --python "$CONDA_PREFIX/bin/python" \
    torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$CONDA_PREFIX/bin/python" einops ninja packaging psutil

# 3. build + install (the script fixes CUDA_HOME/CC/CXX/PATH for conda's nvcc)
bash scripts/build_cuda126.sh

# alternative: build a wheel into ./dist
bash scripts/build_cuda126.sh --wheel
```

For a **torch 2.8 + CUDA 12.8** environment (the one used for the released wheel):

```bash
conda create -n v100 python=3.12 -y && conda activate v100
conda install -n v100 -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging psutil einops

CUDA_HOME=/usr/local/cuda MAX_JOBS=8 NVCC_THREADS=16 \
  pip install --no-build-isolation --no-deps .
```

Build notes:

* `nvcc` must see `cicc`/`cudafe++`: the build script prepends
  `$CONDA_PREFIX/nvvm/bin` and `$CONDA_PREFIX/bin` to `PATH` (conda's
  `targets/x86_64-linux/bin/nvcc` is a symlink and resolves components via `PATH`).
* `NVCC_THREADS` controls per-translation-unit parallelism (the build has 6 TUs);
  large machines benefit from `MAX_JOBS=8 NVCC_THREADS=16`.
* No physical GPU is required at build time (you only get a warning).
* `scripts/build_with_evidence.sh` wraps the build and archives logs under
  `tests/artifacts/build_cuda126/`.

## Usage

```python
import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache

q = torch.randn(2, 4096, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 4096,  8, 128, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(2, 4096,  8, 128, device="cuda", dtype=torch.float16, requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)          # official API / GQA supported
out.sum().backward()

# packed varlen (RL fine-tuning)
cu = torch.tensor([0, 1024, 3072, 4096], device="cuda", dtype=torch.int32)
qv = q.reshape(-1, 32, 128); kv = k.reshape(-1, 8, 128); vv = v.reshape(-1, 8, 128)
out = flash_attn_varlen_func(qv, kv, vv, cu, cu, max_seqlen_q=2048, max_seqlen_k=2048,
                             causal=True)

# inference with a paged KV cache (block_table supported)
out = flash_attn_with_kvcache(q[:, :1], k_cache, v_cache, cache_seqlens=seq_lens)
```

## Memory / performance (measured on V100S 32 GB)

Training `fwd+bwd`, fp16, GQA `H=32 / H_k=8 / D=128`, causal, peak CUDA memory and
median latency (torch 2.8.0+cu128; `bench/bench_pareto.py`):

| workload | this build | previous V100 build | PyTorch SDPA (mem-efficient) |
|----------|-----------:|--------------------:|-----------------------------:|
| dense, N=8192 | **418 MiB / 120.9 ms** | 738 MiB / 126.6 ms | 770 MiB / 69.8 ms |
| dense MHA (H=H_k=32), N=8192 | **706 MiB / 112.6 ms** | – | 834 MiB / 69.6 ms |
| packed varlen, T=16384 (4 seqs) | **836 MiB / 112.6 ms** | 1092 MiB / 123.4 ms | 1540 MiB / 71.8 ms (padded) |
| packed varlen, T=32768 | **1672 MiB / 442.9 ms** | 2184 MiB / 478.7 ms | 3080 MiB / 285.7 ms (padded) |
| inference prefill, N=8192 | **224 MiB / 24.2 ms** | 321 MiB / 25.9 ms | 288 MiB / 15.3 ms |
| KV-cache decode, cache=8192 | 48 MiB / 0.57 ms | 48 MiB / 0.48 ms | – |

What changed vs the previous V100 implementation:

* **Dense inputs use the official BMHD layout end-to-end** — no `permute().contiguous()`
  copies of q/k/v/out (that alone was ~320 MiB at 8k tokens during training).
* **GQA/MQA backward no longer allocates `(T_k, H_q, D)` expansion buffers**; one CTA
  per KV head loops over its Q-head group and accumulates dK/dV in shared memory.
* Tuned tile shapes and `launch_bounds`; empty-KV sentinel, all-masked rows, paged KV,
  dropout RNG and dmask fixes from the red/blue review rounds.

Compute vs PyTorch SDPA: this is the deliberate trade-off. SDPA's cutlass
mem-efficient kernels are faster on V100, but they need KV-head expansion for GQA
(extra memory) and have no packed-varlen path. This build uses **40–50 % less memory**
than SDPA for GQA/varlen training, with full GQA and varlen support. Use SDPA only if
raw speed matters more than memory on an MHA workload.

## Real applications on V100

Two complete application benchmarks live in [`bench/apps/`](bench/apps/README.md):
DINOv3-H LoRA (rank 64) fine-tuning on TinyImageNet-200 and SD3.5-Medium
text-to-image generation. Each has an SDPA baseline and a `flash_attn` variant with
identical hyper-parameters, run on the same V100S-32GB / torch 2.8.0+cu128 setup:

| workload | PyTorch SDPA | this build | difference |
|----------|-------------:|-----------:|-----------:|
| DINOv3-H LoRA(64), 224 px, batch 16 | 386.8 ms/step, 13.2 GiB | 394.6 ms/step, 13.2 GiB | +2.03 % |
| SD3.5-Medium, 512×512, 20 steps | 2768.6 ms/image, 17.7 GiB | 3005.2 ms/image, 17.7 GiB | +8.55 % |

DINOv3 LoRA training is essentially neutral (0–2 % depending on run/environment);
SD3.5 inference is consistently ~8.5 % slower with flash-attn on V100. Attention
is not the bottleneck in these short-sequence workloads; the repo's advantage is
memory at large batch/sequence (table above). The two generated images are visually
equivalent (PSNR 42.46 dB). Scripts, logs, JSON results and the PNG outputs are
committed under `bench/apps/`.

## Verification

```bash
# GPU-free: API parity, wrapper semantics vs a CPU mock, official-source parity
python tests/run_all_cpu.py

# full acceptance report (JSON evidence; after a build)
python tests/run_all.py --repo . --python "$CONDA_PREFIX/bin/python"

# on a V100: numeric correctness vs an eager fp32 reference
python tests/gpu_verify.py

# static sm_70 SASS legality (persistent tools, CUDA 12.6 nvdisasm)
bash scripts/verify_sass.sh
```

## 16 GB note

The V100 16 GB and 32 GB SKUs run identical sm_70 code; no compile-time flag differs.
Memory use is proportional to the tensors you pass (FlashAttention is O(N) in sequence
length), so a 16 GB card fits the same workloads up to the smaller capacity. For a
strict check, run with a 16 GB allocator cap:

```python
torch.cuda.set_per_process_memory_fraction(0.5)   # 16 GB of a 32 GB card
```

## Scope / known gaps

* `flash_attn` core API (dense, varlen, packed variants, kvcache) and
  `flash_attn.bert_padding` are implemented and official-parity tested.
  Upstream's optional subpackages (`flash_attn.ops`, `layers`, `modules`, `utils`,
  `models`) are not shipped.
* fp16 (and fp32 inputs cast to fp16) on sm_70. Head dims `{16,32,64,128,256}` are
  native; other sizes are zero-padded to the next supported size by the Python layer
  (exact results, small overhead).
* `num_splits > 1` is not implemented (single-pass path); `deterministic=True` is
  accepted with a warning.
* varlen `S_dmask` keeps the tight `(T_q, H, max_k)` shape (dense matches upstream's
  padded 4-D shape) — documented deviation.
* No performance claim for hardware other than V100; the kernels target sm_70.
  The default build is the WMMA path (verified by SASS to use `HMMA.884`); the
  optional `MMA_884` variant is not part of the release.

## License

BSD-3-Clause. Based on [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
and Tri Dao's FlashAttention-2.
