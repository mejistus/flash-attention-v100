# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *

import os
import shutil
import pathlib
from pathlib import Path
try:
    from packaging.version import parse
except ImportError:  # `packaging` is not a hard dependency of torch
    def parse(s):
        return tuple(int(x) for x in str(s).split(".") if x.isdigit())
from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = Path(__file__).parent.resolve()


def _extra_include_dirs():
    """Include dirs beyond this repo's own `include/`.

    1. CUDA toolkit headers, covering both layouts:
       - conda `cuda-nvcc` installs: $CONDA_PREFIX/targets/x86_64-linux/include
       - standard toolkit / pip nvidia-cuda-nvcc: $CUDA_HOME/include
    2. cuBLAS/cuSPARSE/cuSOLVER headers shipped inside torch's PyPI wheels under
       site-packages/nvidia/*/include (torch's ATen CUDA headers pull them in even
       though this project does not call those libraries directly).
    """
    dirs = [str(this_dir / "include")]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        for sub in ("include", "targets/x86_64-linux/include"):
            cand = Path(cuda_home) / sub
            if cand.is_dir():
                dirs.append(str(cand))
    try:
        import torch as _torch

        site_pkgs = Path(_torch.__file__).resolve().parent.parent
        nvidia_dir = site_pkgs / "nvidia"
        if nvidia_dir.is_dir():
            for pkg in sorted(nvidia_dir.iterdir()):
                inc = pkg / "include"
                if inc.is_dir():
                    dirs.append(str(inc))
    except Exception as e:
        print(f"Warning: could not scan nvidia wheel include dirs: {e}")
    # de-duplicate, keep order
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def auto_tune():
    if os.environ.get("MAX_JOBS") and os.environ.get("NVCC_THREADS"):
        return

    try:
        import psutil

        cores = os.cpu_count() or 4
        mem_sys  = psutil.virtual_memory().available / (1024**3)
        max_jobs = max(1, int((mem_sys * 0.9) / 2.5))
        max_jobs = min(max_jobs, cores)
        max_thrd = max_jobs
        os.environ["MAX_JOBS"]     = str(max_jobs)
        os.environ["NVCC_THREADS"] = str(max_thrd)

        print(f"autoset max_jobs={max_jobs}, nvcc_threads={max_thrd} (current {cores} cores, {mem_sys:.1f}GB free mem)")
    except Exception as e:
        print(f"Warning: could not auto-tune build params: {e}")
        os.environ.setdefault("MAX_JOBS", "4")
        os.environ.setdefault("NVCC_THREADS", "4")

auto_tune()

def get_ext_modules():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn. "
            "Please install a CUDA-enabled torch (e.g., `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126`)."
        ) from e

    nvcc_threads = int(os.environ.get("NVCC_THREADS", 4))
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-gencode", "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Wno-deprecated-gpu-targets",
        f"--threads={nvcc_threads}",
    ]

    if os.environ.get("ATTENTION_DEBUG"):
        nvcc_flags.extend([
            "-DKERNEL_DEBUG",
            "-g",
            "--keep",
            "--keep-dir", str(this_dir / "build"),
            "-Xptxas", "-v",
        ])
        (this_dir / "build").mkdir(exist_ok=True)

    if os.environ.get("FLASH_ATTN_PTXAS_VERBOSE"):
        # Flags-pure resource report for the release build: identical codegen to the
        # default build, only ptxas' verbose statistics are enabled. Parsed by
        # tests/collect_ptxas_stats.py.
        nvcc_flags.extend(["-Xptxas", "-v"])

    if os.environ.get("MMA_NATIVE"):
        nvcc_flags.extend(["-DMMA_NATIVE"])

    if os.environ.get("MMA_884"):
        nvcc_flags.extend(["-DMMA_884"])

    return [
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "kernel/api.cpp",
                "kernel/forward.cu",
                "kernel/forward_varlen.cu",
                "kernel/backward.cu",
                "kernel/backward_varlen.cu",
                "kernel/forward_kvcache.cu",
            ],
            include_dirs=_extra_include_dirs(),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ]


def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn. "
            "Please install a CUDA-enabled torch (e.g., `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126`)."
        ) from e

    class BuildAttention(BuildExtension):
        def build_extensions(self):
            import torch

            # Minimum CUDA version can be overridden for experiments; CUDA 12.6 is the
            # oldest toolkit with full Volta (sm_70) support that we validate against.
            min_cuda = os.environ.get("FLASH_ATTN_MIN_CUDA", "12.6")
            cuda_ver = torch.version.cuda
            if cuda_ver is None:
                raise RuntimeError(
                    "flash_attn requires a CUDA-enabled PyTorch build (torch.version.cuda is None). "
                    "Install e.g. `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126`."
                )
            if parse(cuda_ver) < parse(min_cuda):
                raise RuntimeError(
                    f"CUDA version {cuda_ver} < {min_cuda} is not supported "
                    f"(set FLASH_ATTN_MIN_CUDA to override)."
                )
            if not torch.cuda.is_available():
                # Building does not require a physical GPU. The produced extension targets
                # sm_70 and is only usable on a Volta GPU at runtime.
                print(
                    "Warning: no CUDA device detected at build time. Building without a GPU; "
                    "the extension will only run on machines with a Volta (sm_70) GPU."
                )

            super().build_extensions()


    return {"build_ext": BuildAttention}

try:
    with open(this_dir / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Flash Attention implementation for Tesla V100"

setup(
    name="flash-attn",
    version="2.8.3",
    packages=["flash_attn"],
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
    python_requires=">=3.10",
    zip_safe=False,
    author="D.Skryabin",
    author_email="tg @ai_bond007",
    description="FlashAttention-2 for NVIDIA Volta GPUs (drop-in flash-attn API)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-bond/flash-attention-v100",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
)