#!/usr/bin/env bash
# ======================================================================================
# Build flash_attn against CUDA 12.6 inside the conda env (default: /mnt/envs/v100),
# on a machine without a physical GPU.
#
# Usage:
#   bash scripts/build_cuda126.sh                 # install into the env (uv)
#   bash scripts/build_cuda126.sh --wheel         # build a wheel into ./dist instead
#
# Environment fixes it applies (see research/build_compat_audit.md):
#   * CUDA_HOME -> conda toolkit layout (headers under targets/x86_64-linux/include)
#   * CC/CXX   -> conda gcc 12.4 (nvcc 12.6 rejects host gcc > 13)
#   * nvidia-* pip wheel headers (cublas/cusparse/cusolver) are added by setup.py itself.
# ======================================================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${CONDA_PREFIX:-/mnt/envs/v100}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "error: conda env python not found at $ENV_PREFIX/bin/python" >&2
    exit 1
fi

export CUDA_HOME="$ENV_PREFIX/targets/x86_64-linux"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# conda cuda-nvcc keeps cicc under $PREFIX/nvvm/bin; nvcc invoked through the
# targets/x86_64-linux/bin/nvcc symlink cannot locate it via argv[0]-relative
# search, so expose it on PATH (nvcc falls back to PATH for its components).
export PATH="$ENV_PREFIX/nvvm/bin:$ENV_PREFIX/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC"
echo "MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"
"$CUDA_HOME/bin/nvcc" --version | tail -2
"$ENV_PREFIX/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

cd "$REPO_DIR"
if [[ "${1:-}" == "--wheel" ]]; then
    uv build --wheel --no-build-isolation --out-dir dist .
    echo "wheel written to $REPO_DIR/dist"
else
    # The distribution is named `flash-attn` and ships `flash_attn` +
    # `flash_attn_2_cuda`, so the wheel carries the official `flash_attn-2.8.3.dist-info`
    # itself; no post-install metadata step is needed.
    uv pip install --python "$ENV_PREFIX/bin/python" --no-build-isolation --no-deps "$@" .
fi
