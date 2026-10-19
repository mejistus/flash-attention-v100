#!/usr/bin/env bash
# ======================================================================================
# build_with_evidence.sh — full CUDA 12.6 / sm_70 build with repo-visible evidence.
#
# Wraps scripts/build_cuda126.sh (which installs into the conda env via uv) and
# archives everything under tests/artifacts/build_cuda126/ so the evidence survives
# reboots (never keep the only copy in /tmp):
#
#   build_env.txt   toolchain + versions snapshot at build time
#   build.log       full build output (stdout+stderr of the uv/setuptools run)
#   build.log.*     previous attempts, timestamped (kept for the repair history)
#
# Usage:
#   bash scripts/build_with_evidence.sh            # build + install into $V100_ENV
#   V100_ENV=/path/to/env bash scripts/build_with_evidence.sh
#
# The conda env itself must live on persistent storage (default /mnt/envs/v100).
# ======================================================================================
set -u
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

ENV_PREFIX="${V100_ENV:-/mnt/envs/v100}"
AD="${BUILD_EVIDENCE_DIR:-$REPO_DIR/tests/artifacts/build_cuda126}"
mkdir -p "$AD"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "error: conda env python not found at $ENV_PREFIX/bin/python" >&2
    exit 1
fi

{
  echo "=== build env snapshot $(date -Iseconds) ==="
  echo "CONDA_PREFIX=$ENV_PREFIX"
  echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
  echo "CC=${CC:-<unset>}"
  echo "CXX=${CXX:-<unset>}"
  echo "MAX_JOBS=${MAX_JOBS:-<unset>}"
  echo "NVCC_THREADS=${NVCC_THREADS:-<default>}"
  echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
  echo "FLASH_ATTN_V100_ALLOW_NO_GPU=${FLASH_ATTN_V100_ALLOW_NO_GPU:-<unset>}"
  "$ENV_PREFIX/bin/nvcc" --version | tail -2
  "$ENV_PREFIX/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
  "$ENV_PREFIX/bin/python" -c "import sys; print('python', sys.version.split()[0])"
  "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" --version | head -1
} > "$AD/build_env.txt" 2>&1

echo "=== launching build $(date -Iseconds) ==="
cp -f "$AD/build.log" "$AD/build.log.$(date +%H%M%S)" 2>/dev/null || true
# Force the target conda env: the ambient CONDA_PREFIX may point at the base miniforge.
export CONDA_PREFIX="$ENV_PREFIX"
MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-4}" \
    bash "$REPO_DIR/scripts/build_cuda126.sh" > "$AD/build.log" 2>&1
RC=$?
echo "BUILD_EXIT=$RC" | tee -a "$AD/build.log"
echo "=== build finished $(date -Iseconds) exit=$RC ==="
tail -25 "$AD/build.log"
exit $RC
