#!/usr/bin/env bash
# ======================================================================================
# capture_ptxas_stats.sh — capture `ptxas -v` register/smem/spill stats for the release
# build, for every kernel translation unit.
#
# Why not FLASH_ATTN_V100_PTXAS_VERBOSE=1? It does add -Xptxas -v (see setup.py), but
# torch's cpp_extension runs ninja with stdout=PIPE and only prints the captured output
# when the build FAILS, so the stats never reach the build log on a successful build.
# This script instead replays the exact per-TU nvcc command recorded by the build
# (build/temp.*/build.ninja via `ninja -t commands`) with `-Xptxas -v` appended.
#
# Usage:
#   bash scripts/capture_ptxas_stats.sh            # all 5 kernel TUs, all in parallel
#   PTXAS_OUT_DIR=/path bash scripts/capture_ptxas_stats.sh
# Then parse:
#   python tests/collect_ptxas_stats.py tests/artifacts/ptxas/ptxas_all.log \
#          --json tests/artifacts/ptxas/ptxas_stats.json --expect-arch 70
# ======================================================================================
set -u
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$(ls -d "$REPO_DIR"/build/temp.linux-x86_64-cpython-312 2>/dev/null | head -1)"
if [ -z "$BUILD_DIR" ]; then
    echo "error: build dir not found (run scripts/build_cuda126.sh first)" >&2
    exit 1
fi
OUT_DIR="${PTXAS_OUT_DIR:-$REPO_DIR/tests/artifacts/ptxas}"
mkdir -p "$OUT_DIR"
# The replayed command invokes targets/.../bin/nvcc (a symlink): nvcc finds its
# components through PATH (see scripts/build_cuda126.sh).
ENV_PREFIX="${V100_ENV:-/mnt/envs/v100}"
export PATH="$ENV_PREFIX/nvvm/bin:$ENV_PREFIX/bin:$PATH"
cd "$BUILD_DIR" || exit 1

TUS="${PTXAS_TUS:-forward backward forward_varlen backward_varlen forward_kvcache}"
CMDS="$OUT_DIR/ninja_commands.txt"
ninja -t commands > "$CMDS" 2>/dev/null || { echo "error: ninja -t commands failed" >&2; exit 1; }

for tu in $TUS; do
    awk -v t="kernel/$tu.o" 'index($0, t) && /nvcc/ {print; exit}' "$CMDS" > "$OUT_DIR/cmd_$tu.sh"
    if [ ! -s "$OUT_DIR/cmd_$tu.sh" ]; then
        echo "error: no nvcc command found for kernel/$tu.o" >&2
        exit 1
    fi
done

run_one() {
    local tu="$1"
    echo "[$tu] replaying nvcc with -Xptxas -v ..."
    ( cd "$BUILD_DIR" && eval "$(cat "$OUT_DIR/cmd_$tu.sh") -Xptxas -v" ) \
        > "$OUT_DIR/$tu.out" 2> "$OUT_DIR/$tu.log"
    echo "[$tu] exit=$?"
}

pids=()
for tu in $TUS; do
    run_one "$tu" &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done

cat "$OUT_DIR"/forward.log "$OUT_DIR"/backward.log "$OUT_DIR"/forward_varlen.log \
    "$OUT_DIR"/backward_varlen.log "$OUT_DIR"/forward_kvcache.log \
    > "$OUT_DIR/ptxas_all.log" 2>/dev/null

echo "ptxas logs: $OUT_DIR (combined: ptxas_all.log)"
exit $rc
