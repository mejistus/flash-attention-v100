#!/usr/bin/env bash
# ======================================================================================
# build_matrix.sh — build flash_attn_v100 in scratch copies for each MMA mode and
#                   collect per-mode evidence (ptxas stats + static SASS arch/simplified ISA).
#
# Modes (repo env vars, see setup.py L108-122 and include/mat_mul.h):
#   default     : no extra -D; kernel picks its default tile path
#   MMA_NATIVE=1: -DMMA_NATIVE  (native sm_70 HMMA.884 m8n8k4 path)
#   MMA_884=1   : -DMMA_884     (explicit m8n8k4 tile shape)
#
# All three must produce sm_70-only cubins; the SASS ISA check needs nvdisasm/cuobjdump
# (optional: mamba install -n v100 -c nvidia cuda-nvdisasm cuda-cuobjdump), otherwise
# verify_sass.sh degrades to an arch-tag-only PARTIAL.
#
# Nothing is written into the source repo: each mode builds in $WORK/<mode>/src.
#
# Usage:
#   bash build_matrix.sh [--repo DIR] [--workdir DIR] [--python PY]
#                        [--modes "default,MMA_NATIVE,MMA_884"] [--release]
#   --release : build WITHOUT ATTENTION_DEBUG (no -g/--keep/-Xptxas -v; faster, but then
#               collect_ptxas_stats has no data -> stats section is skipped)
# ======================================================================================
set -u

REPO="${FA_V100_REPO:-/home/eculid/Documents/Github/flash-attention-v100}"
WORK="${TMPDIR:-/tmp}/opencode/build_matrix"
PY="${FA_V100_PYTHON:-/mnt/envs/v100/bin/python}"
MODES="default,MMA_NATIVE,MMA_884"
DEBUG=1

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --workdir) WORK="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --release) DEBUG=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
HERE="$ROOT"                       # verify_sass.sh lives next to this script;
COLLECT="$ROOT/../tests/collect_ptxas_stats.py"   # stats parser lives in ../tests/
ENVBIN="$(dirname "$PY")"

# ---- toolchain pinning -----------------------------------------------------------------
# nvcc 12.6 rejects the system gcc (16.x); the conda env ships gcc 12.4 which works.
CONDA_GCC="$ENVBIN/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="$ENVBIN/x86_64-conda-linux-gnu-g++"
if [ ! -x "$CONDA_GXX" ]; then
    echo "FATAL: $CONDA_GXX not found; set --python to the env that has the conda gcc" >&2
    exit 2
fi

"$PY" -c "import torch" 2>/dev/null || { echo "FATAL: torch not importable by $PY (build needs it)" >&2; exit 2; }
[ -x "$ENVBIN/nvcc" ] || { echo "FATAL: $ENVBIN/nvcc not found" >&2; exit 2; }

mkdir -p "$WORK"
SUMMARY="$WORK/summary.tsv"
printf "mode\texit\tsm70_only\tisa\tso_sha256_12\tso_bytes\n" > "$SUMMARY"

for MODE in ${MODES//,/ }; do
    echo "==================== MODE: $MODE ===================="
    SRC="$WORK/$MODE/src"
    LOG="$WORK/$MODE/build.log"
    mkdir -p "$WORK/$MODE"
    rm -rf "$SRC"
    rsync -a --exclude build --exclude .git --exclude '__pycache__' \
          --exclude '*.egg-info' "$REPO/" "$SRC/"

    # ---- build env ----
    export PATH="$ENVBIN:$PATH"
    export CC="$CONDA_GCC" CXX="$CONDA_GXX" CUDAHOSTCXX="$CONDA_GXX"
    export FLASH_ATTN_MIN_CUDA="${FLASH_ATTN_MIN_CUDA:-12.6}"
    export NVCC_THREADS="${NVCC_THREADS:-8}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    unset MMA_NATIVE MMA_884
    case "$MODE" in
        default) ;;
        MMA_NATIVE) export MMA_NATIVE=1 ;;
        MMA_884)    export MMA_884=1 ;;
    esac
    if [ "$DEBUG" -eq 1 ]; then export ATTENTION_DEBUG=1; else unset ATTENTION_DEBUG; fi

    # ---- build (in-place .so inside the scratch copy) ----
    ( cd "$SRC" && "$PY" setup.py build_ext --inplace ) >"$LOG" 2>&1
    RC=$?
    echo "build exit=$RC log=$LOG"

    SO=$(ls "$SRC"/flash_attn_v100_cuda*.so 2>/dev/null | head -1)
    SHA=$( [ -n "$SO" ] && sha256sum "$SO" | cut -c1-12 || echo "-" )
    BYTES=$( [ -n "$SO" ] && stat -c%s "$SO" || echo 0 )

    # ---- ptxas stats ----
    if [ "$DEBUG" -eq 1 ] && [ -f "$LOG" ]; then
        "$PY" "$COLLECT" "$LOG" --expect-arch 70 \
              --json "$WORK/$MODE/ptxas_stats.json" >"$WORK/$MODE/ptxas_stats.txt" 2>&1 \
            && echo "ptxas stats: $WORK/$MODE/ptxas_stats.txt"
    fi

    # ---- static SASS ----
    SM70="?"; ISA="?"
    if [ -n "$SO" ]; then
        OUT=$(bash "$HERE/verify_sass.sh" --so "$SO" --out-dir "$WORK/$MODE/sass" 2>&1)
        echo "$OUT" > "$WORK/$MODE/verify_sass.out"
        echo "$OUT" | grep -q "RESULT STATIC_SASS PARTIAL" && ISA="arch-only(PARTIAL)"
        echo "$OUT" | grep -q "RESULT STATIC_SASS PASS"    && ISA="checked(PASS)"
        echo "$OUT" | grep -q "RESULT STATIC_SASS FAIL"    && ISA="FAIL"
        if echo "$OUT" | grep -q "RESULT STATIC_SASS FAIL"; then SM70="no"; else SM70="yes"; fi
    else
        SM70="no-so"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$MODE" "$RC" "$SM70" "$ISA" "$SHA" "$BYTES" >> "$SUMMARY"
done

echo
echo "==================== SUMMARY ($SUMMARY) ===================="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

# PASS criteria for the matrix as a whole:
#   1. every mode exit=0 and produced a .so
#   2. sm70_only=yes for all modes
#   3. isa is PASS everywhere (FAIL anywhere is a blocking failure)
#   4. all three ptxas_stats.json exist (when built with ATTENTION_DEBUG)
FAILS=$(awk -F'\t' 'NR>1 && ($2!=0 || $3!="yes" || $4=="FAIL") {n++} END {print n+0}' "$SUMMARY")
if [ "$FAILS" -eq 0 ]; then
    echo "RESULT BUILD_MATRIX PASS"
    exit 0
else
    echo "RESULT BUILD_MATRIX FAIL ($FAILS mode(s) failed criteria)"
    exit 1
fi
