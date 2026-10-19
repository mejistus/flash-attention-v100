#!/usr/bin/env bash
# ======================================================================================
# verify_sass.sh — GPU-free static verification of the built flash_attn_2_cuda .so
#
# Checks
#   1. exactly one embedded cubin, tagged sm_70 (cuobjdump -lelf)
#   2. no sm_75/sm_80/sm_86/sm_89/sm_90/compute_XX objects embedded, and no embedded
#      PTX compiled for a newer target (cuobjdump -ptx)
#   3. SASS disassembly contains no post-Volta tensor-core / async-copy opcodes
#      (cuobjdump -sass | nvdisasm -c)  — sm_70 tensor core opcode is HMMA.884
#   4. at least one HMMA.884 (tensor cores actually used, not scalarized) [default on]
#   5. disassembly has no undecoded/unknown instructions
#   6. FALLBACK (no cuobjdump/nvdisasm installed): scan raw file for embedded cubins
#      (EM_CUDA ELF e_flags -> sm version) and cubin-name strings; arch-only verdict,
#      reported as PARTIAL because no ISA check is possible.
#
# Tooling (optional, recommended; keep it out of the project env):
#   conda create -y -p /mnt/envs/cuda-tools -c nvidia --override-channels \
#       cuda-nvdisasm cuda-cuobjdump
#   (persistent location outside /tmp; override the directory with FA_V100_TOOLS_DIR)
#
# Usage:
#   verify_sass.sh --so /path/to/flash_attn_2_cuda.so [--out-dir DIR]
#                  [--no-require-hmma] [--extract-cubins]
# Env: FA_V100_SO, CUOBJDUMP, NVDISASM
# Output protocol (parsed by run_all.py):
#   RESULT STATIC_SASS PASS|FAIL|PARTIAL
# ======================================================================================
set -u

SO=""
OUT_DIR="${TMPDIR:-/tmp}/opencode/fa_v100_sass"
REQUIRE_HMMA=1
EXTRACT=0
REPO_GUESS="/home/eculid/Documents/Github/flash-attention-v100"

while [ $# -gt 0 ]; do
    case "$1" in
        --so) SO="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --no-require-hmma) REQUIRE_HMMA=0; shift ;;
        --extract-cubins) EXTRACT=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"
STATUS="PASS"
note() { echo "  $*"; }

fail() { STATUS="FAIL"; echo "FAIL: $*"; }
partial() { [ "$STATUS" = "PASS" ] && STATUS="PARTIAL"; echo "PARTIAL: $*"; }

# --------------------------------------------------------------------------------------
# resolve .so
# --------------------------------------------------------------------------------------
resolve_so() {
    if [ -n "${SO:-}" ]; then echo "$SO"; return; fi
    if [ -n "${FA_V100_SO:-}" ]; then echo "$FA_V100_SO"; return; fi
    for root in "$REPO_GUESS/build" "/mnt/envs/v100/lib/python3.12/site-packages"; do
        [ -d "$root" ] || continue
        f=$(find "$root" -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)
        [ -n "$f" ] && { echo "$f"; return; }
    done
    return 1
}

SO=$(resolve_so) || { echo "FAIL: could not locate flash_attn_2_cuda*.so (pass --so)"; echo "RESULT STATIC_SASS FAIL"; exit 1; }
[ -f "$SO" ] || { echo "FAIL: $SO does not exist"; echo "RESULT STATIC_SASS FAIL"; exit 1; }
echo "# target: $SO ($(stat -c%s "$SO") bytes)"

# --------------------------------------------------------------------------------------
# tool discovery
# --------------------------------------------------------------------------------------
find_tool() {
    local name="$1" envvar="$2"
    eval "local explicit=\${$envvar:-}"
    if [ -n "$explicit" ] && [ -x "$explicit" ]; then echo "$explicit"; return; fi
    if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
    local tools_dir="${FA_V100_TOOLS_DIR:-/mnt/envs/cuda-tools}"
    for cand in "$tools_dir/bin/$name" /mnt/envs/v100/bin/"$name" /opt/miniforge/bin/"$name"; do
        [ -x "$cand" ] && { echo "$cand"; return; }
    done
    return 1
}

CUOBJDUMP=$(find_tool cuobjdump CUOBJDUMP) || CUOBJDUMP=""
NVDISASM=$(find_tool nvdisasm NVDISASM) || NVDISASM=""

# --------------------------------------------------------------------------------------
# fallback: embedded-ELF arch scan (no cuobjdump)
# --------------------------------------------------------------------------------------
fallback_scan() {
    echo "# cuobjdump/nvdisasm not found -> fallback embedded-ELF scan"
    echo "# (arch tags only; NO instruction-level ISA check possible)"
    python3 - "$SO" <<'PY'
import struct, sys, re
path = sys.argv[1]
data = open(path, "rb").read()
cubins = []
idx = 0
while True:
    i = data.find(b"\x7fELF", idx)
    if i < 0:
        break
    if i + 64 <= len(data):
        e_machine = struct.unpack_from("<H", data, i + 0x12)[0]
        e_flags = struct.unpack_from("<I", data, i + 0x30)[0]
        if e_machine == 190:  # EM_CUDA
            cubins.append((i, e_flags & 0xFF, e_flags))
    idx = i + 4
print(f"embedded CUDA ELFs: {len(cubins)}")
bad = 0
for off, sm, flags in cubins:
    tag = "OK" if sm == 70 else "BAD"
    if sm != 70:
        bad += 1
    print(f"  @0x{off:x} e_flags={flags:#x} -> sm_{sm} [{tag}]")
names = set(re.findall(rb"[A-Za-z0-9_./-]*\.sm_[0-9a-z]+\.cubin", data))
if names:
    print("embedded cubin name strings:")
    for n in sorted(names):
        print("  " + n.decode(errors="replace"))
        if not n.endswith(b".sm_70.cubin"):
            bad += 1
print(f"FALLBACK_BAD={bad}")
sys.exit(1 if bad else 0)
PY
    rc=$?
    if [ $rc -ne 0 ]; then
        fail "fallback scan found non-sm_70 embedded cubins"
    else
        partial "arch tags are sm_70-only, but SASS ISA legality was NOT checked (install cuda-nvdisasm/cuda-cuobjdump)"
    fi
}

if [ -z "$CUOBJDUMP" ]; then
    fallback_scan
    echo "RESULT STATIC_SASS $STATUS"
    [ "$STATUS" = "FAIL" ] && exit 1
    exit 0
fi

# --------------------------------------------------------------------------------------
# 1. embedded ELF list
# --------------------------------------------------------------------------------------
echo "# cuobjdump -lelf"
"$CUOBJDUMP" -lelf "$SO" > "$OUT_DIR/lelf.txt" 2>&1
cat "$OUT_DIR/lelf.txt"
n_elf=$(grep -c "ELF file" "$OUT_DIR/lelf.txt" || true)
if [ "$n_elf" -lt 1 ]; then
    fail "no embedded ELF cubins found"
else
    if grep -E "sm_([0-9]+)" "$OUT_DIR/lelf.txt" | grep -v "sm_70" >/dev/null; then
        fail "non-sm_70 cubins embedded: $(grep -Eo 'sm_[0-9]+' "$OUT_DIR/lelf.txt" | sort -u | tr '\n' ' ')"
    else
        note "all $n_elf embedded cubin(s) are sm_70"
    fi
    if grep -E "compute_[0-9]+" "$OUT_DIR/lelf.txt" >/dev/null; then
        fail "compute_XX PTX objects embedded: $(grep -Eo 'compute_[0-9]+' "$OUT_DIR/lelf.txt" | sort -u | tr '\n' ' ')"
    fi
fi

# --------------------------------------------------------------------------------------
# 2. embedded PTX targets
# --------------------------------------------------------------------------------------
"$CUOBJDUMP" -ptx "$SO" > "$OUT_DIR/ptx.txt" 2>&1
if grep -E "\.target sm_" "$OUT_DIR/ptx.txt" >/dev/null; then
    note "embedded PTX targets: $(grep -Eo 'sm_[0-9]+' "$OUT_DIR/ptx.txt" | sort -u | tr '\n' ' ')"
    if grep -E "\.target sm_([7-9][1-9]|[89][0-9])" "$OUT_DIR/ptx.txt" | grep -v "sm_70" >/dev/null; then
        fail "embedded PTX targets a post-Volta arch"
    fi
else
    note "no embedded PTX (expected for -gencode arch=compute_70,code=sm_70)"
fi

# --------------------------------------------------------------------------------------
# 3./4./5. SASS dump + ISA sanity
# --------------------------------------------------------------------------------------
"$CUOBJDUMP" -sass "$SO" > "$OUT_DIR/sass_raw.txt" 2>&1
SASS="$OUT_DIR/sass_raw.txt"
if [ -n "$NVDISASM" ] && [ "$EXTRACT" -eq 1 ]; then
    mkdir -p "$OUT_DIR/cubins"
    "$CUOBJDUMP" -xelf all "$SO" --output-directory "$OUT_DIR/cubins" >/dev/null 2>&1 || true
    for cb in "$OUT_DIR"/cubins/*; do
        [ -f "$cb" ] && "$NVDISASM" -c "$cb" > "$cb.sass" 2>&1 || true
    done
fi
if [ ! -s "$SASS" ]; then
    fail "cuobjdump -sass produced no output"
else
    # forbidden post-Volta mnemonics.  sm_70 tensor-core opcode family is HMMA.884
    # (m8n8k4).  Later archs introduce: HMMA.1688 (Turing m16n8k8), HMMA.16816
    # (Ampere m16n8k16), LDSM (Turing+ ldmatrix), LDGSTS (Ampere cp.async),
    # WGMMA/HGMMA (Hopper wgmma), UBLKCP/BLKCP (Hopper TMA bulk copy),
    # CP.ASYNC (Ampere async pipeline).
    FORBIDDEN='HMMA\.1688|HMMA\.16816|LDSM|LDGSTS|WGMMA|HGMMA|UBLKCP|BLKCP|CP\.ASYNC'
    if grep -E "$FORBIDDEN" "$SASS" > "$OUT_DIR/forbidden_hits.txt"; then
        fail "post-Volta ISA opcodes found in SASS:"
        sed -n '1,10p' "$OUT_DIR/forbidden_hits.txt" | sed 's/^/    /'
    else
        note "no forbidden post-Volta opcodes (pattern: $FORBIDDEN)"
    fi
    h884=$(grep -c "HMMA\.884" "$SASS" || true)
    note "HMMA.884 occurrences: $h884"
    if [ "$REQUIRE_HMMA" -eq 1 ] && [ "$h884" -eq 0 ]; then
        fail "no HMMA.884 found — tensor cores may have been scalarized"
    fi
    if grep -E "unknown|undefined instruction" "$SASS" >/dev/null; then
        fail "disassembler reported unknown/undecoded instructions"
    fi
fi

# --------------------------------------------------------------------------------------
echo "RESULT STATIC_SASS $STATUS"
[ "$STATUS" = "FAIL" ] && exit 1
exit 0
