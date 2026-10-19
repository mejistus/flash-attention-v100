#!/usr/bin/env python
# ======================================================================================
# collect_ptxas_stats.py — per-kernel {registers, smem, spills} from `ptxas -v` output.
#
# How to capture ptxas -v on this build (no setup.py change needed):
#
#   ATTENTION_DEBUG=1 python setup.py build_ext 2>&1 | tee build_debug.log
#       # setup.py (lines 65-73) adds -g --keep --keep-dir build -Xptxas -v
#       # NOTE: -g/--keep change artifact retention; try NVCC_APPEND_FLAGS for a
#       # "flags-pure" run when supported by the installed torch cpp_extension:
#   NVCC_APPEND_FLAGS="-Xptxas -v" python setup.py build_ext 2>&1 | tee build_ptxas.log
#
# Usage:
#   python collect_ptxas_stats.py build.log --json stats.json
#   python collect_ptxas_stats.py build_old.log --compare build_new.log
#   python collect_ptxas_stats.py build.log --expect-arch 70 --fail-on-spill-bytes 0
#
# Notes on the ptxas -v text format (CUDA 12.6, verified on this machine):
#   ptxas info    : Compiling entry function '_Z...' for 'sm_70'
#   ptxas info    : Function properties for _Z...
#       0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#   ptxas info    : Used 12 registers, used 0 barriers, 376 bytes cmem[0]
#   (smem clause only present when nonzero: "Used 255 registers, 88064 bytes smem, ...")
# ======================================================================================
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

RE_COMPILE = re.compile(r"ptxas info\s*:\s*Compiling entry function '([^']+)' for 'sm_(\d+)'")
RE_PROPS = re.compile(r"ptxas info\s*:\s*Function properties for ([^\s']+)")
RE_SPILL = re.compile(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads")
RE_USED = re.compile(r"ptxas info\s*:\s*Used (\d+) registers(.*)")
RE_RE_TAIL_SMEM = re.compile(r"(\d+) bytes smem")
RE_RE_TAIL_BAR = re.compile(r"used (\d+) barriers")
RE_RE_TAIL_CMEM = re.compile(r"(\d+) bytes cmem\[(\d+)\]")


def _demangle(names):
    cxxfilt = shutil.which("c++filt")
    uniq = sorted(set(names))
    if not cxxfilt or not uniq:
        return {}
    try:
        out = subprocess.run([cxxfilt] + uniq, capture_output=True, text=True, check=True,
                             timeout=30)
        return dict(zip(uniq, out.stdout.strip().splitlines()))
    except Exception:
        return {}


def parse_log(path: str):
    kernels = []
    cur = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_COMPILE.search(line)
            if m:
                cur = {
                    "mangled": m.group(1), "arch": int(m.group(2)),
                    "registers": None, "smem_bytes": 0, "barriers": None,
                    "cmem": {}, "stack_frame": None,
                    "spill_stores": None, "spill_loads": None,
                }
                kernels.append(cur)
                continue
            m = RE_PROPS.search(line)
            if m:
                if cur is None or cur["mangled"] != m.group(1):
                    cur = {"mangled": m.group(1), "arch": None, "registers": None,
                           "smem_bytes": 0, "barriers": None, "cmem": {},
                           "stack_frame": None, "spill_stores": None, "spill_loads": None}
                    kernels.append(cur)
                continue
            if cur is None:
                continue
            m = RE_SPILL.search(line)
            if m:
                cur["stack_frame"], cur["spill_stores"], cur["spill_loads"] = (
                    int(m.group(1)), int(m.group(2)), int(m.group(3)))
                continue
            m = RE_USED.search(line)
            if m:
                cur["registers"] = int(m.group(1))
                tail = m.group(2)
                ms = RE_RE_TAIL_SMEM.search(tail)
                if ms:
                    cur["smem_bytes"] = int(ms.group(1))
                mb = RE_RE_TAIL_BAR.search(tail)
                if mb:
                    cur["barriers"] = int(mb.group(1))
                for mc in RE_RE_TAIL_CMEM.finditer(tail):
                    cur["cmem"][f"cmem[{mc.group(2)}]"] = int(mc.group(1))
    # demangle / dedupe bookkeeping
    dem = _demangle([k["mangled"] for k in kernels])
    for k in kernels:
        k["demangled"] = dem.get(k["mangled"], k["mangled"])
    return kernels


def summarize(kernels):
    return {
        "num_entry_functions": len(kernels),
        "max_registers": max((k["registers"] or 0) for k in kernels) if kernels else 0,
        "total_smem_bytes": sum(k["smem_bytes"] or 0 for k in kernels),
        "total_spill_bytes": sum((k["spill_stores"] or 0) + (k["spill_loads"] or 0)
                                 for k in kernels),
        "arches_seen": sorted({k["arch"] for k in kernels if k["arch"] is not None}),
    }


def print_table(kernels):
    print(f"{'arch':>4} {'regs':>5} {'smem':>8} {'spill_s':>8} {'spill_l':>8} {'stack':>7}  kernel")
    for k in kernels:
        print(f"{k['arch']!s:>4} {k['registers']!s:>5} {k['smem_bytes']!s:>8} "
              f"{k['spill_stores']!s:>8} {k['spill_loads']!s:>8} {k['stack_frame']!s:>7}  "
              f"{k['demangled']}")


def compare(old, new):
    """old/new: kernel lists -> delta table keyed by demangled name."""
    def key(k):
        return (k["demangled"], k["arch"])

    o = {key(k): k for k in old}
    n = {key(k): k for k in new}
    rows = []
    for kk in sorted(set(o) | set(n)):
        a, b = o.get(kk), n.get(kk)
        def d(field):
            av = a.get(field) if a else None
            bv = b.get(field) if b else None
            if av is None or bv is None:
                return "?"
            return f"{bv - av:+d}" if bv != av else "="
        rows.append({
            "kernel": kk[0], "arch": kk[1],
            "new": b is not None, "removed": a is not None and b is None,
            "regs": d("registers"), "smem": d("smem_bytes"),
            "spill_stores": d("spill_stores"), "spill_loads": d("spill_loads"),
        })
    print(f"{'regs':>10} {'smem':>10} {'spill_s':>10} {'spill_l':>10}  kernel")
    for r in rows:
        tag = "" if r["new"] else " [REMOVED]"
        print(f"{r['regs']:>10} {r['smem']:>10} {r['spill_stores']:>10} "
              f"{r['spill_loads']:>10}  {r['kernel']}{tag}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="build log containing ptxas -v output")
    ap.add_argument("--json", default=None, help="write parsed stats JSON here")
    ap.add_argument("--compare", default=None, help="second build log to diff against")
    ap.add_argument("--expect-arch", type=int, default=None, help="e.g. 70")
    ap.add_argument("--fail-on-spill-bytes", type=int, default=None,
                    help="exit 3 if total spill bytes exceed this")
    ap.add_argument("--max-registers", type=int, default=None,
                    help="exit 4 if any kernel uses more than this")
    args = ap.parse_args(argv)

    kernels = parse_log(args.log)
    if not kernels:
        print("ERROR: no 'ptxas info : Compiling entry function' lines found — was the "
              "build run with -Xptxas -v (ATTENTION_DEBUG=1 or NVCC_APPEND_FLAGS)?")
        return 5

    print(f"# {args.log}: {len(kernels)} compiled entry function records")
    print_table(kernels)
    stats = {"source_logs": [args.log], "kernels": kernels, "summary": summarize(kernels)}
    print(f"\nsummary: {json.dumps(stats['summary'])}")

    if args.compare:
        print(f"\n# diff vs {args.compare}")
        other = parse_log(args.compare)
        stats["compare"] = compare(other, kernels)

    rc = 0
    if args.expect_arch is not None:
        bad = [k["mangled"] for k in kernels if k["arch"] != args.expect_arch]
        if bad:
            print(f"FAIL: {len(bad)} kernels not compiled for sm_{args.expect_arch}: {bad[:5]}")
            rc = 4
    if args.fail_on_spill_bytes is not None:
        tot = stats["summary"]["total_spill_bytes"]
        if tot > args.fail_on_spill_bytes:
            print(f"FAIL: total spill bytes {tot} > {args.fail_on_spill_bytes}")
            rc = 3
    if args.max_registers is not None:
        bad = [k["demangled"] for k in kernels if (k["registers"] or 0) > args.max_registers]
        if bad:
            print(f"FAIL: kernels above {args.max_registers} registers: {bad}")
            rc = 4

    if args.json:
        # attach compare if the JSON output is consumed by run_all.py
        with open(args.json, "w") as f:
            json.dump(stats, f, indent=1)
        print(f"wrote {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
