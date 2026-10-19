#!/usr/bin/env python
# ======================================================================================
# * CPU acceptance runner for pre-build / no-GPU machines: runs every test_*.py script
# * in this directory (each in its own process) plus the API drop-in parity check
# * (tests/parity_check.py) and reports each result as PASS / PARTIAL / SKIP / FAIL:
# *
# *   PASS    - test script rc==0 with at least one real check; parity PASS.
# *   SKIP    - test script rc==0 but every check was skipped (e.g. the extension
# *             smoke test with no compiled extension) -> nothing was verified, so
# *             this is NOT counted as a pass.
# *   PARTIAL - parity PARTIAL: only known gaps / skipped metadata, or AST fallback
# *             (no runtime signatures compared). Non-fatal.
# *   FAIL    - test script rc!=0, or parity rc==1 / "RESULT PARITY FAIL".
# *
# * Exit code: 1 only when at least one real FAIL; SKIP/PARTIAL keep exit 0.
# *
# * The repo root is placed on PYTHONPATH for the child processes, so `import
# * flash_attn` resolves even when this runner is executed as a plain script from an
# * unrelated directory. Parity mode is chosen as in tests/run_all.py: runtime when
# * torch and the compiled extension are importable, runtime+mock when only torch is,
# * and --ast (source-only) when torch is missing.
# *
# * Usage: /mnt/envs/v100/bin/python tests/run_all_cpu.py [--repo DIR]
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# "== <title>: <passed>/<total> passed, <failed> failed, <skipped> skipped =="
SUMMARY_RE = re.compile(r"^== .*: (\d+)/(\d+) passed, (\d+) failed, (\d+) skipped ==", re.M)
PARITY_RE = re.compile(r"RESULT\s+PARITY\s+(PASS|PARTIAL|FAIL|SKIP)")


def run(cmd, env):
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def echo(p):
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)


def classify_test(p):
    """PASS / SKIP / FAIL for one test-script result."""
    if p.returncode != 0:
        return "FAIL"
    last = None
    for last in SUMMARY_RE.finditer(p.stdout):
        pass
    if last is None:
        # A script that exits 0 without printing its summary line proves nothing:
        # do not let silent no-ops count as PASS (red-team R8).
        return "FAIL"
    if int(last.group(2)) == 0 and int(last.group(4)) > 0:
        return "SKIP"  # nothing but skips ran: not evidence of a passing build
    return "PASS"


def parity_command(repo, env):
    """Return (cmd, mode) for parity_check.py — same mode logic as run_all.py."""
    parity = os.path.join(HERE, "parity_check.py")
    if run([sys.executable, "-c", "import torch"], env).returncode != 0:
        return ([sys.executable, parity, "--repo", repo, "--ast", "--allow-known-gaps"],
                "AST fallback: torch not importable, runtime signatures NOT compared")
    if run([sys.executable, "-c", "import torch, flash_attn_2_cuda"], env).returncode == 0:
        return ([sys.executable, parity, "--repo", repo, "--allow-known-gaps"],
                "runtime (compiled extension importable)")
    return ([sys.executable, parity, "--repo", repo, "--mock", "--mock-dir", HERE,
             "--allow-known-gaps"],
            "runtime+mock (extension not built)")


def main():
    ap = argparse.ArgumentParser(description="CPU-only acceptance runner (no GPU needed)")
    ap.add_argument("--repo", default=os.path.dirname(HERE),
                    help="repository root (default: parent of this script's directory)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")

    tests = sorted(f for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py"))
    results = []
    for t in tests:
        print(f"\n########## {t} ##########")
        p = run([sys.executable, os.path.join(HERE, t)], env)
        echo(p)
        results.append((t, classify_test(p), f"exit={p.returncode}"))

    cmd, mode = parity_command(repo, env)
    print(f"\n########## parity_check.py ({mode}) ##########")
    p = run(cmd, env)
    echo(p)
    m = PARITY_RE.search(p.stdout)
    status = m.group(1) if m else ("FAIL" if p.returncode else "PASS")
    if mode.startswith("AST fallback") and status == "PASS":
        status = "PARTIAL"  # runtime signatures were not compared
    results.append(("parity_check.py", status, f"exit={p.returncode}; {mode}"))

    print("\n================ summary ================")
    for name, status, detail in results:
        print(f"{status:7} {name}  ({detail})")
    counts = {k: sum(1 for _, s, _ in results if s == k)
              for k in ("PASS", "PARTIAL", "SKIP", "FAIL")}
    print(f"-- {len(results)} checks: {counts['PASS']} PASS, {counts['PARTIAL']} PARTIAL, "
          f"{counts['SKIP']} SKIP, {counts['FAIL']} FAIL")
    if counts["SKIP"]:
        print("-- SKIP means the check did not run (e.g. no compiled extension); it is NOT a pass")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
