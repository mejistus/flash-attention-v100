#!/usr/bin/env python
# ======================================================================================
# run_all.py — GPU-free acceptance runner: build evidence + artifacts + api parity +
#              CPU mock wrapper tests + static SASS/ISA checks -> machine-readable report.
#
# Report JSON schema (schema_version=1)
# -------------------------------------
# {
#   "schema_version": 1,
#   "generated_at": "<iso8601>",
#   "repo": "<abs path>",
#   "git": {"rev": "<sha|null>", "dirty": true|false|null},
#   "environment": {"python": "...", "torch": "...", "cuda": "...", "nvcc": "...",
#                   "gpu_present": false, "hostname": "..."},
#   "build": {"mode": "default|MMA_NATIVE|MMA_884|unknown", "log": "path|null",
#             "exit_code": int|null, "so_path": "path|null",
#             "so_sha256": "hex|null", "so_size": int|null,
#             "ptxas_stats_json": "path|null", "ptxas_summary": {...}|null},
#   "sections": {
#       "build":       SECTION,
#       "artifacts":   SECTION,
#       "api_parity":  SECTION,
#       "cpu_mock":    SECTION,
#       "static_sass": SECTION
#   },
#   "overall": "PASS|FAIL|PARTIAL",
#   "claims":      ["..."],        # what a PASS does prove
#   "not_verified":["..."]         # things that still require a real V100
# }
# SECTION = {"status": "PASS|FAIL|PARTIAL|SKIP", "detail": "...",
#            "evidence": ["path", ...], "metrics": {...}}
#
# Usage:
#   python run_all.py --repo /path/to/flash-attention-v100 \
#       [--so PATH] [--build-log path] [--build-exit 0] [--build-mode default] \
#       [--sections build,artifacts,api_parity,cpu_mock,static_sass] \
#       [--artifacts-dir tests/artifacts] [--python /mnt/envs/v100/bin/python]
#       [--no-exec]     # only assemble the report from existing artifacts
# ======================================================================================
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_VERSION = 1

NOT_VERIFIED = [
    "kernel numeric equality vs a real V100 (mock covers wrapper/plumbing only)",
    "race conditions / missing __syncthreads / atomics ordering",
    "dropout: rng offsets, dmask encoding, determinism property",
    "rotary, leftpad_k, paged block_table paths in fwd_kvcache",
    "performance/occupancy; real shared-memory limits enforcement on device",
    "behavior of C++ TORCH_CHECKs that require a CUDA context (getCurrentDeviceProperties)",
]

CLAIMS = [
    "python wrapper packages import and expose the documented call surface",
    "wrapper semantics (layout, padding, GQA map, masks, varlen packing, kvcache "
    "in-place update, grad flow) behave as designed on CPU with reference math",
    "embedded cubins target sm_70 only (unless static_sass=PARTIAL)",
    "no post-Volta SASS opcodes in the built binary (only with cuobjdump/nvdisasm)",
]


def sh(cmd, cwd=None, env=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_line(text):
    m = re.findall(r"RESULT\s+(\w+)\s+(PASS|FAIL|PARTIAL|SKIP)", text)
    return m[-1] if m else None


def section(status, detail="", evidence=None, metrics=None):
    return {"status": status, "detail": detail, "evidence": evidence or [],
            "metrics": metrics or {}}


# --------------------------------------------------------------------------------------
def env_info(python):
    script = (
        "import json,sys,platform\n"
        "d={'python':sys.version.split()[0],'executable':sys.executable,"
        "'platform':platform.platform(),'hostname':platform.node()}\n"
        "try:\n"
        "    import torch\n"
        "    d['torch']=torch.__version__\n"
        "    d['torch_cuda']=torch.version.cuda\n"
        "except Exception:\n"
        "    d['torch']=None\n"
        "print(json.dumps(d))\n"
    )
    r = sh([python, "-c", script])
    info = {}
    if r.returncode == 0:
        try:
            info = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            pass
    nvcc = shutil.which("nvcc") or "/mnt/envs/v100/bin/nvcc"
    if os.path.exists(nvcc):
        rr = sh([nvcc, "--version"])
        m = re.search(r"release ([\d.]+)", rr.stdout)
        info["nvcc"] = m.group(1) if m else "unknown"
    else:
        info["nvcc"] = None
    info["gpu_present"] = False  # by construction of this harness
    return info


def git_info(repo):
    if not os.path.isdir(os.path.join(repo, ".git")):
        return {"rev": None, "dirty": None}
    rev = sh(["git", "-C", repo, "rev-parse", "HEAD"]).stdout.strip() or None
    dirty = bool(sh(["git", "-C", repo, "status", "--porcelain"]).stdout.strip())
    return {"rev": rev, "dirty": dirty}


# --------------------------------------------------------------------------------------
def section_build(args, artd):
    so = args.so
    ev = []
    metrics = {}
    if not args.build_log:
        return section("SKIP", "no --build-log; build not executed by this runner",
                       evidence=[], metrics={}), so
    log = args.build_log
    ev.append(os.path.abspath(log))
    if args.build_exit is None:
        return (section("SKIP", "--build-log given but --build-exit missing (cannot audit exit code)",
                        ev), so)
    text = open(log, errors="replace").read()
    # ptxas stats
    ptxas_json = os.path.join(artd, "ptxas_stats.json")
    stats = None
    try:
        r = sh([args.python, os.path.join(HERE, "collect_ptxas_stats.py"), log,
                "--json", ptxas_json, "--expect-arch", "70"])
        if r.returncode == 0 and os.path.exists(ptxas_json):
            stats = json.load(open(ptxas_json))
            ev.append(ptxas_json)
            metrics["ptxas"] = stats["summary"]
    except Exception as e:
        metrics["ptxas_error"] = str(e)
    ok = args.build_exit == 0
    if not so or not os.path.exists(so):
        ok = False
        metrics["so_present"] = False
    else:
        metrics.update(so_path=os.path.abspath(so), so_size=os.path.getsize(so),
                       so_sha256=sha256(so))
    err_hits = len(re.findall(r"error:|Error:", text))
    metrics["log_error_lines"] = err_hits
    return (section("PASS" if ok else "FAIL",
                    f"build exit={args.build_exit}, .so present={bool(so and os.path.exists(so))}, "
                    f"log errors={err_hits}", ev, metrics), so)


def section_artifacts(args, artd, so, torch_ok):
    if not torch_ok:
        return section("SKIP", "torch not importable by target python; cannot probe extension")
    ev, metrics = [], {}
    script = (
        "import json,os,sys\n"
        "out={}\n"
        "try:\n"
        " import torch\n"
        " out['torch']=torch.__version__\n"
        "except Exception as e:\n"
        " out['torch_error']=str(e)\n"
        "for name in ('flash_attn_2_cuda',):\n"
        " try:\n"
        "  import importlib\n"
        "  m=importlib.import_module(name)\n"
        "  out[name]={'file':getattr(m,'__file__',None),'dir':getattr(m,'__file__',None) and os.path.dirname(os.path.abspath(m.__file__)),"
        "'funcs':[f for f in ('fwd','bwd','varlen_fwd','varlen_bwd','fwd_kvcache') if callable(getattr(m,f,None))]}\n"
        " except Exception as e:\n"
        "  out[name]={'error':str(e)}\n"
        "try:\n"
        " import importlib.metadata as md\n"
        " d=md.distribution('flash-attn')\n"
        " out['dist']={'version':d.version,'name':d.metadata['Name'],'top_level':d.read_text('top_level.txt')}\n"
        "except Exception as e:\n"
        " out['dist']={'error':str(e)}\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ)
    so_dir = os.path.dirname(os.path.abspath(so)) if so else None
    if so_dir:
        env["PYTHONPATH"] = so_dir + os.pathsep + env.get("PYTHONPATH", "")
    r = sh([args.python, "-c", script], env=env)
    out = {}
    if r.returncode == 0:
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            out = {"parse_error": r.stdout[-500:]}
    rep = os.path.join(artd, "artifacts_probe.json")
    with open(rep, "w") as f:
        json.dump(out, f, indent=1)
    ev.append(rep)

    problems = []
    ext = out.get("flash_attn_2_cuda", {})
    ext2 = out.get("flash_attn_2_cuda", {})
    if ext.get("error"):
        problems.append(f"import flash_attn_2_cuda: {ext['error']}")
    elif len(ext.get("funcs", [])) != 5:
        problems.append(f"missing extension funcs: {ext.get('funcs')}")
    if ext2.get("error"):
        problems.append(f"import flash_attn_2_cuda alias: {ext2['error']}")
    dist = out.get("dist", {})
    if dist.get("error"):
        problems.append(f"dist-info: {dist['error']}")
    else:
        if dist.get("version") != "2.8.3":
            problems.append(f"dist version {dist.get('version')} != 2.8.3")
        if (dist.get("top_level") or "").strip() != "flash_attn":
            problems.append(f"top_level.txt {dist.get('top_level')!r}")
    # symlink presence next to the module
    d = ext.get("dir")
    if d:
        link = os.path.join(d, "flash_attn_2_cuda.so")
        if os.path.islink(link):
            metrics["symlink"] = f"-> {os.readlink(link)}"
        else:
            problems.append("flash_attn_2_cuda.so symlink missing")
    metrics["probe"] = out
    status = "FAIL" if problems else ("SKIP" if not so else "PASS")
    return section(status, "; ".join(problems) or "extension + alias + dist-info OK", ev, metrics)


def section_api_parity(args, artd, so, torch_ok):
    ev = []
    spec = os.path.join(args.repo, "research", "03_api_spec.json")
    cmd = [args.python, os.path.join(HERE, "parity_check.py"), "--repo", args.repo]
    ast_only = not torch_ok
    if ast_only:
        cmd += ["--ast"]
        detail_extra = " (torch missing -> AST fallback, runtime signatures NOT compared)"
    elif not so:
        cmd += ["--mock", "--mock-dir", HERE]
        detail_extra = " (mock-injected runtime mode)"
    else:
        detail_extra = ""
    if os.path.exists(spec):
        cmd += ["--spec", spec]
    cmd += ["--json", os.path.join(artd, "api_parity.json"), "--allow-known-gaps"]
    r = sh(cmd)
    jout = os.path.join(artd, "api_parity.json")
    log = os.path.join(artd, "api_parity.log")
    open(log, "w").write(r.stdout + "\n" + r.stderr)
    ev += [log, jout]
    parsed = result_line(r.stdout)
    status = parsed[1] if parsed else ("FAIL" if r.returncode else "PASS")
    if ast_only and status == "PASS":
        status = "PARTIAL"
    metrics = {}
    if os.path.exists(jout):
        try:
            data = json.load(open(jout))
            metrics = data.get("summary", {})
            metrics["known_gaps"] = data.get("known_gaps", [])
        except Exception:
            pass
    detail = f"exit={r.returncode} result={status}{detail_extra}"
    if metrics.get("known_gaps"):
        detail += f"; known gaps: {metrics['known_gaps']}"
    return section(status, detail, ev, metrics)


def section_cpu_mock(args, artd, torch_ok):
    if not torch_ok:
        return section("SKIP", "torch not importable by target python")
    ev = []
    log = os.path.join(artd, "cpu_mock.log")
    r = sh([args.python, os.path.join(HERE, "test_wrapper_cpu.py")], cwd=HERE)
    with open(log, "w") as f:
        f.write(r.stdout + "\n" + r.stderr)
    ev.append(log)
    m = re.search(r"(\d+)/(\d+) passed", r.stdout)
    metrics = {"passed": int(m.group(1)), "total": int(m.group(2))} if m else {}
    status = "PASS" if r.returncode == 0 else "FAIL"
    fails = re.findall(r"^FAIL\s+(\S+)", r.stdout, re.M)
    return section(status, f"exit={r.returncode}; fails={fails[:8]}", ev, metrics)


def section_static_sass(args, artd, so):
    if not so or not os.path.exists(so):
        return section("SKIP", "no .so available (build first)", [])
    outd = os.path.join(artd, "sass")
    os.makedirs(outd, exist_ok=True)
    sass_sh = next((p for p in (os.path.join(HERE, "verify_sass.sh"),
                                os.path.join(HERE, "..", "scripts", "verify_sass.sh"),
                                os.path.join(os.environ.get("FA_V100_REPO", ""), "scripts", "verify_sass.sh"))
                    if os.path.exists(p)), os.path.join(HERE, "verify_sass.sh"))
    r = sh(["bash", sass_sh, "--so", so, "--out-dir", outd])
    log = os.path.join(artd, "static_sass.log")
    with open(log, "w") as f:
        f.write(r.stdout + "\n" + r.stderr)
    parsed = result_line(r.stdout)
    status = parsed[1] if parsed else ("FAIL" if r.returncode else "PASS")
    metrics = {}
    m = re.search(r"HMMA\.884 occurrences: (\d+)", r.stdout)
    if m:
        metrics["hmma884_count"] = int(m.group(1))
    lelf = os.path.join(outd, "lelf.txt")
    if os.path.exists(lelf):
        metrics["lelf"] = open(lelf, errors="replace").read().strip()[:500]
    missing_tools = "not found -> fallback" in r.stdout
    if missing_tools:
        status = "PARTIAL"
    return section(status, "cuobjdump unavailable -> arch-only" if missing_tools else
                   f"exit={r.returncode}", [log] + [os.path.join(outd, f)
                                                     for f in sorted(os.listdir(outd))], metrics)


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get(
        "FA_V100_REPO", "/home/eculid/Documents/Github/flash-attention-v100"))
    ap.add_argument("--so", default=os.environ.get("FA_V100_SO"))
    ap.add_argument("--build-log", default=None)
    ap.add_argument("--build-exit", type=int, default=None)
    ap.add_argument("--build-mode", default="unknown",
                    choices=["default", "MMA_NATIVE", "MMA_884", "unknown"])
    ap.add_argument("--sections", default="build,artifacts,api_parity,cpu_mock,static_sass")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)

    artd = args.artifacts_dir or os.path.join(args.repo, "tests", "artifacts")
    os.makedirs(artd, exist_ok=True)
    wanted = set(args.sections.split(","))

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "repo": os.path.abspath(args.repo),
        "git": git_info(args.repo),
        "environment": env_info(args.python),
        "build": {"mode": args.build_mode, "log": args.build_log, "exit_code": args.build_exit,
                  "so_path": os.path.abspath(args.so) if args.so else None},
        "sections": {},
        "overall": "PASS",
        "claims": CLAIMS,
        "not_verified": NOT_VERIFIED,
    }

    so = args.so
    torch_ok = bool(report["environment"].get("torch"))
    if "build" in wanted:
        sec, so = section_build(args, artd)
        report["sections"]["build"] = sec
    if "artifacts" in wanted:
        report["sections"]["artifacts"] = section_artifacts(args, artd, so, torch_ok)
    if "api_parity" in wanted:
        report["sections"]["api_parity"] = section_api_parity(args, artd, so, torch_ok)
    if "cpu_mock" in wanted:
        report["sections"]["cpu_mock"] = section_cpu_mock(args, artd, torch_ok)
    if "static_sass" in wanted:
        report["sections"]["static_sass"] = section_static_sass(args, artd, so)

    statuses = [s["status"] for s in report["sections"].values()]
    if "FAIL" in statuses:
        report["overall"] = "FAIL"
    elif any(s in ("PARTIAL", "SKIP") for s in statuses):
        report["overall"] = "PARTIAL"

    jpath = args.json or os.path.join(artd, "verification_report.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {jpath}")

    md = ["# GPU-free verification report", "",
          f"- generated: {report['generated_at']}",
          f"- repo: `{report['repo']}` @ `{report['git']['rev']}`"
          f"{' (dirty)' if report['git']['dirty'] else ''}",
          f"- env: python {report['environment'].get('python')}, "
          f"torch {report['environment'].get('torch')}, nvcc {report['environment'].get('nvcc')}",
          f"- overall: **{report['overall']}**", "",
          "| section | status | detail |", "|---|---|---|"]
    for name, s in report["sections"].items():
        md.append(f"| {name} | {s['status']} | {s['detail'][:160]} |")
    md += ["", "## What a PASS proves", ""] + [f"- {c}" for c in CLAIMS]
    md += ["", "## NOT verified without a V100 GPU", ""] + [f"- {c}" for c in NOT_VERIFIED]
    mdtext = "\n".join(md) + "\n"
    if args.md:
        with open(args.md, "w") as f:
            f.write(mdtext)
        print(f"wrote {args.md}")
    print(mdtext)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
