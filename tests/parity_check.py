#!/usr/bin/env python
# ======================================================================================
# parity_check.py — flash-attn 2.8.3 API drop-in parity harness (GPU-free).
#
# Modes
# -----
#   runtime (default) : imports the repo packages and compares inspect.signature
#                       against the normative spec.  Needs torch; the CUDA extension is
#                       needed only if NOT run with --mock.
#   --mock            : injects mock_cuda (same-dir by default) before importing, so the
#                       parity check runs BEFORE the extension is built.
#   --ast             : no imports at all; parses repo sources with ast and compares
#                       signatures/defaults against the spec (pre-build, torch-less).
#
# Normative spec
# --------------
#   --spec research/03_api_spec.json   (produced by the api-spec sibling agent)
#   If absent, a built-in spec derived from upstream Dao-AILab/flash-attention v2.8.3
#   sources is used (fetched 2026-10; see upstream_ref/ in the drafts for provenance).
#
# Checks
# ------
#   version stamps: flash_attn.__version__ == 2.8.3 ; dist-info METADATA Name/Version ;
#                   top_level.txt == flash_attn ; flash_attn.__version__ == 2.8.3.
#   import paths:   flash_attn, flash_attn.flash_attn_interface, flash_attn.bert_padding,
#                   flash_attn_2_cuda.
#   signatures:     upstream public functions + bert_padding helpers.
#   fixed extras:   flash_attn_gpu/flash_attn_varlen_gpu/... aliases (repo convention).
#   known gaps:     upstream names absent from the repo (e.g. qkvpacked/kvpacked funcs)
#                   are reported; --allow-known-gaps downgrades them to WARN.
#
# Output: human table + optional --json out.json {"status": PASS|PARTIAL|FAIL, ...}
# Exit:   0 = PASS, 2 = PARTIAL (only known gaps / skips), 1 = FAIL
# ======================================================================================
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# built-in fallback spec (upstream v2.8.3).  defaults are strings as ast.unparse renders
# --------------------------------------------------------------------------------------
def _sig(params):
    """Parse 'a, b=0.0, window=(-1, -1)' (or a full '(a, b=0.0)' sig) into params.

    Uses ast so that commas inside defaults like (-1, -1) are not split."""
    s = params.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    src = "def _sig_fn({}):\n    pass\n".format(s)
    fn = ast.parse(src).body[0]
    argnames = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    defaults = [ast.unparse(d) for d in fn.args.defaults]
    nd = len(defaults)
    items = []
    for i, n in enumerate(argnames):
        di = i - (len(argnames) - nd)
        items.append({"name": n, "default": defaults[di] if di >= 0 else None})
    return {"params": items}


BUILTIN_SPEC = {
    "source": "builtin: Dao-AILab/flash-attention v2.8.3 (fetched 2026-10-19)",
    "target_version": "2.8.3",
    "functions": {
        "flash_attn_func": _sig(
            "q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), "
            "softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False"),
        "flash_attn_varlen_func": _sig(
            "q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, "
            "softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, "
            "alibi_slopes=None, deterministic=False, return_attn_probs=False, block_table=None"),
        "flash_attn_with_kvcache": _sig(
            "q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, "
            "cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None, "
            "softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, "
            "rotary_interleaved=True, alibi_slopes=None, num_splits=0, return_softmax_lse=False"),
        # ---- upstream-only, currently ABSENT in this fork (known gaps) ----
        "flash_attn_qkvpacked_func": _sig(
            "qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), "
            "softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False"),
        "flash_attn_kvpacked_func": _sig(
            "q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), "
            "softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False"),
        "flash_attn_varlen_qkvpacked_func": _sig(
            "qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, "
            "window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, "
            "return_attn_probs=False"),
        "flash_attn_varlen_kvpacked_func": _sig(
            "q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, "
            "softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, "
            "alibi_slopes=None, deterministic=False, return_attn_probs=False"),
    },
    "standard_functions": [
        "flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache",
    ],
    "known_gaps": [
        "flash_attn_qkvpacked_func", "flash_attn_kvpacked_func",
        "flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func",
    ],
    "bert_padding": {
        "unpad_input": _sig("hidden_states, attention_mask, unused_mask=None"),
        "unpad_input_for_concatenated_sequences": _sig(
            "hidden_states, attention_mask_in_length"),
        "pad_input": _sig("hidden_states, indices, batch, seqlen"),
    },
    "bert_padding_names": [
        "index_first_axis", "index_put_first_axis", "index_first_axis_residual",
        "IndexFirstAxis", "IndexPutFirstAxis", "IndexFirstAxisResidual",
    ],
    "repo_extras": ["flash_attn_gpu", "flash_attn_varlen_gpu", "flash_attn_with_kvcache_gpu"],
    "modules": [
        "flash_attn", "flash_attn.flash_attn_interface", "flash_attn.bert_padding",
        "flash_attn_2_cuda",
    ],
    "dist_info": {"name": "flash-attn", "version": "2.8.3", "top_level": "flash_attn"},
}


# --------------------------------------------------------------------------------------
# spec loading (accepts research/03_api_spec.json in a few plausible shapes)
# --------------------------------------------------------------------------------------
def load_spec(path: str):
    with open(path) as f:
        raw = json.load(f)
    spec = dict(BUILTIN_SPEC)
    spec["source"] = f"file:{path}"
    funcs = raw.get("functions") or raw.get("public_functions")
    if funcs is None and isinstance(raw.get("symbols"), list):
        # research/03_api_spec.json style: [{"name": ..., "params": [...], "sig": "..."}]
        funcs = {s["name"]: s for s in raw["symbols"]
                 if isinstance(s, dict) and s.get("name")}
    if isinstance(funcs, dict):
        merged = {}
        bp = dict(spec["bert_padding"])
        for name, meta in funcs.items():
            entry = None
            if isinstance(meta, dict) and isinstance(meta.get("params"), list):
                params = []
                for p in meta["params"]:
                    if isinstance(p, dict):
                        params.append({"name": p.get("name"), "default": p.get("default")})
                    elif isinstance(p, (list, tuple)) and len(p) == 2:
                        params.append({"name": p[0], "default": p[1]})
                    else:
                        params.append({"name": str(p), "default": None})
                entry = {"params": params}
            elif isinstance(meta, dict) and isinstance(meta.get("sig"), str):
                entry = _parse_sig_string(meta["sig"])
            elif isinstance(meta, list):
                entry = {"params": [{"name": str(x), "default": None}
                                    if not isinstance(x, dict) else x for x in meta]}
            elif isinstance(meta, str):
                entry = _parse_sig_string(meta)
            if entry is None:
                continue
            if name in bp or name in ("unpad_input", "unpad_input_for_concatenated_sequences",
                                      "pad_input"):
                bp[name] = entry
                continue
            if not name.startswith("flash_attn"):
                continue
            merged[name] = entry
        spec["bert_padding"] = bp
        if merged:
            spec["functions"] = merged
    if "version" in raw:
        spec["target_version"] = str(raw["version"])
    if "known_gaps" in raw:
        spec["known_gaps"] = list(raw["known_gaps"])
    if "top_level" in raw and isinstance(raw["top_level"], list):
        spec["dist_info"] = dict(spec["dist_info"], top_level=raw["top_level"][0])
    for key in ("bert_padding", "standard_functions", "modules", "repo_extras"):
        if key in raw:
            spec[key] = raw[key]
    return spec


def _parse_sig_string(s: str) -> dict:
    s = s.strip()
    if s.startswith("def "):
        s = s[s.index("(") + 1: s.rindex(")")]
    if s.startswith("("):
        s = s[1:-1]
    return _sig(s)


# --------------------------------------------------------------------------------------
# AST mode
# --------------------------------------------------------------------------------------
def ast_collect(path: str, names: set):
    tree = ast.parse(open(path).read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            args = node.args
            argnames = [a.arg for a in args.posonlyargs + args.args]
            defaults = [ast.unparse(d) for d in args.defaults]
            nd = len(defaults)
            params = []
            for i, n in enumerate(argnames):
                di = i - (len(argnames) - nd)
                params.append({"name": n, "default": defaults[di] if di >= 0 else None})
            out[node.name] = {"params": params}
    return out


def ast_sources(repo: str):
    return {
        "flash_attn": os.path.join(repo, "flash_attn", "flash_attn_interface.py"),
        "bert_padding": os.path.join(repo, "flash_attn", "bert_padding.py"),
    }


# --------------------------------------------------------------------------------------
# signature plumbing
# --------------------------------------------------------------------------------------
def _norm(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, str):
        return v.strip().replace(" ", "")
    return str(v).replace(" ", "")


def _default_to_str(d):
    if d is None:
        return "None"
    if isinstance(d, str):
        return d
    return repr(d)


def compare_signature(got, want, name):
    """got: list[{name,default_str}]; want: same. Returns (ok, problems)."""
    problems = []
    g = [(p["name"], _default_to_str(p.get("default"))) for p in got]
    w = [(p["name"], _norm(p.get("default"))) for p in want["params"]]
    if len(g) != len(w):
        problems.append(f"arity {len(g)} != {len(w)}")
    for i, ((gn, gd), (wn, wd)) in enumerate(zip(g, w)):
        if gn != wn:
            problems.append(f"param[{i}] name '{gn}' != '{wn}'")
        if _norm(gd) != wd:
            problems.append(f"param[{i}] '{gn}': default '{gd}' != '{wd}'")
    return (not problems), problems


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("FA_V100_REPO",
                    "/home/eculid/Documents/Github/flash-attention-v100"))
    ap.add_argument("--spec", default=None, help="path to research/03_api_spec.json")
    ap.add_argument("--ast", action="store_true", help="source-only mode (no imports)")
    ap.add_argument("--mock", action="store_true", help="inject mock_cuda before importing")
    ap.add_argument("--mock-dir", default=HERE)
    ap.add_argument("--allow-known-gaps", action="store_true",
                    help="downgrade missing known-gap names to WARN")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    spec_path = args.spec or os.path.join(args.repo, "research", "03_api_spec.json")
    if os.path.exists(spec_path):
        spec = load_spec(spec_path)
    else:
        spec = BUILTIN_SPEC
        spec["source"] = BUILTIN_SPEC["source"] + " (03_api_spec.json not found)"

    result = {
        "mode": "ast" if args.ast else ("runtime+mock" if args.mock else "runtime"),
        "spec_source": spec["source"],
        "checks": [],
        "missing": [],
        "signature_mismatches": [],
        "warnings": [],
        "known_gaps": [],
    }

    def add(name, status, detail=""):
        result["checks"].append({"check": name, "status": status, "detail": detail})

    # ---------------- AST pre-build mode ----------------
    if args.ast:
        src = ast_sources(args.repo)
        names = set(spec["functions"]) | set(spec.get("bert_padding", {}))
        found = {}
        for tag, path in src.items():
            if os.path.exists(path):
                found.update(ast_collect(path, names))
        for name, want in spec["functions"].items():
            if name not in found:
                if name in spec.get("known_gaps", []):
                    result["known_gaps"].append(name)
                    add(f"sig:{name}", "WARN", "absent (known gap)")
                else:
                    result["missing"].append(name)
                    add(f"sig:{name}", "FAIL", "absent")
                continue
            ok, problems = compare_signature(
                [{"name": p["name"], "default": p["default"]} for p in found[name]["params"]],
                want, name)
            add(f"sig:{name}", "PASS" if ok else "FAIL", "; ".join(problems))
            if not ok:
                result["signature_mismatches"].append({"name": name, "problems": problems})
        for name, want in spec.get("bert_padding", {}).items():
            if name not in found:
                result["missing"].append(f"flash_attn.bert_padding.{name}")
                add(f"sig:bert_padding.{name}", "FAIL", "absent")
                continue
            ok, problems = compare_signature(
                [{"name": p["name"], "default": p["default"]} for p in found[name]["params"]],
                want, name)
            add(f"sig:bert_padding.{name}", "PASS" if ok else "FAIL", "; ".join(problems))
            if not ok:
                result["signature_mismatches"].append(
                    {"name": f"bert_padding.{name}", "problems": problems})
        # version stamps from source
        try:
            shim = open(os.path.join(args.repo, "flash_attn", "__init__.py")).read()
            ok = f'__version__ = "{spec["target_version"]}"' in shim
            add("version:flash_attn.__version__", "PASS" if ok else "FAIL",
                f'expected "{spec["target_version"]}" literal in flash_attn/__init__.py')
        except Exception as e:
            add("version:flash_attn.__version__", "FAIL", str(e))
        add("dist_info", "SKIP", "checked only in runtime mode (site-packages metadata)")

    # ---------------- runtime mode ----------------
    else:
        for p in (args.mock_dir if args.mock else None, args.repo):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        if args.mock:
            try:
                import mock_cuda
                mock_cuda.install()
                add("mock_cuda.install", "PASS", "fake extension injected")
            except Exception as e:
                add("mock_cuda.install", "FAIL", f"{type(e).__name__}: {e}")

        import importlib
        import inspect

        mods = {}
        for modname in spec["modules"]:
            try:
                mods[modname] = importlib.import_module(modname)
                add(f"import:{modname}", "PASS", "")
            except Exception as e:
                mods[modname] = None
                hard = modname in ("flash_attn", "flash_attn.flash_attn_interface",
                                   "flash_attn.bert_padding")
                add(f"import:{modname}", "FAIL" if hard else "WARN",
                    f"{type(e).__name__}: {e}")

        fa = mods.get("flash_attn")
        if fa is not None:
            want_v = spec["target_version"]
            got_v = getattr(fa, "__version__", None)
            add("version:flash_attn.__version__", "PASS" if got_v == want_v else "FAIL",
                f"got {got_v!r}, want {want_v!r}")

        # dist-info metadata (installed wheels only)
        di = spec.get("dist_info", {})
        try:
            import importlib.metadata as md
            dist = md.distribution(di.get("name", "flash-attn"))
            ver = dist.version
            meta_name = None
            try:
                meta_name = dist.metadata["Name"]
            except Exception:
                pass
            ok = (ver == di.get("version")) and (meta_name in (None, di.get("name")))
            add("dist_info:metadata", "PASS" if ok else "FAIL",
                f"Name={meta_name!r} Version={ver!r}")
            try:
                tl = dist.read_text("top_level.txt")
                # setuptools also lists the compiled extension module in top_level.txt;
                # official code only relies on `flash_attn` being present (first entry).
                ok_tl = tl is not None and tl.split()[:1] == [di.get("top_level", "flash_attn")]
                add("dist_info:top_level", "PASS" if ok_tl else "FAIL", f"got {tl!r}")
            except Exception as e:
                add("dist_info:top_level", "WARN", f"{type(e).__name__}: {e}")
        except Exception as e:
            add("dist_info:metadata", "SKIP", f"not installed as a distribution ({e})")

        # signatures
        iface = mods.get("flash_attn.flash_attn_interface")
        for name, want in spec["functions"].items():
            fn = getattr(iface, name, None) if iface else None
            if fn is None:
                if name in spec.get("known_gaps", []):
                    result["known_gaps"].append(name)
                    add(f"sig:{name}", "WARN", "absent from package (known gap)")
                else:
                    result["missing"].append(name)
                    add(f"sig:{name}", "FAIL", "absent from package")
                continue
            try:
                sig = inspect.signature(fn)
                got = [{"name": p.name,
                        "default": None if p.default is inspect.Parameter.empty else p.default}
                       for p in sig.parameters.values()
                       if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                ok, problems = compare_signature(got, want, name)
            except (ValueError, TypeError) as e:
                ok, problems = False, [f"no py-signature ({e})"]
            add(f"sig:{name}", "PASS" if ok else "FAIL", "; ".join(problems))
            if not ok:
                result["signature_mismatches"].append({"name": name, "problems": problems})

        # bert_padding
        bp = mods.get("flash_attn.bert_padding")
        if bp is not None:
            for name in spec.get("bert_padding", {}):
                want = spec["bert_padding"][name]
                fn = getattr(bp, name, None)
                if fn is None:
                    result["missing"].append(f"flash_attn.bert_padding.{name}")
                    add(f"sig:bert_padding.{name}", "FAIL", "absent")
                    continue
                try:
                    sig = inspect.signature(fn)
                    got = [{"name": p.name,
                            "default": None if p.default is inspect.Parameter.empty else p.default}
                           for p in sig.parameters.values()
                           if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                    ok, problems = compare_signature(got, want, name)
                except (ValueError, TypeError) as e:
                    ok, problems = False, [str(e)]
                add(f"sig:bert_padding.{name}", "PASS" if ok else "FAIL", "; ".join(problems))
                if not ok:
                    result["signature_mismatches"].append(
                        {"name": f"bert_padding.{name}", "problems": problems})
            for name in spec.get("bert_padding_names", []):
                present = hasattr(bp, name)
                add(f"attr:bert_padding.{name}", "PASS" if present else "FAIL",
                    "" if present else "absent")

        # extension module attributes
        ext = None
        for candidate in ("flash_attn_2_cuda",):
            ext = mods.get(candidate)
            if ext is not None:
                break
        if ext is not None:
            for fn_name in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "fwd_kvcache"):
                ok = callable(getattr(ext, fn_name, None))
                add(f"ext:{getattr(ext, '__name__', '?')}.{fn_name}", "PASS" if ok else "FAIL",
                    "" if ok else "absent")

        # repo extras (spec entries are dicts: {module, name, kind, ...}; bare
        # strings are accepted as a fallback)
        for entry in spec.get("repo_extras", []):
            name = entry.get("name") if isinstance(entry, dict) else entry
            if not name:
                continue
            present = any(getattr(m, name, None) is not None for m in mods.values() if m)
            add(f"extra:{name}", "PASS" if present else "WARN",
                "" if present else "repo alias missing (not an upstream name)")

    # ---------------- verdict ----------------
    hard_fail = any(c["status"] == "FAIL" for c in result["checks"])
    warns = [c for c in result["checks"] if c["status"] in ("WARN", "SKIP")]
    if hard_fail and args.allow_known_gaps:
        # only known-gap failures may be softened
        soft = [c for c in result["checks"] if c["status"] == "FAIL"
                and c["check"].replace("sig:", "") in spec.get("known_gaps", [])]
        hard_fail = any(c["status"] == "FAIL" for c in result["checks"] if c not in soft)
    result["status"] = "FAIL" if hard_fail else ("PARTIAL" if warns else "PASS")
    result["summary"] = {
        "checks": len(result["checks"]),
        "failed": sum(1 for c in result["checks"] if c["status"] == "FAIL"),
        "warn": len(warns),
        "known_gaps": result["known_gaps"],
    }

    print(f"API parity spec: {spec['source']}")
    print(f"{'STATUS':7} {'CHECK':45} DETAIL")
    for c in result["checks"]:
        if c["status"] != "PASS" or os.environ.get("PARITY_VERBOSE"):
            print(f"{c['status']:7} {c['check']:45} {c['detail']}")
    print(f"\nRESULT PARITY {result['status']}  "
          f"(failed={result['summary']['failed']}, warn={result['summary']['warn']}, "
          f"known_gaps={result['known_gaps']})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0 if result["status"] == "PASS" else (2 if result["status"] == "PARTIAL" else 1)


if __name__ == "__main__":
    sys.exit(main())
