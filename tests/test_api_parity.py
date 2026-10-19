#!/usr/bin/env python
# ======================================================================================
# * API parity test: flash_attn (this repo, Volta) vs official FlashAttention-2 2.8.x.
# *
# * Runs on CPU only. The expected signatures below are pinned from
# * Dao-AILab/flash-attention@main (flash-attn 2.8.4), which matches the version range
# * this repo emulates (2.8.3).
# * Set FLASH_ATTENTION_OFFICIAL_SRC=/path/to/flash_attn_interface.py to additionally
# * compare directly against an official checkout (AST-parsed).
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import ast
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stub_cuda_backend as stub  # noqa: E402

stub.install_stub()

import flash_attn  # noqa: E402
import flash_attn.flash_attn_interface as fai  # noqa: E402

# --------------------------------------------------------------------------------------
# Pinned official signatures (param name, default as source text) for flash-attn 2.8.x
# --------------------------------------------------------------------------------------
OFFICIAL_SIGNATURES = {
    "flash_attn_func": [
        ("q", None), ("k", None), ("v", None), ("dropout_p", "0.0"), ("softmax_scale", "None"),
        ("causal", "False"), ("window_size", "(-1, -1)"), ("softcap", "0.0"),
        ("alibi_slopes", "None"), ("deterministic", "False"), ("return_attn_probs", "False"),
    ],
    "flash_attn_varlen_func": [
        ("q", None), ("k", None), ("v", None), ("cu_seqlens_q", None), ("cu_seqlens_k", None),
        ("max_seqlen_q", None), ("max_seqlen_k", None), ("dropout_p", "0.0"),
        ("softmax_scale", "None"), ("causal", "False"), ("window_size", "(-1, -1)"),
        ("softcap", "0.0"), ("alibi_slopes", "None"), ("deterministic", "False"),
        ("return_attn_probs", "False"), ("block_table", "None"),
    ],
    "flash_attn_qkvpacked_func": [
        ("qkv", None), ("dropout_p", "0.0"), ("softmax_scale", "None"), ("causal", "False"),
        ("window_size", "(-1, -1)"), ("softcap", "0.0"), ("alibi_slopes", "None"),
        ("deterministic", "False"), ("return_attn_probs", "False"),
    ],
    "flash_attn_kvpacked_func": [
        ("q", None), ("kv", None), ("dropout_p", "0.0"), ("softmax_scale", "None"),
        ("causal", "False"), ("window_size", "(-1, -1)"), ("softcap", "0.0"),
        ("alibi_slopes", "None"), ("deterministic", "False"), ("return_attn_probs", "False"),
    ],
    "flash_attn_varlen_qkvpacked_func": [
        ("qkv", None), ("cu_seqlens", None), ("max_seqlen", None), ("dropout_p", "0.0"),
        ("softmax_scale", "None"), ("causal", "False"), ("window_size", "(-1, -1)"),
        ("softcap", "0.0"), ("alibi_slopes", "None"), ("deterministic", "False"),
        ("return_attn_probs", "False"),
    ],
    "flash_attn_varlen_kvpacked_func": [
        ("q", None), ("kv", None), ("cu_seqlens_q", None), ("cu_seqlens_k", None),
        ("max_seqlen_q", None), ("max_seqlen_k", None), ("dropout_p", "0.0"),
        ("softmax_scale", "None"), ("causal", "False"), ("window_size", "(-1, -1)"),
        ("softcap", "0.0"), ("alibi_slopes", "None"), ("deterministic", "False"),
        ("return_attn_probs", "False"),
    ],
    "flash_attn_with_kvcache": [
        ("q", None), ("k_cache", None), ("v_cache", None), ("k", "None"), ("v", "None"),
        ("rotary_cos", "None"), ("rotary_sin", "None"), ("cache_seqlens", "None"),
        ("cache_batch_idx", "None"), ("cache_leftpad", "None"), ("block_table", "None"),
        ("softmax_scale", "None"), ("causal", "False"), ("window_size", "(-1, -1)"),
        ("softcap", "0.0"), ("rotary_interleaved", "True"), ("alibi_slopes", "None"),
        ("num_splits", "0"), ("return_softmax_lse", "False"),
    ],
}


def _norm(value):
    if value is None:
        return None
    s = value.replace(" ", "")
    # py inspect renders floats like 0.0; tuple defaults via str()
    return s


def our_signature(fn):
    params = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        default = None
        if p.default is not inspect.Parameter.empty:
            default = str(p.default)
        params.append((p.name, default))
    return params


def parse_official(path):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
            params = []
            for a, d in zip(node.args.args, defaults):
                params.append((a.arg, ast.unparse(d) if d is not None else None))
            out[node.name] = params
    return out


def compare(name, expected, actual, label):
    exp_names = [p[0] for p in expected]
    got_names = [p[0] for p in actual]
    if exp_names != got_names:
        stub.RESULT.fail(f"{label}:{name}", f"params differ\n    official={exp_names}\n    ours    ={got_names}")
        return
    bad = []
    for (en, ed), (gn, gd) in zip(expected, actual):
        if _norm(ed) != _norm(gd):
            bad.append(f"{en}: official={ed} ours={gd}")
    if bad:
        stub.RESULT.fail(f"{label}:{name}", "; ".join(bad))
    else:
        stub.RESULT.ok(f"{label}:{name}")


print("== API parity: signature checks ==")
for name, expected in OFFICIAL_SIGNATURES.items():
    fn = getattr(flash_attn, name, None)
    if fn is None:
        stub.RESULT.fail(f"flash_attn.{name}", "missing")
        continue
    compare(name, expected, our_signature(fn), "flash_attn")
    # the shim module must expose the same function objects
    shim_fn = getattr(fai, name, None)
    if shim_fn is None:
        stub.RESULT.fail(f"flash_attn.flash_attn_interface.{name}", "missing")
    else:
        stub.RESULT.ok(f"flash_attn.flash_attn_interface.{name} present")

print("\n== API parity: exports / version ==")
official_exports = {
    "flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func", "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func", "flash_attn_with_kvcache",
}
missing = official_exports - set(dir(flash_attn))
if missing:
    stub.RESULT.fail("flash_attn exports", f"missing {sorted(missing)}")
else:
    stub.RESULT.ok("flash_attn exposes all 7 official FA2 entry points")

version = getattr(flash_attn, "__version__", None)
if version and version.startswith("2."):
    stub.RESULT.ok(f"__version__ = {version}")
else:
    stub.RESULT.fail("__version__", f"unexpected {version!r}")

print("\n== API parity: live comparison against official source (optional) ==")
official_path = os.environ.get("FLASH_ATTENTION_OFFICIAL_SRC",
                               "/tmp/opencode/reference/flash-attention/flash_attn/flash_attn_interface.py")
if os.path.exists(official_path):
    official = parse_official(official_path)
    for name in OFFICIAL_SIGNATURES:
        if name in official:
            compare(name, official[name], our_signature(getattr(flash_attn, name)), "official-src")
        else:
            stub.RESULT.skip(f"official-src:{name}", "not in source")
else:
    stub.RESULT.skip("official-source comparison", f"{official_path} not found")

stub.RESULT.finish("test_api_parity")
