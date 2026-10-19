#!/usr/bin/env python
# ======================================================================================
# * bert_padding parity test: flash_attn.bert_padding vs official flash-attn 2.8.x.
# * Pure CPU; compares outputs AND gradients numerically against the official module
# * (loaded from a checkout path, see FLASH_ATTENTION_OFFICIAL_BERT_PADDING).
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import _stub_cuda_backend as sb  # noqa: E402

sb.install_stub(force=True)  # flash_attn.bert_padding is imported via the flash_attn package
R = sb.RESULT

import flash_attn.bert_padding as ours  # noqa: E402

OFFICIAL_PATH = os.environ.get(
    "FLASH_ATTENTION_OFFICIAL_BERT_PADDING",
    "/tmp/opencode/reference/flash-attention/flash_attn/bert_padding.py",
)


def load_official(path):
    spec = importlib.util.spec_from_file_location("official_bert_padding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


official = None
if os.path.exists(OFFICIAL_PATH):
    try:
        official = load_official(OFFICIAL_PATH)
    except Exception as e:  # e.g. einops missing
        R.skip("load official bert_padding", str(e))
else:
    R.skip("official bert_padding source", f"not found at {OFFICIAL_PATH}")


def check(name, cond, detail=""):
    (R.ok if cond else R.fail)(name, detail)


torch.manual_seed(0)
B, S, D = 3, 7, 5
hidden = torch.randn(B, S, D)
mask_bool = torch.tensor(
    [[1, 1, 1, 0, 0, 1, 0],
     [1, 0, 1, 1, 0, 0, 1],
     [0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
mask_int = mask_bool.to(torch.int32)


def test_unpad_pad_roundtrip():
    n = int(mask_bool.sum())
    hs, indices, cu, maxlen, seqused = ours.unpad_input(hidden, mask_bool)
    check("unpad: hidden shape (nnz, D)", hs.shape == (n, D))
    check("unpad: indices shape", indices.shape == (n,))
    check("unpad: cu_seqlens int32 + values", cu.dtype == torch.int32
          and cu.tolist() == [0, 4, 8, 13])  # mask rows sum to 4 + 4 + 5
    check("unpad: max_seqlen", maxlen == int(mask_bool.sum(-1).max()))
    check("unpad: seqused == per-row sums", seqused.tolist() == mask_bool.sum(-1).tolist())
    check("unpad: gathered values", torch.equal(hs, hidden[mask_bool]))

    padded = ours.pad_input(hs, indices, B, S)
    expected = hidden.masked_fill(~mask_bool.unsqueeze(-1), 0)
    check("pad_input: roundtrip with zero fill", torch.equal(padded, expected))

    hs_i, idx_i, cu_i, ml_i, su_i = ours.unpad_input(hidden, mask_int)
    check("unpad: int mask == bool mask", torch.equal(hs_i, hs) and torch.equal(cu_i, cu)
          and su_i.tolist() == seqused.tolist())


def test_unused_mask():
    attn = torch.tensor(
        [[1, 1, 0, 0, 1, 0, 0],
         [1, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 1]], dtype=torch.bool)
    unused = torch.tensor(
        [[0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0]], dtype=torch.bool)
    out = ours.unpad_input(hidden, attn, unused)
    all_masks = attn | unused
    check("unused_mask: gathers attn|unused", torch.equal(out[0], hidden[all_masks]))
    check("unused_mask: cu from all masks",
          out[2].tolist() == [0] + torch.cumsum(all_masks.sum(-1), 0).tolist())
    check("unused_mask: seqused counts only used (official semantics)",
          out[4].tolist() == attn.sum(-1).tolist())
    if official is not None:
        o = official.unpad_input(hidden, attn, unused)
        check("unused_mask: exact official parity",
              torch.equal(o[0], out[0]) and torch.equal(o[1], out[1])
              and torch.equal(o[2], out[2]) and o[3] == out[3]
              and torch.equal(o[4], out[4]))


def test_concatenated_sequences():
    hidden6 = torch.randn(3, 6, 4)
    aml = torch.tensor(
        [[2, 3, 0, 0, 0, 0],
         [3, 2, 0, 0, 0, 0],
         [6, 0, 0, 0, 0, 0]], dtype=torch.int32)
    out = ours.unpad_input_for_concatenated_sequences(hidden6, aml)
    total = int(aml.sum())
    check("concat: total tokens", out[0].shape == (total, 4))
    # unpad_input_for_concatenated_sequences returns one entry per *concatenated
    # subsequence* (upstream semantics), not per padded row.
    seqlens = out[2].diff()
    check("concat: per-sample lengths", seqlens.tolist() == aml[aml > 0].tolist())
    check("concat: max_seqlen", out[3] == int(aml.sum(-1).max()))
    if official is not None:
        o = official.unpad_input_for_concatenated_sequences(hidden6, aml)
        check("concat: exact official parity",
              torch.equal(o[0], out[0]) and torch.equal(o[1], out[1])
              and torch.equal(o[2], out[2]) and o[3] == out[3])


def test_gradients():
    if official is None:
        R.skip("gradient parity", "official module not loaded")
        return
    h1 = torch.randn(B, S, D, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    w = torch.randn(int(mask_bool.sum()), D)

    hs1, idx1, _, _, _ = ours.unpad_input(h1, mask_bool)
    (hs1 * w).sum().backward()
    hs2, idx2, _, _, _ = official.unpad_input(h2, mask_bool)
    (hs2 * w).sum().backward()
    check("grad: unpad_input backward parity",
          torch.allclose(h1.grad, h2.grad, atol=0, rtol=0) and torch.equal(idx1, idx2))

    hs, indices, _, _, _ = ours.unpad_input(hidden, mask_bool)
    w2 = torch.randn(B, S, D)
    a = hs.detach().clone().requires_grad_(True)
    ours.pad_input(a, indices, B, S).mul(w2).sum().backward()
    b = hs.detach().clone().requires_grad_(True)
    official.pad_input(b, indices, B, S).mul(w2).sum().backward()
    check("grad: pad_input backward parity",
          torch.allclose(a.grad, b.grad, atol=0, rtol=0))

    flat = hidden.reshape(B * S, D)
    idx = torch.tensor([0, 3, 5, 9, 12, 14], dtype=torch.long)
    x1 = flat.clone().requires_grad_(True)
    ours.index_first_axis(x1, idx).sum().backward()
    x2 = flat.clone().requires_grad_(True)
    official.index_first_axis(x2, idx).sum().backward()
    check("grad: index_first_axis backward parity", torch.allclose(x1.grad, x2.grad))

    v1 = torch.randn(len(idx), D, requires_grad=True)
    v2 = v1.detach().clone().requires_grad_(True)
    ours.index_put_first_axis(v1, idx, B * S).mul(w2.reshape(-1, D)).sum().backward()
    official.index_put_first_axis(v2, idx, B * S).mul(w2.reshape(-1, D)).sum().backward()
    check("grad: index_put_first_axis backward parity", torch.allclose(v1.grad, v2.grad))


def test_forward_value_parity():
    if official is None:
        R.skip("forward parity", "official module not loaded")
        return
    o = official.unpad_input(hidden, mask_bool)
    m = ours.unpad_input(hidden, mask_bool)
    check("unpad_input: exact official parity",
          torch.equal(o[0], m[0]) and torch.equal(o[1], m[1])
          and torch.equal(o[2], m[2]) and o[3] == m[3] and torch.equal(o[4], m[4]))

    p_o = official.pad_input(o[0], o[1], B, S)
    p_m = ours.pad_input(m[0], m[1], B, S)
    check("pad_input: exact official parity", torch.equal(p_o, p_m))


if __name__ == "__main__":
    print("== bert_padding parity (official source:", OFFICIAL_PATH, ") ==")
    test_unpad_pad_roundtrip()
    test_unused_mask()
    test_concatenated_sequences()
    test_forward_value_parity()
    test_gradients()
    R.finish("test_bert_padding")
