"""CPU-only checks that the public API matches official flash-attn 2.8.3.

Importing :mod:`flash_attn` requires the compiled ``flash_attn_2_cuda``
extension, so this module skips cleanly on machines where it has not been
built (or where it cannot be loaded without a Volta GPU).

The signature strings below are pinned from the official 2.8.3 source,
``flash_attn/flash_attn_interface.py``.
"""

import importlib
import inspect
import os
import sys
import unittest

# Prefer the repository package over any stale installed flash-attn copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The package imports flash_attn_2_cuda (via _v100_compat); use the CPU stub when
# the compiled extension is not built.  Must happen before importing flash_attn.
try:
    import _stub_cuda_backend as _sb

    _sb.install_stub()
except Exception:  # noqa: BLE001
    # torch itself may be unavailable; the test class then skips cleanly.
    pass


OFFICIAL_VERSION = "2.8.3"

OFFICIAL_PUBLIC_NAMES = (
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
)

# Exact ``str(inspect.signature(fn))`` values from official flash-attn 2.8.3.
OFFICIAL_SIGNATURES = {
    "flash_attn_func": (
        "(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, "
        "window_size=(-1, -1), softcap=0.0, alibi_slopes=None, "
        "deterministic=False, return_attn_probs=False)"
    ),
    "flash_attn_varlen_func": (
        "(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, "
        "dropout_p=0.0, softmax_scale=None, causal=False, "
        "window_size=(-1, -1), softcap=0.0, alibi_slopes=None, "
        "deterministic=False, return_attn_probs=False, block_table=None)"
    ),
    "flash_attn_qkvpacked_func": (
        "(qkv, dropout_p=0.0, softmax_scale=None, causal=False, "
        "window_size=(-1, -1), softcap=0.0, alibi_slopes=None, "
        "deterministic=False, return_attn_probs=False)"
    ),
    "flash_attn_kvpacked_func": (
        "(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, "
        "window_size=(-1, -1), softcap=0.0, alibi_slopes=None, "
        "deterministic=False, return_attn_probs=False)"
    ),
    "flash_attn_varlen_qkvpacked_func": (
        "(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, "
        "causal=False, window_size=(-1, -1), softcap=0.0, "
        "alibi_slopes=None, deterministic=False, return_attn_probs=False)"
    ),
    "flash_attn_varlen_kvpacked_func": (
        "(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, "
        "dropout_p=0.0, softmax_scale=None, causal=False, "
        "window_size=(-1, -1), softcap=0.0, alibi_slopes=None, "
        "deterministic=False, return_attn_probs=False)"
    ),
    "flash_attn_with_kvcache": (
        "(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, "
        "rotary_sin=None, "
        "cache_seqlens: Union[int, torch.Tensor, NoneType] = None, "
        "cache_batch_idx: Optional[torch.Tensor] = None, "
        "cache_leftpad: Optional[torch.Tensor] = None, "
        "block_table: Optional[torch.Tensor] = None, "
        "softmax_scale=None, causal=False, window_size=(-1, -1), "
        "softcap=0.0, rotary_interleaved=True, alibi_slopes=None, "
        "num_splits=0, return_softmax_lse=False)"
    ),
}


def _require_flash_attn():
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise unittest.SkipTest(
            "flash_attn (or torch/the extension stub) is not importable in this "
            f"environment ({type(exc).__name__}: {exc}); skipping official-API checks"
        ) from exc


class TestOfficialAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_flash_attn()
        import flash_attn

        cls.flash_attn = flash_attn
        cls.interface = importlib.import_module("flash_attn.flash_attn_interface")

    def test_version(self):
        self.assertEqual(self.flash_attn.__version__, OFFICIAL_VERSION)

    def test_public_names_exist(self):
        for name in OFFICIAL_PUBLIC_NAMES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(self.flash_attn, name))
                self.assertTrue(callable(getattr(self.flash_attn, name)))
                # The package re-exports the interface function objects.
                self.assertIs(
                    getattr(self.flash_attn, name),
                    getattr(self.interface, name),
                )

    def test_official_signatures(self):
        for name, expected in OFFICIAL_SIGNATURES.items():
            with self.subTest(name=name):
                fn = getattr(self.flash_attn, name)
                self.assertEqual(str(inspect.signature(fn)), expected)


def main():
    """Run the suite and emit the CPU runner's summary line."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOfficialAPI)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - skipped
    print(f"\n== test_official_api: {passed}/{total} passed, {failed} failed, "
          f"{skipped} skipped ==")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
