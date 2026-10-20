__version__ = "2.8.3"
from . import _v100_compat  # noqa: F401  (private V100 extension adapter)

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

# Keep the official public API; route backward through the private V100 adapter.
_v100_compat.patch_interface()
