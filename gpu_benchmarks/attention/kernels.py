from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

# (q, k, v, causal) -> out
AttentionKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]

KERNELS = ("sdpa_math", "sdpa_mem_efficient", "sdpa_flash")


def get_kernel(name: str) -> AttentionKernel:
    sdp_backend: Any | None = None
    kernel_ctx: Any | None = None
    try:
        from torch.nn import attention

        sdp_backend = attention.SDPBackend
        kernel_ctx = attention.sdpa_kernel
    except Exception:
        pass

    backend_map = {
        "sdpa_math": getattr(sdp_backend, "MATH", None) if sdp_backend else None,
        "sdpa_mem_efficient": getattr(sdp_backend, "EFFICIENT_ATTENTION", None) if sdp_backend else None,
        "sdpa_flash": getattr(sdp_backend, "FLASH_ATTENTION", None) if sdp_backend else None,
    }

    if name not in backend_map:
        raise ValueError(f"unknown kernel {name}")

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        backend = backend_map[name]
        if kernel_ctx is None or backend is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        with kernel_ctx(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return run
