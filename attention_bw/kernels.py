from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

# (q, k, v, causal) -> out
AttentionKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]

KERNELS = ("sdpa_math", "sdpa_mem_efficient", "sdpa_flash", "flash_attn", "xformers")


def _sdpa_kernel(name: str) -> AttentionKernel:
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

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        backend = backend_map[name]
        if kernel_ctx is None or backend is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        with kernel_ctx(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return run


def _flash_attn_kernel() -> AttentionKernel:
    from flash_attn import flash_attn_func  # type: ignore

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal).transpose(1, 2)

    return run


def _xformers_kernel() -> AttentionKernel:
    from xformers.ops import memory_efficient_attention  # type: ignore

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        bias = None
        if causal:
            from xformers.ops import LowerTriangularMask  # type: ignore

            bias = LowerTriangularMask()
        return memory_efficient_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_bias=bias
        ).transpose(1, 2)

    return run


def get_kernel(name: str) -> AttentionKernel:
    match name:
        case "sdpa_math" | "sdpa_mem_efficient" | "sdpa_flash":
            return _sdpa_kernel(name)
        case "flash_attn":
            return _flash_attn_kernel()
        case "xformers":
            return _xformers_kernel()
        case _:
            raise ValueError(f"unknown kernel {name}")
