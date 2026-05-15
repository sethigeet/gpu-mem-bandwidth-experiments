import argparse

import torch

from attention_bw.type import Case


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be B,H,CACHE_SEQ,D, e.g. 4,16,4096,64")
    batch, heads, cache_seq, dim = (int(p) for p in parts)
    return batch, heads, cache_seq, dim


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name}; expected one of {sorted(mapping)}") from exc


def make_tensors(case: Case, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = dtype_from_name(case.dtype)
    q_shape = (case.batch, case.heads, 1, case.dim)
    kv_shape = (case.batch, case.heads, case.cache_seq, case.dim)
    return (
        torch.randn(q_shape, device=device, dtype=dtype),
        torch.randn(kv_shape, device=device, dtype=dtype),
        torch.randn(kv_shape, device=device, dtype=dtype),
    )
