import gc

import torch

from attention_bw.kernels import get_kernel
from attention_bw.type import Case
from attention_bw.utils import make_tensors


def run_case(case: Case, kernel_name: str, warmup: int, iters: int) -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda")
    q, k, v = make_tensors(case, device)
    fn = get_kernel(kernel_name)

    torch.cuda.nvtx.range_push(f"attention_bw:{kernel_name}:case")
    try:
        for _ in range(warmup):
            torch.cuda.nvtx.range_push(f"attention_bw:{kernel_name}:warmup")
            try:
                fn(q, k, v, False)
            finally:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

        out = None
        for _ in range(iters):
            torch.cuda.nvtx.range_push(f"attention_bw:{kernel_name}:iter")
            try:
                out = fn(q, k, v, False)
            finally:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            if out.numel() == 0:
                raise RuntimeError("empty output")
    finally:
        torch.cuda.nvtx.range_pop()

    del q, k, v
    if out is not None:
        del out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
