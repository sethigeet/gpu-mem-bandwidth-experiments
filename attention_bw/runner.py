import torch

from attention_bw.kernels import get_kernel
from attention_bw.type import Case
from attention_bw.utils import make_tensors


def run_case(case: Case, kernel_name: str, warmup: int, iters: int) -> None:
    device = torch.device("cuda")
    q, k, v = make_tensors(case, device)
    fn = get_kernel(kernel_name)

    for _ in range(warmup):
        fn(q, k, v, case.causal)
    torch.cuda.synchronize()

    for _ in range(iters):
        out = fn(q, k, v, case.causal)
        torch.cuda.synchronize()
        if out.numel() == 0:
            raise RuntimeError("empty output")
