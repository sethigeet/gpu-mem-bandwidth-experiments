import math

import torch

from attention_bw.kernels import AttentionKernel, get_kernel
from attention_bw.metrics import estimate_attention_bytes, estimate_attention_flops
from attention_bw.type import Case, Result
from attention_bw.utils import make_tensors


def time_kernel(
    fn: AttentionKernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        fn(q, k, v, causal)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(q, k, v, causal)
        end.record()
        torch.cuda.synchronize()
        if out.numel() == 0:
            raise RuntimeError("empty output")
        times.append(start.elapsed_time(end))
    return times


def run_case(case: Case, kernel_name: str, warmup: int, iters: int, peak_gb_s: float | None) -> Result:
    device = torch.device("cuda")
    q, k, v = make_tensors(case, device)
    times = sorted(time_kernel(get_kernel(kernel_name), q, k, v, case.causal, warmup, iters))
    median_ms = times[len(times) // 2]
    estimated_bytes = estimate_attention_bytes(case, materializes_scores=kernel_name == "sdpa_math")
    effective_gb_s = estimated_bytes / (median_ms / 1000.0) / 1e9
    flops = estimate_attention_flops(case)

    return Result(
        kernel=kernel_name,
        batch=case.batch,
        heads=case.heads,
        seq=case.seq,
        dim=case.dim,
        dtype=case.dtype,
        causal=case.causal,
        median_ms=median_ms,
        p20_ms=times[max(0, math.floor(0.2 * (len(times) - 1)))],
        p80_ms=times[min(len(times) - 1, math.ceil(0.8 * (len(times) - 1)))],
        effective_gb_s=effective_gb_s,
        utilization_pct_of_peak=None if peak_gb_s is None else 100.0 * effective_gb_s / peak_gb_s,
        estimated_bytes=estimated_bytes,
        tflops=flops / (median_ms / 1000.0) / 1e12,
    )
