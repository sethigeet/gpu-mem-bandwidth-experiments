import os
import subprocess

import torch

from attention_bw.type import Case
from attention_bw.utils import dtype_from_name


def estimate_attention_bytes(case: Case, materializes_scores: bool) -> int:
    dtype_size = torch.empty((), dtype=dtype_from_name(case.dtype)).element_size()
    qkv_o = 4 * case.batch * case.heads * case.seq * case.dim * dtype_size
    scores = 0
    if materializes_scores:
        scores = 2 * case.batch * case.heads * case.seq * case.seq * dtype_size
    return qkv_o + scores


def estimate_attention_flops(case: Case) -> float:
    factor = 0.5 if case.causal else 1.0
    return 4.0 * factor * case.batch * case.heads * case.seq * case.seq * case.dim


def get_peak_mem_gb_s() -> float | None:
    env = os.getenv("ATTN_BW_PEAK_GB_S")
    if env:
        return float(env)
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.max_clock,memory.bus_width", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
        )
        mem_clock_mhz, bus_width_bits = [float(x.strip()) for x in out.split(",")]
        return mem_clock_mhz * 2.0 * (bus_width_bits / 8.0) / 1000.0
    except Exception:
        return None
