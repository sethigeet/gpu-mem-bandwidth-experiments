import argparse
import sys
from pathlib import Path

import torch

from attention_bw.kernels import KERNELS
from attention_bw.metrics import get_peak_mem_gb_s
from attention_bw.output import print_results, write_outputs
from attention_bw.runner import run_case
from attention_bw.type import Case, Result
from attention_bw.utils import parse_shape

DEFAULT_SHAPES = [(1, 16, 1024, 64), (1, 16, 2048, 64), (1, 16, 4096, 64), (1, 16, 8192, 64)]
DEFAULT_KERNELS = ["sdpa_math", "sdpa_mem_efficient", "sdpa_flash"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention kernel memory bandwidth on CUDA GPUs.")
    parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS, choices=KERNELS)
    parser.add_argument("--shape", type=parse_shape, action="append", default=[])
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/attention_bw.csv"))
    return parser


def collect_results(args: argparse.Namespace, peak_gb_s: float | None) -> tuple[list[Result], list[str]]:
    shapes = args.shape or DEFAULT_SHAPES
    cases = [Case(b, h, s, d, args.dtype, args.causal) for b, h, s, d in shapes]

    results: list[Result] = []
    failures: list[str] = []
    for case in cases:
        for kernel in args.kernels:
            try:
                results.append(run_case(case, kernel, args.warmup, args.iters, peak_gb_s))
            except Exception as exc:
                shape = f"{case.batch},{case.heads},{case.seq},{case.dim}"
                failures.append(f"{kernel} B,H,S,D={shape}: {exc}")
    return results, failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on the remote GPU host.")

    torch.backends.cuda.matmul.allow_tf32 = True
    results, failures = collect_results(args, get_peak_mem_gb_s())

    if results:
        print_results(results)
        write_outputs(results, args.out)
        print(f"\nwrote {args.out}")
    if failures:
        print("\nfailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    return 0 if results else 1
