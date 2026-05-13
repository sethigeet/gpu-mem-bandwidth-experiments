import argparse
import sys
from pathlib import Path

import torch

from attention_bw.kernels import KERNELS
from attention_bw.runner import run_case
from attention_bw.type import Case
from attention_bw.utils import parse_shape
from attention_bw.visualize import load_results, visualize, visualize_nsys

# DEFAULT_SHAPES = [(1, 16, 1024, 64), (1, 16, 2048, 64), (1, 16, 4096, 64), (1, 16, 8192, 64)]
DEFAULT_SHAPES = [(2, 64, 4096, 128)]
DEFAULT_KERNELS = ["sdpa_math", "sdpa_mem_efficient", "sdpa_flash"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention kernel memory bandwidth on CUDA GPUs.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run kernels for NCU profiling")
    run_parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS, choices=[*KERNELS, "all"])
    run_parser.add_argument("--shape", type=parse_shape, action="append", default=[])
    run_parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    run_parser.add_argument("--causal", action="store_true")
    run_parser.add_argument("--warmup", type=int, default=10)
    run_parser.add_argument("--iters", type=int, default=50)

    viz_parser = subparsers.add_parser("visualize", help="Visualize results from a previous run")
    viz_parser.add_argument("input", type=Path, help="Path to NCU CSV or nsys sqlite file")
    viz_parser.add_argument("--output", "-o", type=Path, help="Save plot to file instead of displaying")

    return parser


def run_benchmarks(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on the remote GPU host.")

    if "all" in args.kernels:
        args.kernels = list(KERNELS)

    shapes = args.shape or DEFAULT_SHAPES
    cases = [Case(b, h, s, d, args.dtype, args.causal) for b, h, s, d in shapes]

    failures: list[str] = []
    for case in cases:
        for kernel in args.kernels:
            try:
                print(f"Running {kernel} with shape ({case.batch}, {case.heads}, {case.seq}, {case.dim})", flush=True)
                run_case(case, kernel, args.warmup, args.iters)
                print("  Done.", flush=True)
            except Exception as exc:
                shape = f"{case.batch},{case.heads},{case.seq},{case.dim}"
                failures.append(f"{kernel} B,H,S,D={shape}: {exc}")
                print("  Failed.", flush=True)

    if failures:
        print("\nfailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def run_visualize(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1
    if args.input.suffix == ".sqlite":
        visualize_nsys(args.input, args.output)
    else:
        df = load_results(args.input)
        visualize(df, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_benchmarks(args)
    elif args.command == "visualize":
        return run_visualize(args)
    else:
        parser.print_help()
        return 0
