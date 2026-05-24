import argparse
import sys
from pathlib import Path

import torch

from llm_bw.models import MODEL_REGISTRY, load_model
from llm_bw.runner import run_decode_benchmark
from llm_bw.visualize import visualize_ncu, visualize_nsys

DEFAULT_MODELS = ["phi-3-mini"]
"""
"flash_attention_3"	improves FlashAttention-2 by also overlapping operations and fusing forward and backward passes more tightly
"flash_attention_2"	tiles computations into smaller blocks and uses fast on-chip memory
"flex_attention"	framework for specifying custom attention patterns (sparse, block-local, sliding window) without writing low-level kernels by hand
"sdpa"	built-in PyTorch implementation of scaled dot product attention
“paged|flash_attention_3”	Paged version of FlashAttention-3
“paged|flash_attention_2”	Paged version of FlashAttention-2
“paged|sdpa”	Paged version of SDPA
“paged|eager”	Paged version of eager mode
"""
ATTENTION_IMPLS = [
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flex_attention",
    "paged|flash_attention_3",
    "paged|flash_attention_2",
    "paged|sdpa",
    "paged|eager",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LLM decode-phase memory bandwidth on CUDA GPUs.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run LLM decode benchmark")
    run_parser.add_argument(
        "--model",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model(s) to benchmark",
    )
    run_parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )
    run_parser.add_argument(
        "--attention",
        choices=ATTENTION_IMPLS,
        default="sdpa",
        help="Attention implementation",
    )
    run_parser.add_argument("--prompt-length", type=int, default=512)
    run_parser.add_argument("--decode-tokens", type=int, default=50)
    run_parser.add_argument("--warmup-tokens", type=int, default=5)
    run_parser.add_argument("--batch-size", type=int, default=1)

    viz_parser = subparsers.add_parser("visualize", help="Visualize results")
    viz_parser.add_argument("input", type=Path, help="Path to NCU CSV or NSYS sqlite")
    viz_parser.add_argument("--output", "-o", type=Path, help="Save plot to file")
    viz_parser.add_argument("--model", type=str, help="Model name for plot title")
    viz_parser.add_argument("--dtype", type=str, help="Data type for plot title")
    viz_parser.add_argument("--attention", type=str, help="Attention impl for plot title")
    viz_parser.add_argument("--prompt-length", type=int, help="Prompt length for plot title")
    viz_parser.add_argument("--batch-size", type=int, help="Batch size for plot title")

    return parser


def run_benchmarks(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on the remote GPU host.")

    failures: list[str] = []
    for model_name in args.model:
        try:
            print(f"Loading {model_name}...", flush=True)
            model, tokenizer = load_model(model_name, args.dtype, args.attention)
            print(
                f"Running decode benchmark: prompt_length={args.prompt_length}, "
                f"decode_tokens={args.decode_tokens}, batch_size={args.batch_size}",
                flush=True,
            )
            run_decode_benchmark(
                model,
                tokenizer,
                model_name,
                args.prompt_length,
                args.decode_tokens,
                args.warmup_tokens,
                args.batch_size,
            )
            print("  Done.", flush=True)
            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")
            print(f"  Failed: {exc}", flush=True)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def run_visualize(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    config = {
        "model": args.model,
        "dtype": args.dtype,
        "attention": args.attention,
        "prompt_length": args.prompt_length,
        "batch_size": args.batch_size,
    }
    config = {k: v for k, v in config.items() if v is not None}

    if args.input.suffix == ".sqlite":
        visualize_nsys(args.input, args.output, config=config)
    else:
        visualize_ncu(args.input, args.output, config=config)
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
