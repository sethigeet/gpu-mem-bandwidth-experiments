from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

from component_bw.config import LAYOUTS, MODEL_PRESETS, STAGES, StageName, SyntheticConfig, with_model_preset
from component_bw.runner import result_to_row, run_stage, write_config, write_rows


def _batch_size(text: str) -> int | None:
    if text == "auto":
        return None
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive or 'auto'")
    return value


def _stage_list(values: list[str]) -> list[StageName]:
    if "all" in values:
        return list(STAGES)
    return cast(list[StageName], [stage for stage in values if stage in STAGES])


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=list(MODEL_PRESETS), default="phi-3-mini")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--layers", dest="num_layers", type=int)
    parser.add_argument("--heads", dest="num_attention_heads", type=int)
    parser.add_argument("--kv-heads", dest="num_key_value_heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--prefix-len", type=int, default=10_000)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=5)
    parser.add_argument("--batch-size", type=_batch_size, default=None, help="Batch size, or 'auto' (default)")
    parser.add_argument("--max-auto-batch", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--layout", choices=LAYOUTS, default="dense_duplicate")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--reserve-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small shape for remote validation before the 10k-prefix run.",
    )


def _config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    config = with_model_preset(SyntheticConfig(), args.model)
    config = replace(
        config,
        prefix_len=args.prefix_len,
        decode_tokens=args.decode_tokens,
        warmup_tokens=args.warmup_tokens,
        batch_size=args.batch_size,
        max_auto_batch=args.max_auto_batch,
        dtype=args.dtype,
        layout=args.layout,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        reserve_gb=args.reserve_gb,
        seed=args.seed,
    )

    overrides = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "intermediate_size": args.intermediate_size,
        "vocab_size": args.vocab_size,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if overrides:
        config = replace(config, **overrides)

    if config.hidden_size != config.num_attention_heads * config.head_dim:
        raise SystemExit("--hidden-size must equal --heads * --head-dim for this synthetic harness")

    return config


def _apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.prefix_len = min(args.prefix_len, 512)
    args.decode_tokens = min(args.decode_tokens, 4)
    args.warmup_tokens = min(args.warmup_tokens, 1)
    args.batch_size = args.batch_size or 2
    args.max_auto_batch = min(args.max_auto_batch, 8)
    if args.num_layers is None:
        args.num_layers = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark staged synthetic LLM decode components on CUDA GPUs.")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run one staged component benchmark")
    run_parser.add_argument("--stage", choices=STAGES, default="attention_kernel")
    run_parser.add_argument("--output", "-o", type=Path)
    _add_config_args(run_parser)

    matrix_parser = sub.add_parser("matrix", help="Run several stages with the same shape/config")
    matrix_parser.add_argument("--stages", nargs="+", choices=[*STAGES, "all"], default=["all"])
    matrix_parser.add_argument("--output", "-o", type=Path, required=True)
    _add_config_args(matrix_parser)

    viz_parser = sub.add_parser("visualize", help="Visualize a component CSV or nsys SQLite file")
    viz_parser.add_argument("input", type=Path)
    viz_parser.add_argument("--output", "-o", type=Path)
    viz_parser.add_argument("--summary-output", type=Path)

    report_parser = sub.add_parser("report", help="Generate a Markdown report from throughput and NCU CSVs")
    report_parser.add_argument("--throughput-csv", type=Path, required=True)
    report_parser.add_argument("--ncu-glob", required=True)
    report_parser.add_argument("--output", "-o", type=Path, required=True)
    report_parser.add_argument("--plot-output", type=Path)

    return parser


def run_one(args: argparse.Namespace) -> int:
    _apply_smoke(args)
    config = _config_from_args(args)
    result = run_stage(args.stage, config)
    row = result_to_row(result)
    print(
        f"{result.stage}: batch={result.batch_size}, layout={result.layout}, "
        f"throughput={result.throughput_toks_s:.1f} toks/s, "
        f"wall={result.wall_time_s:.3f}s",
        flush=True,
    )
    if args.output:
        write_rows([row], args.output)
        write_config(config, args.output.with_suffix(".config.json"))
    return 0


def run_stage_matrix(args: argparse.Namespace) -> int:
    _apply_smoke(args)
    config = _config_from_args(args)
    stages = _stage_list(args.stages)
    rows = []
    write_config(config, args.output.with_suffix(".config.json"))
    for result in (run_stage(stage, config) for stage in stages):
        print(
            f"{result.stage}: batch={result.batch_size}, layout={result.layout}, "
            f"throughput={result.throughput_toks_s:.1f} toks/s, "
            f"wall={result.wall_time_s:.3f}s",
            flush=True,
        )
        rows.append(result_to_row(result))
        write_rows(rows, args.output)
    write_rows(rows, args.output)
    return 0


def run_visualize(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1
    if args.input.suffix == ".sqlite":
        from component_bw.visualize import visualize_nsys

        visualize_nsys(args.input, args.output, summary_output=args.summary_output)
    else:
        from component_bw.visualize import visualize_summary

        visualize_summary(args.input, args.output)
    return 0


def run_report(args: argparse.Namespace) -> int:
    from component_bw.report import generate_report

    if not args.throughput_csv.exists():
        print(f"Error: {args.throughput_csv} not found", file=sys.stderr)
        return 1
    generate_report(args.throughput_csv, args.ncu_glob, args.output, args.plot_output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_one(args)
    if args.command == "matrix":
        return run_stage_matrix(args)
    if args.command == "visualize":
        return run_visualize(args)
    if args.command == "report":
        return run_report(args)
    parser.print_help()
    return 0
