import argparse
import sys
from pathlib import Path

from gpu_benchmarks.serving.scheduling_compare import (
    add_scheduling_compare_args,
    run_scheduling_comparison,
)
from gpu_benchmarks.serving.server import (
    add_client_nsys_profile_args,
    add_serve_profile_args,
    run_client_nsys_profile,
    run_serve_profile,
)
from gpu_benchmarks.serving.timing_analysis import write_reconciled_step_csv
from gpu_benchmarks.serving.timing_artifacts import generate_overhead_artifacts, write_gpu_event_csv
from gpu_benchmarks.serving.visualize import summarize_nsys, visualize_nsys
from gpu_benchmarks.serving.visualize.nsys import write_summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-memory-benchmarks vllm",
        description="Profile vLLM serving memory bandwidth.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run vLLM serve and drive request load")
    add_serve_profile_args(serve)

    client_nsys = subparsers.add_parser(
        "client-nsys",
        help="Run vLLM serve normally and profile only the request-load client with nsys",
    )
    add_client_nsys_profile_args(client_nsys)

    scheduling_compare = subparsers.add_parser(
        "scheduling-compare",
        help="Compare request policies across paired async and sync NSYS trials",
    )
    add_scheduling_compare_args(scheduling_compare)

    visualize = subparsers.add_parser("visualize", help="Visualize an nsys SQLite export")
    visualize.add_argument("input", type=Path)
    visualize.add_argument("--output", "-o", type=Path)
    visualize.add_argument("--summary-output", type=Path)
    visualize.add_argument(
        "--full-trace",
        action="store_true",
        help="Use the full trace instead of filtering to the measured NVTX range",
    )

    summarize = subparsers.add_parser("summarize", help="Write only the nsys summary CSV")
    summarize.add_argument("input", type=Path)
    summarize.add_argument("--output", "-o", type=Path, required=True)
    summarize.add_argument(
        "--full-trace",
        action="store_true",
        help="Use the full trace instead of filtering to the measured NVTX range",
    )

    timing_summary = subparsers.add_parser(
        "timing-summary",
        help="Reconcile detailed EngineCore phase timings from a vLLM server log",
    )
    timing_summary.add_argument("--log", type=Path, required=True)
    timing_summary.add_argument("--output", "-o", type=Path, required=True)
    timing_summary.add_argument("--raw-output", type=Path)

    overhead_artifacts = subparsers.add_parser(
        "overhead-artifacts",
        help="Build combined CPU/GPU overhead tables and plots",
    )
    overhead_artifacts.add_argument("--input-dir", type=Path, required=True)

    gpu_events = subparsers.add_parser(
        "gpu-event-summary",
        help="Extract cumulative CUDA-event timings from a server log",
    )
    gpu_events.add_argument("--log", type=Path, required=True)
    gpu_events.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return run_serve_profile(args)
    if args.command == "client-nsys":
        return run_client_nsys_profile(args)
    if args.command == "scheduling-compare":
        return run_scheduling_comparison(args)
    if args.command == "visualize":
        if not args.input.exists():
            print(f"Error: {args.input} not found", file=sys.stderr)
            return 1
        visualize_nsys(
            args.input,
            args.output,
            summary_output=args.summary_output,
            measured_only=not args.full_trace,
        )
        return 0
    if args.command == "summarize":
        if not args.input.exists():
            print(f"Error: {args.input} not found", file=sys.stderr)
            return 1
        rows = summarize_nsys(args.input, measured_only=not args.full_trace)
        write_summary_csv(rows, args.output)
        return 0
    if args.command == "timing-summary":
        write_reconciled_step_csv(args.log, args.output, args.raw_output)
        print(f"Wrote {args.output}")
        return 0
    if args.command == "overhead-artifacts":
        generate_overhead_artifacts(args.input_dir)
        print(f"Wrote analysis artifacts under {args.input_dir}")
        return 0
    if args.command == "gpu-event-summary":
        write_gpu_event_csv(args.log, args.output)
        print(f"Wrote {args.output}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
