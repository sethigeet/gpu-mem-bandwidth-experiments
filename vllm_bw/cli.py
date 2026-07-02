from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm_bw.serve_profile import (
    add_client_nsys_profile_args,
    add_serve_profile_args,
    run_client_nsys_profile,
    run_serve_profile,
)
from vllm_bw.visualize import summarize_nsys, visualize_nsys
from vllm_bw.visualize.nsys import write_summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile vLLM serving memory bandwidth.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run vLLM serve and drive request load")
    add_serve_profile_args(serve)

    client_nsys = subparsers.add_parser(
        "client-nsys",
        help="Run vLLM serve normally and profile only the request-load client with nsys",
    )
    add_client_nsys_profile_args(client_nsys)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return run_serve_profile(args)
    if args.command == "client-nsys":
        return run_client_nsys_profile(args)
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
