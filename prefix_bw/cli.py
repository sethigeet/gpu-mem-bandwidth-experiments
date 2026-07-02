"""Reproduce the Feather paper's prefix-homogeneity claims on vLLM.

Each `sweep` subcommand varies one workload knob, runs the batch on vLLM with prefix
caching enabled, and writes a CSV of decode throughput. `visualize` renders either a
sweep CSV (throughput vs the swept knob) or an nsys `.sqlite` (DRAM bandwidth timeline,
reusing the llm_bw visualizer) so the bandwidth claim can be checked directly.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

from prefix_bw.scheduler import SCHEDULES


def _resolve_model(name: str) -> str:
    from prefix_bw.models import MODEL_REGISTRY

    return MODEL_REGISTRY.get(name, name)


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def _parse_strings(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def _write_csv(rows: list[dict], output: Path) -> None:
    if not rows:
        print("No results to write", file=sys.stderr)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


def _scheduler_kwargs(args: argparse.Namespace) -> dict:
    return {
        "schedule": args.schedule,
        "prefix_batch_size": args.prefix_batch_size,
        "min_prefix_batch_size": args.min_prefix_batch_size,
        "prefix_hash_chunk_size": args.prefix_hash_chunk_size,
        "min_shared_prefix_len": args.min_shared_prefix_len,
        "prefix_auto_min_fill": args.prefix_auto_min_fill,
        "offline_prefix_min_gain": args.offline_prefix_min_gain,
        "offline_prefix_min_fill": args.offline_prefix_min_fill,
        "offline_prefix_max_waves": args.offline_prefix_max_waves,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Feather prefix-homogeneity claims on vLLM.")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="phi-3-mini", help="Registry key or raw HF model id")
        p.add_argument("--dtype", default="float16")
        p.add_argument("--num-requests", type=int, default=128)
        p.add_argument("--decode-tokens", type=int, default=64)
        p.add_argument("--max-num-seqs", type=int, default=256, help="vLLM batch size")
        p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        p.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Allow vLLM to load model repositories with custom code",
        )
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-warmup", action="store_true", help="Skip prefix-cache warmup")
        p.add_argument(
            "--schedule",
            choices=SCHEDULES,
            default="single",
            help="Request scheduling strategy: current all-at-once run or prefix-homogeneous microbatches",
        )
        p.add_argument(
            "--prefix-batch-size",
            type=int,
            help="Maximum requests per prefix-grouped microbatch; defaults to --max-num-seqs",
        )
        p.add_argument(
            "--min-prefix-batch-size",
            type=int,
            default=64,
            help="Minimum same-prefix group size to split out under prefix-adaptive scheduling",
        )
        p.add_argument(
            "--prefix-hash-chunk-size",
            type=int,
            default=64,
            help="Token chunk size for non-oracle prefix-hash-adaptive scheduling",
        )
        p.add_argument(
            "--min-shared-prefix-len",
            type=int,
            default=128,
            help="Minimum discovered shared-prefix length for prefix-hash-adaptive scheduling",
        )
        p.add_argument(
            "--prefix-auto-min-fill",
            type=float,
            default=0.20,
            help="Minimum split-batch fill ratio for prefix-hash-auto scheduling",
        )
        p.add_argument(
            "--offline-prefix-min-gain",
            type=float,
            default=0.10,
            help="Minimum estimated net gain before offline-prefix-wave departs from vLLM-style batching",
        )
        p.add_argument(
            "--offline-prefix-min-fill",
            type=float,
            default=0.20,
            help="Minimum total wave fill before offline-prefix-wave splits into prefix-local waves",
        )
        p.add_argument(
            "--offline-prefix-max-waves",
            type=int,
            default=0,
            help="Maximum generated waves for offline-prefix-wave; 0 means unlimited",
        )
        p.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path")

    homo = sub.add_parser("homogeneity", help="Experiment 1 / Fig 4: vary prefix homogeneity")
    add_common(homo)
    homo.add_argument("--values", type=_parse_floats, default=_parse_floats("0,0.002,0.25,0.5,0.75,1.0"))
    homo.add_argument("--prefix-len", type=int, default=2048)
    homo.add_argument("--suffix-len", type=int, default=32)

    plen = sub.add_parser("prefix-length", help="Experiment 2 / Fig 5: vary shared prefix length")
    add_common(plen)
    plen.add_argument("--values", type=_parse_floats, default=_parse_floats("0,0.25,0.5,0.75,1.0"))
    plen.add_argument("--total-len", type=int, default=2048)

    groups = sub.add_parser("num-groups", help="Experiment 3 / Fig 6: vary number of prefix groups")
    add_common(groups)
    groups.add_argument("--values", type=_parse_ints, default=_parse_ints("1,2,4,8,16"))
    groups.add_argument("--prefix-len", type=int, default=2048)
    groups.add_argument("--suffix-len", type=int, default=32)

    bs = sub.add_parser("batch-size", help="Experiments 5-6 / Figs 8-9: batch size, homo vs hetero")
    add_common(bs)
    bs.add_argument("--values", type=_parse_ints, default=_parse_ints("16,32,64,128,256"))
    bs.add_argument("--prefix-len", type=int, default=2048)
    bs.add_argument("--suffix-len", type=int, default=32)
    bs.add_argument("--hetero-groups", type=int, default=5, help="Prefix groups in heterogeneous run")

    sched = sub.add_parser("schedule", help="Compare current mixed batching with prefix-grouped scheduling")
    add_common(sched)
    sched.add_argument("--values", type=_parse_ints, default=_parse_ints("1,2,4,8,16"))
    sched.add_argument("--prefix-len", type=int, default=2048)
    sched.add_argument("--suffix-len", type=int, default=32)
    sched.add_argument(
        "--schedules",
        type=_parse_strings,
        default=_parse_strings("single,prefix-adaptive,prefix-grouped"),
        help="Comma-separated scheduling strategies to compare",
    )

    viz = sub.add_parser("visualize", help="Plot a sweep CSV or an nsys .sqlite")
    viz.add_argument("input", type=Path)
    viz.add_argument("--output", "-o", type=Path)

    return parser


def _ensure_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on the remote GPU host.")


def run_homogeneity(args: argparse.Namespace) -> int:
    _ensure_cuda()
    from prefix_bw import workload
    from prefix_bw.runner import build_llm, get_vocab_size, result_to_row, run_workload

    max_model_len = args.prefix_len + args.suffix_len + args.decode_tokens + 16
    llm = build_llm(
        _resolve_model(args.model),
        args.dtype,
        max_model_len,
        args.max_num_seqs,
        args.gpu_memory_utilization,
        args.trust_remote_code,
    )
    vocab = get_vocab_size(llm)

    rows = []
    for beta in args.values:
        wl = workload.build_homogeneity_fraction(
            beta, args.num_requests, args.prefix_len, args.suffix_len, vocab, args.seed
        )
        print(f"[homogeneity] {wl.description}", flush=True)
        result = run_workload(
            llm,
            wl,
            "homogeneity_fraction",
            "beta",
            beta,
            args.decode_tokens,
            args.max_num_seqs,
            warmup=not args.no_warmup,
            **_scheduler_kwargs(args),
        )
        print(f"  decode throughput = {result.decode_throughput_toks_s:.1f} toks/s", flush=True)
        rows.append(result_to_row(result))

    _write_csv(rows, args.output)
    return 0


def run_prefix_length(args: argparse.Namespace) -> int:
    _ensure_cuda()
    from prefix_bw import workload
    from prefix_bw.runner import build_llm, get_vocab_size, result_to_row, run_workload

    max_model_len = args.total_len + args.decode_tokens + 16
    llm = build_llm(
        _resolve_model(args.model),
        args.dtype,
        max_model_len,
        args.max_num_seqs,
        args.gpu_memory_utilization,
        args.trust_remote_code,
    )
    vocab = get_vocab_size(llm)

    rows = []
    for p in args.values:
        wl = workload.build_shared_length(p, args.num_requests, args.total_len, vocab, args.seed)
        print(f"[prefix-length] {wl.description}", flush=True)
        result = run_workload(
            llm,
            wl,
            "shared_length",
            "share_fraction",
            p,
            args.decode_tokens,
            args.max_num_seqs,
            warmup=not args.no_warmup,
            **_scheduler_kwargs(args),
        )
        print(f"  decode throughput = {result.decode_throughput_toks_s:.1f} toks/s", flush=True)
        rows.append(result_to_row(result))

    _write_csv(rows, args.output)
    return 0


def run_num_groups(args: argparse.Namespace) -> int:
    _ensure_cuda()
    from prefix_bw import workload
    from prefix_bw.runner import build_llm, get_vocab_size, result_to_row, run_workload

    max_model_len = args.prefix_len + args.suffix_len + args.decode_tokens + 16
    llm = build_llm(
        _resolve_model(args.model),
        args.dtype,
        max_model_len,
        args.max_num_seqs,
        args.gpu_memory_utilization,
        args.trust_remote_code,
    )
    vocab = get_vocab_size(llm)

    rows = []
    for g in args.values:
        wl = workload.build_num_groups(g, args.num_requests, args.prefix_len, args.suffix_len, vocab, args.seed)
        print(f"[num-groups] {wl.description}", flush=True)
        result = run_workload(
            llm,
            wl,
            "num_groups",
            "num_groups",
            g,
            args.decode_tokens,
            args.max_num_seqs,
            warmup=not args.no_warmup,
            **_scheduler_kwargs(args),
        )
        print(f"  decode throughput = {result.decode_throughput_toks_s:.1f} toks/s", flush=True)
        rows.append(result_to_row(result))

    _write_csv(rows, args.output)
    return 0


def run_batch_size(args: argparse.Namespace) -> int:
    _ensure_cuda()
    import torch

    from prefix_bw import workload
    from prefix_bw.runner import build_llm, get_vocab_size, result_to_row, run_workload

    max_model_len = args.prefix_len + args.suffix_len + args.decode_tokens + 16

    rows = []
    for batch in args.values:
        # max_num_seqs is the per-forward-pass batch size, so it must be rebuilt per point.
        llm = build_llm(
            _resolve_model(args.model),
            args.dtype,
            max_model_len,
            batch,
            args.gpu_memory_utilization,
            args.trust_remote_code,
        )
        vocab = get_vocab_size(llm)

        for series, n_groups in (("homogeneous", 1), ("heterogeneous", args.hetero_groups)):
            wl = workload.build_num_groups(
                n_groups, args.num_requests, args.prefix_len, args.suffix_len, vocab, args.seed
            )
            print(f"[batch-size] batch={batch} {series}: {wl.description}", flush=True)
            result = run_workload(
                llm,
                wl,
                "batch_size",
                "batch_size",
                batch,
                args.decode_tokens,
                batch,
                series=series,
                warmup=not args.no_warmup,
                **_scheduler_kwargs(args),
            )
            print(f"  decode throughput = {result.decode_throughput_toks_s:.1f} toks/s", flush=True)
            rows.append(result_to_row(result))

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    _write_csv(rows, args.output)
    return 0


def run_schedule(args: argparse.Namespace) -> int:
    unknown_schedules = sorted(set(args.schedules) - set(SCHEDULES))
    if unknown_schedules:
        print(f"Unknown schedules: {', '.join(unknown_schedules)}", file=sys.stderr)
        return 1

    _ensure_cuda()
    import torch

    from prefix_bw import workload
    from prefix_bw.runner import build_llm, get_vocab_size, result_to_row, run_workload

    max_model_len = args.prefix_len + args.suffix_len + args.decode_tokens + 16

    rows = []
    for schedule in args.schedules:
        llm = build_llm(
            _resolve_model(args.model),
            args.dtype,
            max_model_len,
            args.max_num_seqs,
            args.gpu_memory_utilization,
            args.trust_remote_code,
        )
        vocab = get_vocab_size(llm)
        for n_groups in args.values:
            wl = workload.build_num_groups(
                n_groups, args.num_requests, args.prefix_len, args.suffix_len, vocab, args.seed
            )
            print(f"[schedule] groups={n_groups} schedule={schedule}: {wl.description}", flush=True)
            result = run_workload(
                llm,
                wl,
                "schedule",
                "num_groups",
                n_groups,
                args.decode_tokens,
                args.max_num_seqs,
                series=schedule,
                warmup=not args.no_warmup,
                **(_scheduler_kwargs(args) | {"schedule": schedule}),
            )
            print(f"  decode throughput = {result.decode_throughput_toks_s:.1f} toks/s", flush=True)
            rows.append(result_to_row(result))
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    _write_csv(rows, args.output)
    return 0


def run_visualize(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    if args.input.suffix == ".sqlite":
        from prefix_bw.visualize import visualize_nsys

        visualize_nsys(args.input, args.output)
    else:
        from prefix_bw.visualize import visualize_sweep

        visualize_sweep(args.input, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "homogeneity": run_homogeneity,
        "prefix-length": run_prefix_length,
        "num-groups": run_num_groups,
        "batch-size": run_batch_size,
        "schedule": run_schedule,
        "visualize": run_visualize,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
