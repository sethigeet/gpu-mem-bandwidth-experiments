from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from vllm_bw.bench_parse import parse_bench_artifacts
from vllm_bw.models import resolve_model
from vllm_bw.serve_profile import (
    _bench_command,
    _run_and_log,
    _server_command,
    _terminate_process_group,
    _vllm_executable,
    _wait_for_health,
    add_serve_profile_args,
)
from vllm_bw.visualize import extract_dram_utilization, visualize_nsys


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    scheduling_mode: str
    order_in_pair: int
    seed: int
    output_throughput_toks_s: float
    dram_avg_pct: float
    dram_p95_pct: float
    dram_samples: int
    benchmark_duration_s: float
    total_output_tokens: int
    completed_requests: int
    failed_requests: int
    vllm_version: str
    trial_dir: str


def add_scheduling_compare_args(parser: argparse.ArgumentParser) -> None:
    add_serve_profile_args(parser)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"results/vllm_scheduling_compare_{timestamp}"),
    )
    parser.add_argument("--repetitions", type=int, default=5, help="Paired trials per mode")
    parser.add_argument("--cooldown-s", type=float, default=0)
    parser.add_argument("--nsys-trace", default="nvtx")
    parser.add_argument("--nsys-gpu-metrics-devices", default="all")
    parser.add_argument("--nsys-gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument(
        "--fail-on-scheduler-warning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if vLLM reports that async scheduling was disabled",
    )


def _probe_environment() -> dict[str, str]:
    executable = _vllm_executable()
    version_result = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version = version_result.stdout.strip() or version_result.stderr.strip()
    help_text = subprocess.run(
        [executable, "serve", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "--async-scheduling" not in help_text:
        help_text = subprocess.run(
            [executable, "serve", "--help=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    if "--async-scheduling" not in help_text or "--no-async-scheduling" not in help_text:
        raise RuntimeError(
            "Installed vLLM does not expose both --async-scheduling and "
            "--no-async-scheduling; upgrade vLLM before running this comparison"
        )

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "vllm_executable": executable,
        "vllm_version": version,
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _scheduler_warning(log_text: str) -> str | None:
    for line in log_text.splitlines():
        lowered = line.lower()
        if ("async scheduling" in lowered or "asynchronous scheduling" in lowered) and (
            "disabled" in lowered or "not supported" in lowered
        ):
            return line.strip()
    return None


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_trial(
    args: argparse.Namespace,
    model: str,
    environment: dict[str, str],
    trial_id: int,
    mode: str,
    order_in_pair: int,
) -> TrialResult:
    trial_args = copy.copy(args)
    trial_args.scheduling_mode = mode
    trial_args.seed = args.seed + trial_id
    trial_dir = args.output_dir / "trials" / f"trial_{trial_id:03d}_{mode}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_args.log_dir = trial_dir

    result_path = trial_dir / "bench_result.json"
    bench_log_path = trial_dir / "bench_nsys.log"
    profile_prefix = trial_dir / "profile"
    sqlite_path = trial_dir / "profile.sqlite"
    server_cmd = _server_command(trial_args, model)
    bench_cmd = _bench_command(
        trial_args,
        model,
        trial_args.num_prompts,
        result_path=result_path,
        metadata={
            "scheduling_mode": mode,
            "trial_id": trial_id,
            "seed": trial_args.seed,
        },
    )
    config = {
        "trial_id": trial_id,
        "scheduling_mode": mode,
        "order_in_pair": order_in_pair,
        "seed": trial_args.seed,
        "vllm_version": environment["vllm_version"],
        "server_argv": server_cmd,
        "bench_argv": bench_cmd,
    }
    (trial_dir / "trial_config.json").write_text(json.dumps(config, indent=2) + "\n")

    server_log_path = trial_dir / "server.log"
    server_log = server_log_path.open("w")
    server = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        _wait_for_health(
            f"http://{trial_args.host}:{trial_args.port}/health",
            server,
            trial_args.server_timeout_s,
        )
        if trial_args.warmup_prompts > 0:
            warmup_cmd = _bench_command(trial_args, model, trial_args.warmup_prompts)
            _run_and_log(
                warmup_cmd,
                trial_dir / "warmup.log",
                trial_args.bench_timeout_s,
            )

        nsys_cmd = [
            "nsys",
            "profile",
            "--trace",
            trial_args.nsys_trace,
            "--gpu-metrics-devices",
            trial_args.nsys_gpu_metrics_devices,
            "--gpu-metrics-frequency",
            str(trial_args.nsys_gpu_metrics_frequency),
            "--duration",
            "0",
            "--output",
            str(profile_prefix),
            "--force-overwrite=true",
            *bench_cmd,
        ]
        _run_and_log(nsys_cmd, bench_log_path, trial_args.bench_timeout_s)
    finally:
        _terminate_process_group(server)
        server_log.close()

    warning = _scheduler_warning(server_log_path.read_text())
    if mode == "async" and warning and args.fail_on_scheduler_warning:
        raise RuntimeError(f"vLLM did not honor async scheduling: {warning}")

    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            f"--output={sqlite_path}",
            f"{profile_prefix}.nsys-rep",
        ],
        check=True,
    )
    metrics = parse_bench_artifacts(result_path, bench_log_path)
    if metrics.failed_requests:
        raise RuntimeError(f"Trial {trial_id} ({mode}) had {metrics.failed_requests} failed requests")
    dram = extract_dram_utilization(
        sqlite_path,
        measured_only=False,
        trim_idle_edges=True,
    )
    visualize_nsys(
        sqlite_path,
        trial_dir / "profile.png",
        summary_output=trial_dir / "profile_summary.csv",
        measured_only=False,
    )
    return TrialResult(
        trial_id=trial_id,
        scheduling_mode=mode,
        order_in_pair=order_in_pair,
        seed=trial_args.seed,
        output_throughput_toks_s=metrics.output_throughput_toks_s,
        dram_avg_pct=dram.avg_pct,
        dram_p95_pct=dram.p95_pct,
        dram_samples=dram.samples,
        benchmark_duration_s=metrics.benchmark_duration_s,
        total_output_tokens=metrics.total_output_tokens,
        completed_requests=metrics.completed_requests,
        failed_requests=metrics.failed_requests,
        vllm_version=environment["vllm_version"],
        trial_dir=str(trial_dir),
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {value!r}")


def _aggregate(results: list[TrialResult], output_dir: Path) -> None:
    by_mode = {mode: [result for result in results if result.scheduling_mode == mode] for mode in ("sync", "async")}
    summary_rows: list[dict[str, object]] = []
    for mode, mode_results in by_mode.items():
        throughputs = [result.output_throughput_toks_s for result in mode_results]
        dram = [result.dram_avg_pct for result in mode_results]
        summary_rows.append(
            {
                "scheduling_mode": mode,
                "n": len(mode_results),
                "throughput_mean_toks_s": _mean(throughputs),
                "throughput_stdev_toks_s": _stdev(throughputs),
                "dram_avg_mean_pct": _mean(dram),
                "dram_avg_stdev_pct": _stdev(dram),
            }
        )
    _write_csv(summary_rows, output_dir / "summary.csv")

    pairs: list[dict[str, object]] = []
    for trial_id in sorted({result.trial_id for result in results}):
        pair = {result.scheduling_mode: result for result in results if result.trial_id == trial_id}
        if set(pair) != {"async", "sync"}:
            continue
        async_result = pair["async"]
        sync_result = pair["sync"]
        pairs.append(
            {
                "trial_id": trial_id,
                "async_throughput_toks_s": async_result.output_throughput_toks_s,
                "sync_throughput_toks_s": sync_result.output_throughput_toks_s,
                "throughput_delta_toks_s": (
                    async_result.output_throughput_toks_s - sync_result.output_throughput_toks_s
                ),
                "throughput_change_pct": 100
                * (async_result.output_throughput_toks_s - sync_result.output_throughput_toks_s)
                / sync_result.output_throughput_toks_s,
                "async_dram_avg_pct": async_result.dram_avg_pct,
                "sync_dram_avg_pct": sync_result.dram_avg_pct,
                "dram_delta_pct_points": async_result.dram_avg_pct - sync_result.dram_avg_pct,
                "dram_change_pct": 100
                * (async_result.dram_avg_pct - sync_result.dram_avg_pct)
                / sync_result.dram_avg_pct,
            }
        )
    _write_csv(pairs, output_dir / "paired_comparison.csv")

    impact_rows: list[dict[str, object]] = []
    for metric, field in (
        ("output_throughput_toks_s", "output_throughput_toks_s"),
        ("dram_avg_pct", "dram_avg_pct"),
    ):
        sync_values = [float(getattr(result, field)) for result in by_mode["sync"]]
        async_values = [float(getattr(result, field)) for result in by_mode["async"]]
        sync_mean = _mean(sync_values)
        async_mean = _mean(async_values)
        impact_rows.append(
            {
                "metric": metric,
                "sync_mean": sync_mean,
                "sync_stdev": _stdev(sync_values),
                "async_mean": async_mean,
                "async_stdev": _stdev(async_values),
                "async_minus_sync": async_mean - sync_mean,
                "change_pct": 100 * (async_mean - sync_mean) / sync_mean,
            }
        )
    _write_csv(impact_rows, output_dir / "impact.csv")
    _plot_comparison(summary_rows, output_dir / "comparison.png")


def _plot_comparison(summary_rows: list[dict[str, object]], output: Path) -> None:
    rows = {str(row["scheduling_mode"]): row for row in summary_rows}
    modes = ["sync", "async"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(
        modes,
        [_as_float(rows[mode]["throughput_mean_toks_s"]) for mode in modes],
        yerr=[_as_float(rows[mode]["throughput_stdev_toks_s"]) for mode in modes],
        capsize=5,
    )
    axes[0].set_ylabel("Output tokens / second")
    axes[0].set_title("Serving throughput")
    axes[1].bar(
        modes,
        [_as_float(rows[mode]["dram_avg_mean_pct"]) for mode in modes],
        yerr=[_as_float(rows[mode]["dram_avg_stdev_pct"]) for mode in modes],
        capsize=5,
    )
    axes[1].set_ylabel("Percent of sustained peak")
    axes[1].set_title("Average DRAM bandwidth utilization")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_scheduling_comparison(args: argparse.Namespace) -> int:
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = resolve_model(args.model)
    environment = _probe_environment()
    effective_concurrency = args.max_concurrency or args.max_num_seqs
    planned_request_waves = math.ceil(args.num_prompts / effective_concurrency)
    if planned_request_waves < 2:
        print(
            "warning: num-prompts does not exceed effective concurrency; "
            "this workload has no queued replacement requests",
            flush=True,
        )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": environment,
        "model": model,
        "repetitions": args.repetitions,
        "workload": {
            "random_prefix_len": args.random_prefix_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "num_prompts": args.num_prompts,
            "warmup_prompts": args.warmup_prompts,
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
            "max_num_seqs": args.max_num_seqs,
            "planned_request_waves": planned_request_waves,
            "base_seed": args.seed,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    results: list[TrialResult] = []
    for trial_id in range(args.repetitions):
        order = ("async", "sync") if trial_id % 2 == 0 else ("sync", "async")
        for order_in_pair, mode in enumerate(order):
            print(
                f"Running scheduling trial {trial_id + 1}/{args.repetitions}: {mode}",
                flush=True,
            )
            result = _run_trial(
                args,
                model,
                environment,
                trial_id,
                mode,
                order_in_pair,
            )
            results.append(result)
            _write_csv([asdict(item) for item in results], args.output_dir / "trials.csv")
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

    _aggregate(results, args.output_dir)
    print(f"Wrote scheduling comparison to {args.output_dir}")
    return 0
