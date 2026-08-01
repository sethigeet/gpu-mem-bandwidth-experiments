from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
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
    _server_environment,
    _terminate_process_group,
    _vllm_executable,
    _wait_for_health,
    add_serve_profile_args,
)
from vllm_bw.visualize import extract_dram_utilization, visualize_nsys


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    scheduling_policy: str
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
    mean_ttft_ms: float = math.nan
    mean_tpot_ms: float = math.nan
    mean_itl_ms: float = math.nan


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
        "--collect-dram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect NSYS DRAM metrics; disable for throughput-only comparisons",
    )
    parser.add_argument(
        "--scheduling-policies",
        nargs="+",
        help=(
            "Request policies to compare. Defaults to the singular "
            "--scheduling-policy value; use space-separated policy names."
        ),
    )
    parser.add_argument(
        "--fail-on-scheduler-warning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if vLLM reports that async scheduling was disabled",
    )
    parser.add_argument(
        "--pyspy-duration",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Record server and child-process CPU stacks with py-spy (0 disables)",
    )


def _probe_environment(args: argparse.Namespace, policies: list[str]) -> dict[str, str]:
    executable = _vllm_executable(args)
    version_result = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_output = version_result.stdout.strip() or version_result.stderr.strip()
    version = version_output.splitlines()[-1]
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
    if "--scheduling-policy" not in help_text:
        raise RuntimeError("Installed vLLM does not expose --scheduling-policy")
    unavailable = [policy for policy in policies if policy not in help_text]
    if unavailable:
        raise RuntimeError(
            "Installed vLLM does not advertise the requested scheduling policies: "
            f"{', '.join(unavailable)}. Install the policy-enabled vLLM fork first."
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


def _nsys_profile_failure(log_text: str) -> str | None:
    failure_markers = (
        "TargetProfilingFailed",
        "GPU Metrics event chronological order was broken",
    )
    for line in log_text.splitlines():
        if any(marker in line for marker in failure_markers):
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


def _append_pyspy_warning(log_path: Path, message: str) -> None:
    warning = f"warning: {message}"
    print(warning, flush=True)
    with log_path.open("a") as log:
        log.write(f"{warning}\n")


def _start_pyspy(server_pid: int, duration_s: int, trial_dir: Path) -> subprocess.Popen | None:
    if duration_s <= 0:
        return None
    log_path = trial_dir / "pyspy.log"
    command = [
        "py-spy",
        "record",
        "--pid",
        str(server_pid),
        "--subprocesses",
        "--native",
        "--format",
        "speedscope",
        "-o",
        str(trial_dir / "pyspy.speedscope.json"),
        "-d",
        str(duration_s),
    ]
    print(f"$ {' '.join(command)}", flush=True)
    try:
        with log_path.open("w") as log:
            return subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except OSError as exc:
        _append_pyspy_warning(log_path, f"could not start py-spy: {exc}")
        return None


def _finish_pyspy(
    process: subprocess.Popen | None,
    duration_s: int,
    trial_dir: Path,
) -> None:
    if process is None:
        return
    log_path = trial_dir / "pyspy.log"
    try:
        returncode = process.wait(timeout=duration_s + 60)
    except subprocess.TimeoutExpired:
        _append_pyspy_warning(
            log_path,
            f"py-spy exceeded its {duration_s + 60}s wait timeout; terminating it",
        )
        try:
            process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _append_pyspy_warning(log_path, f"could not stop py-spy: {exc}")
            return
    except OSError as exc:
        _append_pyspy_warning(log_path, f"could not wait for py-spy: {exc}")
        return
    if returncode != 0:
        _append_pyspy_warning(log_path, f"py-spy exited with status {returncode}")


_TIMING_PATTERN = re.compile(
    r"\[VLLM_BW_TIMING\] "
    r"name=(?P<name>\S+) "
    r"calls=(?P<calls>\d+) "
    r"total_ns=(?P<total_ns>\d+) "
    r"avg_ns=(?P<avg_ns>[\d.]+) "
    r"min_ns=(?P<min_ns>\d+) "
    r"max_ns=(?P<max_ns>\d+)"
)


def _aggregate_scheduler_timings(
    results: list[TrialResult],
    output_dir: Path,
) -> None:
    timing_rows: list[dict[str, object]] = []
    for result in results:
        log_path = Path(result.trial_dir) / "server.log"
        latest_by_name: dict[str, re.Match[str]] = {}
        for match in _TIMING_PATTERN.finditer(log_path.read_text()):
            name = match.group("name")
            previous = latest_by_name.get(name)
            if previous is None or int(match.group("calls")) > int(previous.group("calls")):
                latest_by_name[name] = match
        for name, match in latest_by_name.items():
            timing_rows.append(
                {
                    "trial_id": result.trial_id,
                    "scheduling_policy": result.scheduling_policy,
                    "scheduling_mode": result.scheduling_mode,
                    "function": name,
                    "calls": int(match.group("calls")),
                    "total_ms": int(match.group("total_ns")) / 1e6,
                    "avg_us": float(match.group("avg_ns")) / 1e3,
                    "min_us": int(match.group("min_ns")) / 1e3,
                    "max_us": int(match.group("max_ns")) / 1e3,
                }
            )
    if not timing_rows:
        return

    _write_csv(timing_rows, output_dir / "scheduler_timings.csv")
    summary_rows: list[dict[str, object]] = []
    keys = dict.fromkeys(
        (
            str(row["scheduling_policy"]),
            str(row["scheduling_mode"]),
            str(row["function"]),
        )
        for row in timing_rows
    )
    for policy, mode, function in keys:
        matching = [
            row
            for row in timing_rows
            if row["scheduling_policy"] == policy and row["scheduling_mode"] == mode and row["function"] == function
        ]
        summary_rows.append(
            {
                "scheduling_policy": policy,
                "scheduling_mode": mode,
                "function": function,
                "n": len(matching),
                "calls_mean": _mean([_as_float(row["calls"]) for row in matching]),
                "total_ms_mean": _mean([_as_float(row["total_ms"]) for row in matching]),
                "avg_us_mean": _mean([_as_float(row["avg_us"]) for row in matching]),
                "avg_us_stdev": _stdev([_as_float(row["avg_us"]) for row in matching]),
                "max_us_mean": _mean([_as_float(row["max_us"]) for row in matching]),
            }
        )
    _write_csv(summary_rows, output_dir / "scheduler_timing_summary.csv")


def _run_trial(
    args: argparse.Namespace,
    model: str,
    environment: dict[str, str],
    trial_id: int,
    policy: str,
    mode: str,
    order_in_pair: int,
) -> TrialResult:
    trial_args = copy.copy(args)
    trial_args.scheduling_policy = policy
    trial_args.scheduling_mode = mode
    trial_args.seed = args.seed + trial_id
    trial_dir = args.output_dir / "trials" / f"trial_{trial_id:03d}_{policy}_{mode}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_args.log_dir = trial_dir
    if args.torch_profile_dir is not None:
        trial_args.torch_profile_dir = trial_dir / "torch_profile"

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
            "scheduling_policy": policy,
            "scheduling_mode": mode,
            "trial_id": trial_id,
            "seed": trial_args.seed,
        },
    )
    pyspy_config = {
        "requested": trial_args.pyspy_duration > 0,
        "ran": False,
        "duration_s": trial_args.pyspy_duration,
        "output": (str(trial_dir / "pyspy.speedscope.json") if trial_args.pyspy_duration > 0 else None),
    }
    config = {
        "trial_id": trial_id,
        "scheduling_policy": policy,
        "scheduling_mode": mode,
        "order_in_pair": order_in_pair,
        "seed": trial_args.seed,
        "vllm_version": environment["vllm_version"],
        "server_argv": server_cmd,
        "bench_argv": bench_cmd,
        "pyspy": pyspy_config,
        "torch_profile_dir": (str(trial_args.torch_profile_dir) if trial_args.torch_profile_dir is not None else None),
    }
    (trial_dir / "trial_config.json").write_text(json.dumps(config, indent=2) + "\n")

    server_log_path = trial_dir / "server.log"
    server_log = server_log_path.open("w")
    server = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=_server_environment(trial_args),
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
            warmup_cmd = _bench_command(
                trial_args,
                model,
                trial_args.warmup_prompts,
                enable_profile=False,
            )
            _run_and_log(
                warmup_cmd,
                trial_dir / "warmup.log",
                trial_args.bench_timeout_s,
            )

        pyspy = _start_pyspy(server.pid, trial_args.pyspy_duration, trial_dir)
        pyspy_config["ran"] = pyspy is not None
        (trial_dir / "trial_config.json").write_text(json.dumps(config, indent=2) + "\n")
        try:
            if trial_args.collect_dram:
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
                nsys_output = _run_and_log(
                    nsys_cmd,
                    bench_log_path,
                    trial_args.bench_timeout_s,
                )
                profile_failure = _nsys_profile_failure(nsys_output)
                if profile_failure:
                    raise RuntimeError(f"NSYS GPU metric collection failed: {profile_failure}")
            else:
                _run_and_log(bench_cmd, bench_log_path, trial_args.bench_timeout_s)
        finally:
            _finish_pyspy(pyspy, trial_args.pyspy_duration, trial_dir)
    finally:
        _terminate_process_group(server)
        server_log.close()

    warning = _scheduler_warning(server_log_path.read_text())
    if mode == "async" and warning and args.fail_on_scheduler_warning:
        raise RuntimeError(f"vLLM did not honor async scheduling: {warning}")

    metrics = parse_bench_artifacts(result_path, bench_log_path)
    if metrics.failed_requests:
        raise RuntimeError(f"Trial {trial_id} ({policy}, {mode}) had {metrics.failed_requests} failed requests")
    dram_avg_pct = math.nan
    dram_p95_pct = math.nan
    dram_samples = 0
    if trial_args.collect_dram:
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
        dram_avg_pct = dram.avg_pct
        dram_p95_pct = dram.p95_pct
        dram_samples = dram.samples
    return TrialResult(
        trial_id=trial_id,
        scheduling_policy=policy,
        scheduling_mode=mode,
        order_in_pair=order_in_pair,
        seed=trial_args.seed,
        output_throughput_toks_s=metrics.output_throughput_toks_s,
        dram_avg_pct=dram_avg_pct,
        dram_p95_pct=dram_p95_pct,
        dram_samples=dram_samples,
        benchmark_duration_s=metrics.benchmark_duration_s,
        total_output_tokens=metrics.total_output_tokens,
        completed_requests=metrics.completed_requests,
        failed_requests=metrics.failed_requests,
        vllm_version=environment["vllm_version"],
        trial_dir=str(trial_dir),
        mean_ttft_ms=metrics.mean_ttft_ms,
        mean_tpot_ms=metrics.mean_tpot_ms,
        mean_itl_ms=metrics.mean_itl_ms,
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return math.nan
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {value!r}")


def _change_pct(new: float, baseline: float) -> float:
    return 100 * (new - baseline) / baseline


def _aggregate(results: list[TrialResult], output_dir: Path) -> None:
    policies = list(dict.fromkeys(result.scheduling_policy for result in results))
    by_policy_mode = {
        (policy, mode): [
            result for result in results if result.scheduling_policy == policy and result.scheduling_mode == mode
        ]
        for policy in policies
        for mode in ("sync", "async")
    }
    summary_rows: list[dict[str, object]] = []
    for policy in policies:
        for mode in ("sync", "async"):
            mode_results = by_policy_mode[(policy, mode)]
            throughputs = [result.output_throughput_toks_s for result in mode_results]
            dram = [result.dram_avg_pct for result in mode_results]
            ttft = [result.mean_ttft_ms for result in mode_results]
            tpot = [result.mean_tpot_ms for result in mode_results]
            itl = [result.mean_itl_ms for result in mode_results]
            summary_rows.append(
                {
                    "scheduling_policy": policy,
                    "scheduling_mode": mode,
                    "n": len(mode_results),
                    "throughput_mean_toks_s": _mean(throughputs),
                    "throughput_stdev_toks_s": _stdev(throughputs),
                    "dram_avg_mean_pct": _mean(dram),
                    "dram_avg_stdev_pct": _stdev(dram),
                    "mean_ttft_mean_ms": _mean(ttft),
                    "mean_ttft_stdev_ms": _stdev(ttft),
                    "mean_tpot_mean_ms": _mean(tpot),
                    "mean_tpot_stdev_ms": _stdev(tpot),
                    "mean_itl_mean_ms": _mean(itl),
                    "mean_itl_stdev_ms": _stdev(itl),
                }
            )
    _write_csv(summary_rows, output_dir / "summary.csv")

    pairs: list[dict[str, object]] = []
    for policy in policies:
        for trial_id in sorted({result.trial_id for result in results}):
            pair = {
                result.scheduling_mode: result
                for result in results
                if result.scheduling_policy == policy and result.trial_id == trial_id
            }
            if set(pair) != {"async", "sync"}:
                continue
            async_result = pair["async"]
            sync_result = pair["sync"]
            pairs.append(
                {
                    "scheduling_policy": policy,
                    "trial_id": trial_id,
                    "async_throughput_toks_s": async_result.output_throughput_toks_s,
                    "sync_throughput_toks_s": sync_result.output_throughput_toks_s,
                    "throughput_delta_toks_s": (
                        async_result.output_throughput_toks_s - sync_result.output_throughput_toks_s
                    ),
                    "throughput_change_pct": _change_pct(
                        async_result.output_throughput_toks_s,
                        sync_result.output_throughput_toks_s,
                    ),
                    "async_dram_avg_pct": async_result.dram_avg_pct,
                    "sync_dram_avg_pct": sync_result.dram_avg_pct,
                    "dram_delta_pct_points": (async_result.dram_avg_pct - sync_result.dram_avg_pct),
                    "dram_change_pct": _change_pct(
                        async_result.dram_avg_pct,
                        sync_result.dram_avg_pct,
                    ),
                    "async_mean_ttft_ms": async_result.mean_ttft_ms,
                    "sync_mean_ttft_ms": sync_result.mean_ttft_ms,
                    "ttft_delta_ms": (async_result.mean_ttft_ms - sync_result.mean_ttft_ms),
                    "async_mean_tpot_ms": async_result.mean_tpot_ms,
                    "sync_mean_tpot_ms": sync_result.mean_tpot_ms,
                    "tpot_delta_ms": (async_result.mean_tpot_ms - sync_result.mean_tpot_ms),
                }
            )
    _write_csv(pairs, output_dir / "paired_comparison.csv")

    impact_rows: list[dict[str, object]] = []
    policy_comparison_rows: list[dict[str, object]] = []
    for policy in policies:
        policy_metrics: dict[str, tuple[float, float, float, float]] = {}
        for metric, field in (
            ("output_throughput_toks_s", "output_throughput_toks_s"),
            ("dram_avg_pct", "dram_avg_pct"),
            ("mean_ttft_ms", "mean_ttft_ms"),
            ("mean_tpot_ms", "mean_tpot_ms"),
            ("mean_itl_ms", "mean_itl_ms"),
        ):
            sync_values = [float(getattr(result, field)) for result in by_policy_mode[(policy, "sync")]]
            async_values = [float(getattr(result, field)) for result in by_policy_mode[(policy, "async")]]
            sync_mean = _mean(sync_values)
            async_mean = _mean(async_values)
            policy_metrics[metric] = (
                sync_mean,
                _stdev(sync_values),
                async_mean,
                _stdev(async_values),
            )
            impact_rows.append(
                {
                    "scheduling_policy": policy,
                    "metric": metric,
                    "sync_mean": sync_mean,
                    "sync_stdev": _stdev(sync_values),
                    "async_mean": async_mean,
                    "async_stdev": _stdev(async_values),
                    "async_minus_sync": async_mean - sync_mean,
                    "change_pct": _change_pct(async_mean, sync_mean),
                }
            )
        throughput = policy_metrics["output_throughput_toks_s"]
        dram = policy_metrics["dram_avg_pct"]
        ttft = policy_metrics["mean_ttft_ms"]
        tpot = policy_metrics["mean_tpot_ms"]
        itl = policy_metrics["mean_itl_ms"]
        policy_comparison_rows.append(
            {
                "scheduling_policy": policy,
                "n_pairs": len(by_policy_mode[(policy, "sync")]),
                "sync_throughput_mean_toks_s": throughput[0],
                "sync_throughput_stdev_toks_s": throughput[1],
                "async_throughput_mean_toks_s": throughput[2],
                "async_throughput_stdev_toks_s": throughput[3],
                "throughput_delta_toks_s": throughput[2] - throughput[0],
                "throughput_change_pct": _change_pct(throughput[2], throughput[0]),
                "sync_dram_mean_pct": dram[0],
                "sync_dram_stdev_pct": dram[1],
                "async_dram_mean_pct": dram[2],
                "async_dram_stdev_pct": dram[3],
                "dram_delta_pct_points": dram[2] - dram[0],
                "dram_change_pct": _change_pct(dram[2], dram[0]),
                "sync_mean_ttft_ms": ttft[0],
                "async_mean_ttft_ms": ttft[2],
                "ttft_delta_ms": ttft[2] - ttft[0],
                "ttft_change_pct": _change_pct(ttft[2], ttft[0]),
                "sync_mean_tpot_ms": tpot[0],
                "async_mean_tpot_ms": tpot[2],
                "tpot_delta_ms": tpot[2] - tpot[0],
                "tpot_change_pct": _change_pct(tpot[2], tpot[0]),
                "sync_mean_itl_ms": itl[0],
                "async_mean_itl_ms": itl[2],
                "itl_delta_ms": itl[2] - itl[0],
                "itl_change_pct": _change_pct(itl[2], itl[0]),
            }
        )
    _write_csv(impact_rows, output_dir / "impact.csv")
    _write_csv(policy_comparison_rows, output_dir / "policy_comparison.csv")
    _plot_comparison(summary_rows, output_dir / "comparison.png")


def _plot_comparison(summary_rows: list[dict[str, object]], output: Path) -> None:
    rows = {(str(row["scheduling_policy"]), str(row["scheduling_mode"])): row for row in summary_rows}
    policies = list(dict.fromkeys(str(row["scheduling_policy"]) for row in summary_rows))
    modes = ["sync", "async"]
    positions = list(range(len(policies)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(policies) * 3), 4.5))
    for mode_index, mode in enumerate(modes):
        offsets = [position + (mode_index - 0.5) * width for position in positions]
        axes[0].bar(
            offsets,
            [_as_float(rows[(policy, mode)]["throughput_mean_toks_s"]) for policy in policies],
            yerr=[_as_float(rows[(policy, mode)]["throughput_stdev_toks_s"]) for policy in policies],
            width=width,
            capsize=5,
            label=mode,
        )
        axes[1].bar(
            offsets,
            [_as_float(rows[(policy, mode)]["dram_avg_mean_pct"]) for policy in policies],
            yerr=[_as_float(rows[(policy, mode)]["dram_avg_stdev_pct"]) for policy in policies],
            width=width,
            capsize=5,
            label=mode,
        )
    axes[0].set_ylabel("Output tokens / second")
    axes[0].set_title("Serving throughput")
    axes[1].set_ylabel("Percent of sustained peak")
    axes[1].set_title("Average DRAM bandwidth utilization")
    for axis in axes:
        axis.set_xticks(positions, policies, rotation=15, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_scheduling_comparison(args: argparse.Namespace) -> int:
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if args.pyspy_duration < 0:
        raise ValueError("--pyspy-duration must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = resolve_model(args.model)
    policies = list(dict.fromkeys(args.scheduling_policies or [args.scheduling_policy]))
    if not policies:
        raise ValueError("At least one scheduling policy is required")
    invalid_policies = [policy for policy in policies if re.fullmatch(r"[A-Za-z0-9_-]+", policy) is None]
    if invalid_policies:
        raise ValueError(
            "Scheduling policy names may contain only letters, digits, underscores, and hyphens: "
            f"{', '.join(invalid_policies)}"
        )
    environment = _probe_environment(args, policies)
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
        "attention_backend": args.attention_backend,
        "scheduling_policies": policies,
        "repetitions": args.repetitions,
        "collect_dram": args.collect_dram,
        "pyspy_duration_s": args.pyspy_duration,
        "torch_profile_enabled": args.torch_profile_dir is not None,
        "scheduler_profiling": {
            "enabled": os.environ.get("VLLM_BW_PROFILE_SCHEDULER") == "1",
            "interval": os.environ.get("VLLM_BW_PROFILE_INTERVAL", "100"),
        },
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
        policy_order = policies if trial_id % 2 == 0 else list(reversed(policies))
        for policy_index, policy in enumerate(policy_order):
            mode_order = ("async", "sync") if (trial_id + policy_index) % 2 == 0 else ("sync", "async")
            for order_in_pair, mode in enumerate(mode_order):
                print(
                    f"Running trial {trial_id + 1}/{args.repetitions}: policy={policy}, mode={mode}",
                    flush=True,
                )
                result = _run_trial(
                    args,
                    model,
                    environment,
                    trial_id,
                    policy,
                    mode,
                    order_in_pair,
                )
                results.append(result)
                _write_csv([asdict(item) for item in results], args.output_dir / "trials.csv")
                if args.cooldown_s > 0:
                    time.sleep(args.cooldown_s)

    _aggregate(results, args.output_dir)
    _aggregate_scheduler_timings(results, args.output_dir)
    print(f"Wrote scheduling comparison to {args.output_dir}")
    return 0
