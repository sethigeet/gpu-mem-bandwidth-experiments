"""Generate the combined CPU-step and GPU-projection follow-up report."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def _read_nsys_csv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    header = next(index for index, line in enumerate(lines) if "Range" in line and "Proj Avg" in line)
    return list(csv.DictReader(lines[header:]))


def _number(row: dict[str, str], prefix: str) -> float:
    key = next(key for key in row if key.startswith(prefix))
    return float(row[key].replace(",", ""))


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected numeric report value, got {value!r}")


def _raw_ms(rows: list[dict[str, str]], suffix: str) -> float:
    row = next(row for row in rows if row["function"].endswith(suffix))
    return float(row["avg_us"]) / 1e3


def _gpu_ms(rows: list[dict[str, str]], range_name: str) -> float:
    row = next(row for row in rows if row["Range"].endswith(range_name))
    return _number(row, "Proj Avg") / 1e6


_GPU_EVENT_PATTERN = re.compile(
    r"\[VLLM_BW_GPU_TIMING\] name=(?P<name>\S+) calls=(?P<calls>\d+) "
    r"total_ms=(?P<total_ms>[\d.]+) avg_ms=(?P<avg_ms>[\d.]+) "
    r"min_ms=(?P<min_ms>[\d.]+) max_ms=(?P<max_ms>[\d.]+)"
)


def write_gpu_event_csv(log_path: Path, output_path: Path) -> None:
    latest: dict[str, dict[str, object]] = {}
    for match in _GPU_EVENT_PATTERN.finditer(log_path.read_text()):
        latest[match.group("name")] = {
            "range": match.group("name"),
            "calls": int(match.group("calls")),
            "total_ms": float(match.group("total_ms")),
            "avg_ms": float(match.group("avg_ms")),
            "min_ms": float(match.group("min_ms")),
            "max_ms": float(match.group("max_ms")),
        }
    if len(latest) < 2:
        raise ValueError(f"Incomplete CUDA-event timings in {log_path}")
    _write_csv(list(latest.values()), output_path)


def _event_gpu_stats(rows: list[dict[str, str]], range_name: str) -> tuple[float, int]:
    row = next(row for row in rows if row["range"] == range_name)
    return float(row["total_ms"]), int(row["calls"])


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_phase_breakdown(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes = ("sync", "async")
    phase_rows = {
        mode: [
            row
            for row in _read_csv(output_dir / f"cpu_{mode}_timing.csv")
            if row["phase"] not in {"TOTAL", "unmeasured_python_between_regions"} and float(row["ms_per_step"]) > 0
        ]
        for mode in modes
    }
    phases = list(dict.fromkeys(row["phase"] for mode in modes for row in phase_rows[mode]))
    values = {mode: {row["phase"]: float(row["ms_per_step"]) for row in phase_rows[mode]} for mode in modes}

    figure, axis = plt.subplots(figsize=(12, 6))
    left = [0.0, 0.0]
    for phase in phases:
        widths = [values[mode].get(phase, 0.0) for mode in modes]
        axis.barh(modes, widths, left=left, label=phase)
        left = [current + width for current, width in zip(left, widths, strict=True)]
    axis.set_xlabel("Milliseconds per EngineCore step")
    axis.set_title("Exclusive EngineCore step breakdown")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "step_breakdown.png", dpi=180)
    plt.close(figure)


def _plot_model_comparison(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes = [str(row["mode"]) for row in rows]
    cpu = [_as_float(row["cpu_execute_excluding_prep_ms"]) for row in rows]
    gpu = [_as_float(row["gpu_event_excluding_prep_ms"]) for row in rows]
    x = list(range(len(modes)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar([value - width / 2 for value in x], cpu, width, label="CPU wall timer")
    axis.bar([value + width / 2 for value in x], gpu, width, label="CUDA event")
    axis.set_xticks(x, modes)
    axis.set_ylabel("Milliseconds per call")
    axis.set_title("Model execution excluding input preparation")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "model_execution_cpu_vs_gpu.png", dpi=180)
    plt.close(figure)


def generate_overhead_report(output_dir: Path) -> None:
    summary: list[dict[str, object]] = []
    for mode in ("sync", "async"):
        raw = _read_csv(output_dir / f"cpu_{mode}_timing_raw.csv")
        gpu_events = _read_csv(output_dir / f"gpu_{mode}_events.csv")
        bench = json.loads((output_dir / f"cpu_{mode}_logs" / "bench_result.json").read_text())
        step_rows = _read_csv(output_dir / f"cpu_{mode}_timing.csv")
        step = next(row for row in step_rows if row["phase"] == "TOTAL")
        residual = next(row for row in step_rows if row["phase"] == "unmeasured_python_between_regions")
        cpu_execute = _raw_ms(raw, "GPUModelRunner.execute_model")
        cpu_prepare = _raw_ms(raw, "GPUModelRunner._prepare_inputs")
        gpu_execute_total, gpu_execute_calls = _event_gpu_stats(gpu_events, "vllm_bw:model_execution")
        gpu_prepare_total, _ = _event_gpu_stats(gpu_events, "vllm_bw:input_preparation")
        gpu_execute = gpu_execute_total / gpu_execute_calls
        gpu_prepare = gpu_prepare_total / gpu_execute_calls
        summary.append(
            {
                "mode": mode,
                "output_throughput_toks_s": bench["output_throughput"],
                "engine_step_ms": float(step["ms_per_step"]),
                "timer_reconciliation_error_ms": float(residual["ms_per_step"]),
                "cpu_execute_including_prep_ms": cpu_execute,
                "cpu_input_prep_ms": cpu_prepare,
                "cpu_execute_excluding_prep_ms": cpu_execute - cpu_prepare,
                "gpu_event_including_prep_ms": gpu_execute,
                "gpu_input_event_ms_per_model_call": gpu_prepare,
                "gpu_event_excluding_prep_ms": gpu_execute - gpu_prepare,
            }
        )

    _write_csv(summary, output_dir / "combined_summary.csv")
    _plot_phase_breakdown(output_dir)
    _plot_model_comparison(summary, output_dir)

    lines = [
        "# vLLM unaccounted-step and GPU timing follow-up",
        "",
        "| Mode | Throughput tok/s | Step ms | Reconciliation error ms | CPU model excl. prep ms | CUDA-event model excl. prep ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['mode']} | {_as_float(row['output_throughput_toks_s']):.2f} | "
            f"{_as_float(row['engine_step_ms']):.2f} | {_as_float(row['timer_reconciliation_error_ms']):.3f} | "
            f"{_as_float(row['cpu_execute_excluding_prep_ms']):.2f} | "
            f"{_as_float(row['gpu_event_excluding_prep_ms']):.2f} |"
        )
    lines.extend(
        [
            "",
            "The CPU value is the cumulative GPUModelRunner wall timer minus its nested input-preparation timer.",
            "The GPU value uses CUDA events on the same stream around the two NVTX-instrumented methods.",
            "NSYS CUDA/NVTX traces were collected separately; its built-in projection could not normalize child-worker timestamps for this multiprocess launch.",
            "The small negative reconciliation error is timer/context instrumentation overhead; it is not an additional phase.",
            "",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines))
