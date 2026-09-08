"""Render aligned EngineCore CPU ranges and GPU kernels from NSYS exports."""

import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from gpu_benchmarks.profiling.nsys import load_nvtx_ranges, table_names
from gpu_benchmarks.utils import write_dict_rows

_CPU_RANGE_PREFIX = "gpu_memory:vllm:cpu:"
_CALL_SUFFIX = re.compile(r":call=(?P<call>[0-9]+)$")


@dataclass(frozen=True)
class TimelineSelection:
    label: str
    source: Path
    start_ns: int
    end_ns: int
    cpu_ranges: pd.DataFrame
    kernels: pd.DataFrame
    step_starts_ns: tuple[int, ...]
    step_calls: tuple[int, ...]


def _load_kernels(connection: sqlite3.Connection) -> pd.DataFrame:
    available = table_names(connection)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in available or "StringIds" not in available:
        return pd.DataFrame(columns=["start", "end", "name"])
    return pd.read_sql_query(
        """
        SELECT k.start, k.end, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """,
        connection,
    )


def _parse_cpu_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    cpu = ranges[ranges["name"].str.startswith(_CPU_RANGE_PREFIX, na=False)].copy()
    if cpu.empty:
        return pd.DataFrame(columns=["start", "end", "name", "phase", "call"])

    def parse_name(name: object) -> tuple[str, int]:
        text = str(name).removeprefix(_CPU_RANGE_PREFIX)
        match = _CALL_SUFFIX.search(text)
        if match is None:
            return text, -1
        return text[: match.start()], int(match.group("call"))

    parsed = cpu["name"].map(parse_name)
    cpu["phase"] = parsed.map(lambda item: item[0])
    cpu["call"] = parsed.map(lambda item: item[1])
    return cpu.sort_values(["start", "end"]).reset_index(drop=True)


def _schedule_ranges(cpu: pd.DataFrame) -> pd.DataFrame:
    for phase in (
        "EngineCore.step_with_batch_queue::schedule",
        "EngineCore.step::schedule",
        "Scheduler.schedule",
    ):
        matching = cpu[cpu["phase"] == phase]
        if not matching.empty:
            return matching.sort_values("start").reset_index(drop=True)
    return pd.DataFrame(columns=cpu.columns)


def _select_window(
    source: Path,
    label: str,
    *,
    steps: int,
    start_step: int | None,
) -> TimelineSelection:
    with sqlite3.connect(source) as connection:
        cpu = _parse_cpu_ranges(load_nvtx_ranges(connection))
        kernels = _load_kernels(connection)
    schedules = _schedule_ranges(cpu)
    if len(schedules) < steps + 1:
        raise ValueError(
            f"{source} has {len(schedules)} scheduler ranges; at least {steps + 1} are required "
            "to bound a continuous timeline"
        )

    if start_step is None:
        start_index = max(0, (len(schedules) - steps) // 2)
        start_index = min(start_index, len(schedules) - steps - 1)
    else:
        start_index = start_step
        if start_index < 0:
            start_index += len(schedules)
        if start_index < 0 or start_index + steps >= len(schedules):
            raise ValueError(
                f"start step {start_step} cannot select {steps} complete steps from "
                f"{len(schedules)} scheduler ranges in {source}"
            )

    selected_schedules = schedules.iloc[start_index : start_index + steps + 1]
    start_ns = int(selected_schedules.iloc[0]["start"])
    end_ns = int(selected_schedules.iloc[-1]["start"])
    cpu_window = cpu[(cpu["end"] > start_ns) & (cpu["start"] < end_ns)].copy()
    kernel_window = kernels[(kernels["end"] > start_ns) & (kernels["start"] < end_ns)].copy()
    if kernel_window.empty:
        raise ValueError(
            f"No GPU kernels overlap the selected scheduler window in {source}; "
            "ensure NSYS traced the vLLM server rather than only the client"
        )
    return TimelineSelection(
        label=label,
        source=source,
        start_ns=start_ns,
        end_ns=end_ns,
        cpu_ranges=cpu_window,
        kernels=kernel_window,
        step_starts_ns=tuple(int(value) for value in selected_schedules["start"]),
        step_calls=tuple(int(value) for value in selected_schedules.iloc[:-1]["call"]),
    )


def _cpu_category(phase: str) -> str:
    lowered = phase.lower()
    if lowered.endswith("::schedule") or phase == "Scheduler.schedule":
        return "Scheduler"
    if "execute_model" in lowered or "model_execution" in lowered:
        return "Model submit"
    if "sample" in lowered or "future_wait" in lowered or "grammar" in lowered:
        return "Sample / wait"
    if "update_from_output" in lowered or "process_aborts" in lowered:
        return "Update"
    return "Other CPU"


def _is_step_container(phase: str) -> bool:
    """Return whether an NVTX range only provides an inclusive step boundary."""
    return phase in {"EngineCore.step", "EngineCore.step_with_batch_queue"}


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("flash", "fmha", "attention", "paged_attention")):
        return "GPU attention"
    if any(token in lowered for token in ("gemm", "cutlass", "cublas", "matmul")):
        return "GPU GEMM"
    if any(token in lowered for token in ("topk", "sampling", "multinomial", "argmax")):
        return "GPU sampling"
    return "GPU other"


def _merged_intervals(rows: pd.DataFrame) -> list[tuple[int, int]]:
    intervals = sorted(
        (int(start), int(end)) for start, end in rows[["start", "end"]].itertuples(index=False, name=None)
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_duration(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
) -> int:
    total = 0
    first_index = second_index = 0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        total += max(0, min(first_end, second_end) - max(first_start, second_start))
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return total


def _event_rows(selection: TimelineSelection) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start_raw, end_raw, _name, phase_raw, call_raw in selection.cpu_ranges.itertuples(
        index=False,
        name=None,
    ):
        start = max(int(start_raw), selection.start_ns)
        end = min(int(end_raw), selection.end_ns)
        phase = str(phase_raw)
        if _is_step_container(phase):
            continue
        rows.append(
            {
                "timeline": selection.label,
                "resource": "CPU",
                "category": _cpu_category(phase),
                "name": phase,
                "call": int(call_raw),
                "start_ms": (start - selection.start_ns) / 1e6,
                "end_ms": (end - selection.start_ns) / 1e6,
                "duration_ms": (end - start) / 1e6,
            }
        )
    for start_raw, end_raw, name_raw in selection.kernels.itertuples(index=False, name=None):
        start = max(int(start_raw), selection.start_ns)
        end = min(int(end_raw), selection.end_ns)
        name = str(name_raw)
        rows.append(
            {
                "timeline": selection.label,
                "resource": "GPU",
                "category": _kernel_category(name),
                "name": name,
                "call": "",
                "start_ms": (start - selection.start_ns) / 1e6,
                "end_ms": (end - selection.start_ns) / 1e6,
                "duration_ms": (end - start) / 1e6,
            }
        )
    return rows


def _summary_row(selection: TimelineSelection) -> dict[str, object]:
    schedules = selection.cpu_ranges[
        selection.cpu_ranges["phase"].isin(
            (
                "EngineCore.step_with_batch_queue::schedule",
                "EngineCore.step::schedule",
                "Scheduler.schedule",
            )
        )
    ]
    # Explicit EngineCore schedule ranges and the decorated Scheduler.schedule
    # describe the same calls. Prefer the former to avoid double-counting.
    explicit = schedules[schedules["phase"].str.startswith("EngineCore.", na=False)]
    if not explicit.empty:
        schedules = explicit
    scheduler_intervals = _merged_intervals(schedules)
    gpu_intervals = _merged_intervals(selection.kernels)
    scheduler_ns = _duration(scheduler_intervals)
    gpu_ns = _duration(gpu_intervals)
    overlap_ns = _intersection_duration(scheduler_intervals, gpu_intervals)
    window_ns = selection.end_ns - selection.start_ns
    return {
        "timeline": selection.label,
        "source": str(selection.source),
        "steps": len(selection.step_calls),
        "first_scheduler_call": selection.step_calls[0],
        "window_ms": window_ns / 1e6,
        "scheduler_ms": scheduler_ns / 1e6,
        "scheduler_gpu_overlap_ms": overlap_ns / 1e6,
        "scheduler_gpu_overlap_pct": (100 * overlap_ns / scheduler_ns if scheduler_ns else 0.0),
        "gpu_active_ms": gpu_ns / 1e6,
        "gpu_active_pct": 100 * gpu_ns / window_ns,
        "gpu_kernel_count": len(selection.kernels),
    }


def _plot_selection(selection: TimelineSelection, axis: plt.Axes) -> None:
    lane_positions = {
        "GPU kernels": 0,
        "Other CPU": 1,
        "Update": 2,
        "Sample / wait": 3,
        "Model submit": 4,
        "Scheduler": 5,
    }
    cpu_colors = {
        "Scheduler": "#e68613",
        "Model submit": "#7b5ea7",
        "Sample / wait": "#c44e52",
        "Update": "#4c78a8",
        "Other CPU": "#9d9d9d",
    }
    gpu_colors = {
        "GPU attention": "#1b9e77",
        "GPU GEMM": "#66a61e",
        "GPU sampling": "#17becf",
        "GPU other": "#b2df8a",
    }

    for start_raw, end_raw, _name, phase_raw, _call in selection.cpu_ranges.itertuples(
        index=False,
        name=None,
    ):
        phase = str(phase_raw)
        if _is_step_container(phase):
            continue
        category = _cpu_category(phase)
        start = (max(int(start_raw), selection.start_ns) - selection.start_ns) / 1e6
        end = (min(int(end_raw), selection.end_ns) - selection.start_ns) / 1e6
        axis.broken_barh(
            [(start, end - start)],
            (lane_positions[category] - 0.32, 0.64),
            facecolors=cpu_colors[category],
            edgecolors="none",
            alpha=0.85,
        )

    for start_raw, end_raw, name_raw in selection.kernels.itertuples(index=False, name=None):
        category = _kernel_category(str(name_raw))
        start = (max(int(start_raw), selection.start_ns) - selection.start_ns) / 1e6
        end = (min(int(end_raw), selection.end_ns) - selection.start_ns) / 1e6
        axis.broken_barh(
            [(start, max(end - start, 0.002))],
            (lane_positions["GPU kernels"] - 0.32, 0.64),
            facecolors=gpu_colors[category],
            edgecolors="none",
        )

    for step_index, timestamp in enumerate(selection.step_starts_ns):
        position = (timestamp - selection.start_ns) / 1e6
        axis.axvline(position, color="#333333", linestyle="--", linewidth=0.8, alpha=0.55)
        if step_index < len(selection.step_calls):
            axis.text(
                position,
                5.52,
                f"step {selection.step_calls[step_index]}",
                fontsize=8,
                ha="left",
                va="bottom",
            )

    axis.set_yticks(list(lane_positions.values()), list(lane_positions))
    axis.set_ylim(-0.65, 5.9)
    axis.set_xlim(0, (selection.end_ns - selection.start_ns) / 1e6)
    axis.set_title(selection.label, loc="left", fontweight="bold")
    axis.grid(True, axis="x", alpha=0.2)


def visualize_scheduler_timeline(
    inputs: list[Path],
    output: Path,
    *,
    labels: list[str] | None = None,
    steps: int = 3,
    start_step: int | None = None,
    events_output: Path | None = None,
    summary_output: Path | None = None,
) -> list[dict[str, object]]:
    """Plot continuous scheduler steps against GPU kernels for one or more traces."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if labels is not None and len(labels) != len(inputs):
        raise ValueError("the number of labels must match the number of inputs")
    effective_labels = labels or [path.stem for path in inputs]
    selections = [
        _select_window(path, label, steps=steps, start_step=start_step)
        for path, label in zip(inputs, effective_labels, strict=True)
    ]

    figure, axes = plt.subplots(
        len(selections),
        1,
        figsize=(15, max(4.8, 4.4 * len(selections))),
        squeeze=False,
    )
    for selection, axis in zip(selections, axes[:, 0], strict=True):
        _plot_selection(selection, axis)
    axes[-1, 0].set_xlabel("Time from first selected scheduler call (ms)")
    figure.suptitle(f"vLLM CPU/GPU overlap across {steps} continuous EngineCore steps")
    legend = [
        Patch(facecolor="#e68613", label="CPU scheduler"),
        Patch(facecolor="#7b5ea7", label="CPU model submission"),
        Patch(facecolor="#c44e52", label="CPU sampling / wait"),
        Patch(facecolor="#4c78a8", label="CPU output update"),
        Patch(facecolor="#1b9e77", label="GPU attention"),
        Patch(facecolor="#66a61e", label="GPU GEMM"),
        Patch(facecolor="#17becf", label="GPU sampling"),
        Patch(facecolor="#b2df8a", label="GPU other"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    event_rows = [row for selection in selections for row in _event_rows(selection)]
    summary_rows = [_summary_row(selection) for selection in selections]
    if events_output is not None:
        write_dict_rows(event_rows, events_output)
    if summary_output is not None:
        write_dict_rows(summary_rows, summary_output)
    return summary_rows


def read_timeline_summary(path: Path) -> list[dict[str, str]]:
    """Read a generated timeline summary for downstream report tooling."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
