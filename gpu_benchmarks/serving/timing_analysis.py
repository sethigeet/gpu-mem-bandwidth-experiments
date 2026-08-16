"""Parse detailed vLLM timing instrumentation into a reconciled step budget."""

import re
from dataclasses import dataclass
from pathlib import Path

from gpu_benchmarks.utils import write_dict_rows

_TIMING_PATTERN = re.compile(
    r"(?:\((?P<process>[^)]*pid=\d+)\)\s+)?"
    r"\[VLLM_BENCH_TIMING\] "
    r"name=(?P<name>\S+) "
    r"calls=(?P<calls>\d+) "
    r"total_ns=(?P<total_ns>\d+) "
    r"avg_ns=(?P<avg_ns>[\d.]+) "
    r"min_ns=(?P<min_ns>\d+) "
    r"max_ns=(?P<max_ns>\d+)"
)

_STEP_PHASES = {
    "EngineCore.step": (
        "request_check",
        "schedule",
        "execute_model_submit",
        "grammar_bitmask",
        "model_future_wait",
        "fallback_sample_tokens",
        "process_aborts",
        "update_from_output",
    ),
    "EngineCore.step_with_batch_queue": (
        "request_check",
        "schedule",
        "execute_model_submit",
        "initial_grammar_bitmask",
        "initial_sample_submit",
        "queue_append",
        "queue_pop",
        "model_future_wait",
        "failed_execute_wait",
        "process_aborts",
        "update_from_output",
        "deferred_take_draft_tokens",
        "deferred_grammar_bitmask",
        "deferred_sample_submit",
        "deferred_queue_append",
    ),
}


@dataclass(frozen=True)
class Timing:
    process: str
    name: str
    calls: int
    total_ns: int
    avg_ns: float
    min_ns: int
    max_ns: int


def parse_latest_timings(log_path: Path) -> list[Timing]:
    """Return the last cumulative record for every process/function pair."""

    latest: dict[tuple[str, str], Timing] = {}
    for match in _TIMING_PATTERN.finditer(log_path.read_text()):
        timing = Timing(
            process=match.group("process") or "unknown_process",
            name=match.group("name"),
            calls=int(match.group("calls")),
            total_ns=int(match.group("total_ns")),
            avg_ns=float(match.group("avg_ns")),
            min_ns=int(match.group("min_ns")),
            max_ns=int(match.group("max_ns")),
        )
        key = (timing.process, timing.name)
        previous = latest.get(key)
        if previous is None or timing.calls >= previous.calls:
            latest[key] = timing
    return sorted(latest.values(), key=lambda timing: (timing.process, timing.name))


def reconciled_step_rows(timings: list[Timing]) -> list[dict[str, object]]:
    """Build exclusive step budgets and expose the small untimed residual."""

    by_process_name = {(timing.process, timing.name): timing for timing in timings}
    rows: list[dict[str, object]] = []
    for (process, name), step in by_process_name.items():
        if name not in _STEP_PHASES:
            continue
        phase_prefix = f"{name}::"
        observed_phases = {
            timing.name.removeprefix(phase_prefix): timing
            for timing in timings
            if timing.process == process and timing.name.startswith(phase_prefix)
        }
        phases = [observed_phases.get(phase_name) for phase_name in _STEP_PHASES[name]]
        phase_total_ns = sum(phase.total_ns for phase in phases if phase is not None)
        for phase_name, phase in zip(_STEP_PHASES[name], phases, strict=True):
            calls = phase.calls if phase is not None else 0
            total_ns = phase.total_ns if phase is not None else 0
            avg_ns = phase.avg_ns if phase is not None else 0.0
            rows.append(
                {
                    "process": process,
                    "step": name,
                    "phase": phase_name,
                    "calls": calls,
                    "total_ms": total_ns / 1e6,
                    "ms_per_step": total_ns / step.calls / 1e6,
                    "ms_per_call": avg_ns / 1e6,
                    "pct_of_step": total_ns / step.total_ns * 100,
                }
            )
        residual_ns = step.total_ns - phase_total_ns
        rows.append(
            {
                "process": process,
                "step": name,
                "phase": "unmeasured_python_between_regions",
                "calls": step.calls,
                "total_ms": residual_ns / 1e6,
                "ms_per_step": residual_ns / step.calls / 1e6,
                "ms_per_call": residual_ns / step.calls / 1e6,
                "pct_of_step": residual_ns / step.total_ns * 100,
            }
        )
        rows.append(
            {
                "process": process,
                "step": name,
                "phase": "TOTAL",
                "calls": step.calls,
                "total_ms": step.total_ns / 1e6,
                "ms_per_step": step.avg_ns / 1e6,
                "ms_per_call": step.avg_ns / 1e6,
                "pct_of_step": 100.0,
            }
        )
    return rows


def write_reconciled_step_csv(
    log_path: Path,
    output_path: Path,
    raw_output_path: Path | None = None,
) -> None:
    timings = parse_latest_timings(log_path)
    rows = reconciled_step_rows(timings)
    if not rows:
        raise ValueError(f"No detailed EngineCore step timings found in {log_path}")
    write_dict_rows(rows, output_path)
    if raw_output_path is not None:
        raw_rows: list[dict[str, object]] = [
            {
                "process": timing.process,
                "function": timing.name,
                "calls": timing.calls,
                "total_ms": timing.total_ns / 1e6,
                "avg_us": timing.avg_ns / 1e3,
                "min_us": timing.min_ns / 1e3,
                "max_us": timing.max_ns / 1e3,
            }
            for timing in timings
        ]
        write_dict_rows(raw_rows, raw_output_path)
