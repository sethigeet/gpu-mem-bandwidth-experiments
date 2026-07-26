from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchMetrics:
    output_throughput_toks_s: float
    benchmark_duration_s: float
    total_output_tokens: int
    completed_requests: int
    failed_requests: int
    mean_ttft_ms: float
    mean_tpot_ms: float
    mean_itl_ms: float
    source: str


_STDOUT_PATTERNS = {
    "output_throughput": re.compile(
        r"Output token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    "duration": re.compile(
        r"Benchmark duration \(s\):\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    "total_output": re.compile(r"Total (?:generated|output) tokens:\s*(\d+)", re.IGNORECASE),
    "completed": re.compile(r"Successful requests:\s*(\d+)", re.IGNORECASE),
    "failed": re.compile(r"Failed requests:\s*(\d+)", re.IGNORECASE),
    "mean_ttft_ms": re.compile(
        r"Mean TTFT \(ms\):\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    "mean_tpot_ms": re.compile(
        r"Mean TPOT \(ms\):\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    "mean_itl_ms": re.compile(
        r"Mean ITL \(ms\):\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
}


def _number(data: dict[str, object], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"Expected numeric {key!r} in vLLM benchmark result, got {value!r}")
    return float(value)


def parse_bench_json(path: Path) -> BenchMetrics:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty vLLM benchmark result list in {path}")
        data = data[-1]
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    duration = _number(data, "duration")
    total_output = int(_number(data, "total_output_tokens"))
    throughput = _number(data, "output_throughput")
    if throughput <= 0 and duration > 0:
        throughput = total_output / duration

    return BenchMetrics(
        output_throughput_toks_s=throughput,
        benchmark_duration_s=duration,
        total_output_tokens=total_output,
        completed_requests=int(_number(data, "completed")),
        failed_requests=int(_number(data, "failed")),
        mean_ttft_ms=_number(data, "mean_ttft_ms", math.nan),
        mean_tpot_ms=_number(data, "mean_tpot_ms", math.nan),
        mean_itl_ms=_number(data, "mean_itl_ms", math.nan),
        source="json",
    )


def _stdout_match(text: str, name: str, required: bool = True) -> str:
    match = _STDOUT_PATTERNS[name].search(text)
    if match is None:
        if required:
            raise ValueError(f"Could not find {name!r} in vLLM benchmark output")
        return "0"
    return match.group(1)


def parse_bench_stdout(text: str) -> BenchMetrics:
    duration = float(_stdout_match(text, "duration"))
    total_output = int(_stdout_match(text, "total_output"))
    throughput_match = _STDOUT_PATTERNS["output_throughput"].search(text)
    throughput = float(throughput_match.group(1)) if throughput_match is not None else total_output / duration
    return BenchMetrics(
        output_throughput_toks_s=throughput,
        benchmark_duration_s=duration,
        total_output_tokens=total_output,
        completed_requests=int(_stdout_match(text, "completed")),
        failed_requests=int(_stdout_match(text, "failed", required=False)),
        mean_ttft_ms=float(_stdout_match(text, "mean_ttft_ms", required=False)),
        mean_tpot_ms=float(_stdout_match(text, "mean_tpot_ms", required=False)),
        mean_itl_ms=float(_stdout_match(text, "mean_itl_ms", required=False)),
        source="stdout",
    )


def parse_bench_artifacts(json_path: Path, log_path: Path) -> BenchMetrics:
    if json_path.exists():
        return parse_bench_json(json_path)
    return parse_bench_stdout(log_path.read_text())
