import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gpu_benchmarks.profiling.nsys import filter_by_nvtx, load_nvtx_ranges, table_names
from gpu_benchmarks.utils import write_dict_rows

MEASURED_RANGE_PATTERN = r"gpu_memory:vllm:serve:bench"
MAX_PLOT_METRIC_ROWS = 200_000


@dataclass(frozen=True)
class DramUtilization:
    avg_pct: float
    p95_pct: float
    samples: int
    metric_names: tuple[str, ...]


@dataclass(frozen=True)
class GpuActivityStats:
    mean_pct: float
    median_pct: float
    p5_pct: float
    p95_pct: float
    pct_samples_below_10: float
    pct_samples_above_80: float
    samples: int
    metric_names: tuple[str, ...]


@dataclass(frozen=True)
class GpuActivityUtilization:
    sm_active: GpuActivityStats | None
    gr_active: GpuActivityStats | None


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {value!r}")


def _as_int(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"Expected an integer value, got {value!r}")


def _relevant_metric_clause(alias: str = "i") -> str:
    return f"""
    (
        {alias}.metricName LIKE '%DRAM%'
        OR {alias}.metricName LIKE 'SMs Active%'
        OR {alias}.metricName LIKE 'SM Issue%'
        OR {alias}.metricName LIKE 'Tensor Active%'
        OR {alias}.metricName LIKE 'GR Active%'
        OR {alias}.metricName LIKE '%GPU Active%'
        OR {alias}.metricName LIKE '%GPU Utilization%'
    )
    """


def _filter_by_nvtx(
    df: pd.DataFrame,
    nvtx: pd.DataFrame,
    include_pattern: str,
    start_col: str,
) -> tuple[pd.DataFrame, str]:
    include_ranges = nvtx[nvtx["name"].str.contains(include_pattern, regex=True, na=False)]
    if include_ranges.empty:
        return df.copy(), "full_trace"
    return (
        filter_by_nvtx(df, nvtx, include_pattern, timestamp_column=start_col),
        "nvtx_measured_range",
    )


def load_nsys_metrics(path: Path, measured_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    conn = sqlite3.connect(path)
    available_tables = table_names(conn)
    nvtx = load_nvtx_ranges(conn)

    if "GPU_METRICS" not in available_tables or "TARGET_INFO_GPU_METRICS" not in available_tables:
        conn.close()
        return pd.DataFrame(), pd.DataFrame(), "missing_gpu_metrics"

    relevant_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM GPU_METRICS m
        JOIN TARGET_INFO_GPU_METRICS i ON m.metricId = i.metricId
        WHERE
        """
        + _relevant_metric_clause("i"),
    ).fetchone()[0]
    stride = max(1, int(relevant_count) // MAX_PLOT_METRIC_ROWS)

    metrics = pd.read_sql_query(
        """
        WITH relevant_metrics AS (
            SELECT
                m.timestamp,
                m.metricId,
                CAST(m.value AS REAL) AS value,
                i.metricName AS metric_name,
                ROW_NUMBER() OVER (
                    PARTITION BY m.metricId ORDER BY m.timestamp
                ) AS sample_number
            FROM GPU_METRICS m
            JOIN TARGET_INFO_GPU_METRICS i ON m.metricId = i.metricId
            WHERE
        """
        + _relevant_metric_clause("i")
        + """
        )
        SELECT timestamp, metricId, value, metric_name
        FROM relevant_metrics
        WHERE ((sample_number - 1) % ?) = 0
        ORDER BY timestamp
        """,
        conn,
        params=[stride],
    )

    kernels = pd.DataFrame(columns=["start", "end", "name"])
    if "CUPTI_ACTIVITY_KIND_KERNEL" in available_tables and "StringIds" in available_tables:
        kernels = pd.read_sql_query(
            """
            SELECT k.start, k.end, s.value as name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            ORDER BY k.start
            """,
            conn,
        )

    conn.close()

    window = "full_trace"
    if measured_only:
        metrics, window = _filter_by_nvtx(metrics, nvtx, MEASURED_RANGE_PATTERN, "timestamp")
        if not kernels.empty:
            kernels, _ = _filter_by_nvtx(kernels, nvtx, MEASURED_RANGE_PATTERN, "start")

    return metrics, kernels, window


def _metric_subset(metrics: pd.DataFrame, contains: str) -> pd.DataFrame:
    return metrics[metrics["metric_name"].str.contains(contains, case=False, na=False)].copy()


def _dram_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    dram = _metric_subset(metrics, "DRAM")
    throughput = dram[dram["metric_name"].str.contains("throughput", case=False, na=False)]
    return throughput if not throughput.empty else dram


def _trim_to_active_dram_window(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    dram = _dram_metrics(metrics)
    active = dram[dram["value"].astype(float) > 0]
    if active.empty:
        raise ValueError(f"No active DRAM utilization samples found in {path}")
    start = active["timestamp"].min()
    end = active["timestamp"].max()
    return metrics[(metrics["timestamp"] >= start) & (metrics["timestamp"] <= end)].copy()


def _activity_stats(metrics: pd.DataFrame) -> GpuActivityStats | None:
    if metrics.empty:
        return None
    values = metrics["value"].astype(float)
    return GpuActivityStats(
        mean_pct=float(values.mean()),
        median_pct=float(values.median()),
        p5_pct=float(values.quantile(0.05)),
        p95_pct=float(values.quantile(0.95)),
        pct_samples_below_10=float((values < 10).mean() * 100),
        pct_samples_above_80=float((values > 80).mean() * 100),
        samples=int(values.size),
        metric_names=tuple(str(name) for name in metrics["metric_name"].unique()),
    )


def _window_ms(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return 0.0
    return (metrics["timestamp"].max() - metrics["timestamp"].min()) / 1e6


def _histogram_quantile(histogram: list[tuple[float, int]], total: int, q: float) -> float:
    if total <= 0:
        return 0.0
    target = q * (total - 1)
    cumulative = 0
    for value, count in histogram:
        cumulative += count
        if cumulative - 1 >= target:
            return float(value)
    return float(histogram[-1][0]) if histogram else 0.0


def _summarize_full_trace_sql(conn: sqlite3.Connection) -> list[dict[str, object]]:
    metric_rows = conn.execute(
        """
        SELECT metricId, metricName
        FROM TARGET_INFO_GPU_METRICS i
        WHERE
        """
        + _relevant_metric_clause("i")
        + """
        ORDER BY metricId
        """
    ).fetchall()

    rows: list[dict[str, object]] = []
    for metric_id, metric_name in metric_rows:
        stats = conn.execute(
            """
            SELECT COUNT(*), AVG(value), MAX(value), MIN(timestamp), MAX(timestamp)
            FROM GPU_METRICS
            WHERE metricId = ?
            """,
            (metric_id,),
        ).fetchone()
        samples = int(stats[0] or 0)
        if samples == 0:
            continue

        histogram = [
            (float(value), int(count))
            for value, count in conn.execute(
                """
                SELECT value, COUNT(*)
                FROM GPU_METRICS
                WHERE metricId = ?
                GROUP BY value
                ORDER BY value
                """,
                (metric_id,),
            )
        ]
        p95 = _histogram_quantile(histogram, samples, 0.95)
        rows.append(
            {
                "metric_name": str(metric_name),
                "window": "full_trace",
                "window_ms": (float(stats[4]) - float(stats[3])) / 1e6,
                "samples": samples,
                "avg_pct": float(stats[1]),
                "p50_pct": _histogram_quantile(histogram, samples, 0.50),
                "p95_pct": p95,
                "max_pct": float(stats[2]),
                "headroom_vs_p95_pct": max(0.0, 100.0 - p95),
            }
        )
    return rows


def summarize_nsys(path: Path, measured_only: bool = True) -> list[dict[str, object]]:
    if not measured_only:
        conn = sqlite3.connect(path)
        try:
            available_tables = table_names(conn)
            if "GPU_METRICS" not in available_tables or "TARGET_INFO_GPU_METRICS" not in available_tables:
                return []
            return _summarize_full_trace_sql(conn)
        finally:
            conn.close()

    metrics, kernels, window = load_nsys_metrics(path, measured_only=measured_only)
    if metrics.empty:
        return []

    rows: list[dict[str, object]] = []
    for metric_name, group in metrics.groupby("metric_name"):
        metric_name_str = str(metric_name)
        values = group["value"].astype(float)
        if not (
            "DRAM" in metric_name_str
            or "SMs Active" in metric_name_str
            or "SM " in metric_name_str
            or "Tensor" in metric_name_str
            or "GR Active" in metric_name_str
            or "GPU Active" in metric_name_str
            or "GPU Utilization" in metric_name_str
        ):
            continue
        rows.append(
            {
                "metric_name": metric_name_str,
                "window": window,
                "window_ms": _window_ms(group),
                "samples": int(values.size),
                "avg_pct": float(values.mean()),
                "p50_pct": float(values.quantile(0.50)),
                "p95_pct": float(values.quantile(0.95)),
                "max_pct": float(values.max()),
                "headroom_vs_p95_pct": max(0.0, 100.0 - float(values.quantile(0.95))),
            }
        )

    if not kernels.empty:
        kernel_duration_ms = ((kernels["end"] - kernels["start"]).sum()) / 1e6
        rows.append(
            {
                "metric_name": "CUDA kernel duration sum",
                "window": window,
                "window_ms": _window_ms(metrics),
                "samples": int(len(kernels)),
                "avg_pct": kernel_duration_ms,
                "p50_pct": 0.0,
                "p95_pct": 0.0,
                "max_pct": 0.0,
                "headroom_vs_p95_pct": 0.0,
            }
        )

    return rows


def extract_dram_utilization(
    path: Path,
    measured_only: bool = False,
    *,
    trim_idle_edges: bool = False,
) -> DramUtilization:
    if trim_idle_edges:
        metrics, _, _ = load_nsys_metrics(path, measured_only=measured_only)
        metrics = _trim_to_active_dram_window(metrics, path)
        dram = _dram_metrics(metrics)
        values = dram["value"].astype(float)
        return DramUtilization(
            avg_pct=float(values.mean()),
            p95_pct=float(values.quantile(0.95)),
            samples=int(values.size),
            metric_names=tuple(str(name) for name in dram["metric_name"].unique()),
        )

    rows = summarize_nsys(path, measured_only=measured_only)
    dram_rows = [
        row
        for row in rows
        if "dram" in str(row["metric_name"]).lower() and "throughput" in str(row["metric_name"]).lower()
    ]
    if not dram_rows:
        dram_rows = [row for row in rows if "dram" in str(row["metric_name"]).lower()]
    if not dram_rows:
        raise ValueError(f"No DRAM utilization metric found in {path}")

    sample_count = sum(_as_int(row["samples"]) for row in dram_rows)
    if sample_count <= 0:
        raise ValueError(f"DRAM metrics in {path} contain no samples")

    def weighted(field: str) -> float:
        return sum(_as_float(row[field]) * _as_int(row["samples"]) for row in dram_rows) / sample_count

    return DramUtilization(
        avg_pct=weighted("avg_pct"),
        p95_pct=weighted("p95_pct"),
        samples=sample_count,
        metric_names=tuple(str(row["metric_name"]) for row in dram_rows),
    )


def extract_gpu_activity(
    path: Path,
    measured_only: bool = False,
    *,
    trim_idle_edges: bool = False,
) -> GpuActivityUtilization:
    """Extract GPU activity over the same optional active-DRAM window as DRAM stats."""
    metrics, _, _ = load_nsys_metrics(path, measured_only=measured_only)
    if metrics.empty:
        raise ValueError(f"No GPU metrics found in {path}")
    if trim_idle_edges:
        metrics = _trim_to_active_dram_window(metrics, path)

    sm_active = _metric_subset(metrics, "SMs Active")
    gr_active = metrics[
        metrics["metric_name"].str.contains(
            r"GR Active|GPU Active|GPU Utilization",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()
    gr_active_pct = gr_active[
        gr_active["metric_name"].str.contains(
            r"Throughput %|Utilization %",
            case=False,
            regex=True,
            na=False,
        )
    ]
    if not gr_active_pct.empty:
        gr_active = gr_active_pct
    else:
        # Cycle counts are not percentages and must not feed utilization stats.
        gr_active = gr_active[
            ~gr_active["metric_name"].str.contains(
                "Cycles Active",
                case=False,
                regex=False,
                na=False,
            )
        ]
    return GpuActivityUtilization(
        sm_active=_activity_stats(sm_active),
        gr_active=_activity_stats(gr_active),
    )


def write_summary_csv(rows: list[dict[str, object]], output: Path) -> None:
    write_dict_rows(rows, output)
    if rows:
        print(f"Wrote summary to {output}")


def _plot_metric_family(metrics: pd.DataFrame, contains: str, title: str, ax: plt.Axes) -> None:
    subset = _metric_subset(metrics, contains)
    if subset.empty:
        ax.text(0.5, 0.5, f"No {contains} metrics", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    t0 = subset["timestamp"].min()
    subset["time_ms"] = (subset["timestamp"] - t0) / 1e6
    for metric_name in subset["metric_name"].unique():
        metric_df = subset[subset["metric_name"] == metric_name]
        label = str(metric_name).replace(" [Throughput %]", "")
        ax.plot(metric_df["time_ms"], metric_df["value"], label=label, alpha=0.75)
    ax.set_xlabel("Time in measured request window (ms)")
    ax.set_ylabel("Percent of sustained peak")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_summary(rows: list[dict[str, object]], ax: plt.Axes) -> None:
    metric_rows = [row for row in rows if str(row["metric_name"]) != "CUDA kernel duration sum"]
    if not metric_rows:
        ax.text(0.5, 0.5, "No summary metrics", ha="center", va="center", transform=ax.transAxes)
        return

    labels = [str(row["metric_name"]).replace(" [Throughput %]", "") for row in metric_rows]
    p95 = [float(str(row["p95_pct"])) for row in metric_rows]
    ax.barh(labels, p95)
    ax.set_xlabel("p95 percent of sustained peak")
    ax.set_title("Measured-Window p95 Utilization")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.3)


def visualize_nsys(
    path: Path,
    output: Path | None = None,
    summary_output: Path | None = None,
    measured_only: bool = True,
) -> None:
    metrics, _, window = load_nsys_metrics(path, measured_only=measured_only)
    rows = summarize_nsys(path, measured_only=measured_only)

    if summary_output:
        write_summary_csv(rows, summary_output)

    if metrics.empty:
        print("No GPU metrics found")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    _plot_metric_family(metrics, "DRAM", "DRAM Bandwidth During vLLM Request Load", axes[0])
    _plot_metric_family(metrics, "SMs Active", "SM Activity During vLLM Request Load", axes[1])
    _plot_summary(rows, axes[2])
    fig.suptitle(f"vLLM Serving Resource Utilization (NSYS, window={window})", fontsize=12)

    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
