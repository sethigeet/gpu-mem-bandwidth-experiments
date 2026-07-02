from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MEASURED_RANGE_PATTERN = r"vllm_bw:serve:bench"
MAX_PLOT_METRIC_ROWS = 200_000


def _relevant_metric_clause(alias: str = "i") -> str:
    return f"""
    (
        {alias}.metricName LIKE '%DRAM%'
        OR {alias}.metricName LIKE 'SMs Active%'
        OR {alias}.metricName LIKE 'SM Issue%'
        OR {alias}.metricName LIKE 'Tensor Active%'
    )
    """


def _table_names(conn: sqlite3.Connection) -> set[str]:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    return set(tables["name"].tolist())


def load_nvtx_ranges(conn: sqlite3.Connection) -> pd.DataFrame:
    table_names = _table_names(conn)
    for table in ["NVTX_EVENTS", "NVTX_RANGES"]:
        if table not in table_names:
            continue
        try:
            df = pd.read_sql_query(
                f"""
                SELECT start, end, text as name
                FROM {table}
                WHERE end IS NOT NULL AND text IS NOT NULL AND text != ''
                ORDER BY start
                """,
                conn,
            )
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame(columns=["start", "end", "name"])


def _filter_by_nvtx(
    df: pd.DataFrame,
    nvtx: pd.DataFrame,
    include_pattern: str,
    start_col: str,
) -> tuple[pd.DataFrame, str]:
    include_ranges = nvtx[nvtx["name"].str.contains(include_pattern, regex=True, na=False)]
    if include_ranges.empty:
        return df.copy(), "full_trace"

    mask = pd.Series(False, index=df.index)
    for _, row in include_ranges.iterrows():
        mask |= (df[start_col] >= row["start"]) & (df[start_col] <= row["end"])
    return df[mask].copy(), "nvtx_measured_range"


def load_nsys_metrics(path: Path, measured_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    conn = sqlite3.connect(path)
    table_names = _table_names(conn)
    nvtx = load_nvtx_ranges(conn)

    if "GPU_METRICS" not in table_names or "TARGET_INFO_GPU_METRICS" not in table_names:
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
        SELECT m.timestamp, m.metricId, CAST(m.value AS REAL) as value, i.metricName as metric_name
        FROM GPU_METRICS m
        JOIN TARGET_INFO_GPU_METRICS i ON m.metricId = i.metricId
        WHERE
        """
        + _relevant_metric_clause("i")
        + """
          AND (m.rowid % ? = 0)
        ORDER BY m.timestamp
        """,
        conn,
        params=[stride],
    )

    kernels = pd.DataFrame(columns=["start", "end", "name"])
    if "CUPTI_ACTIVITY_KIND_KERNEL" in table_names and "StringIds" in table_names:
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
            table_names = _table_names(conn)
            if "GPU_METRICS" not in table_names or "TARGET_INFO_GPU_METRICS" not in table_names:
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


def write_summary_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
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
