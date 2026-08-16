import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gpu_benchmarks.model.visualize.common import format_config_title
from gpu_benchmarks.profiling.nsys import filter_by_nvtx, load_nvtx_ranges


def load_nsys_metrics(path: Path, exclude_warmup: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(path)
    nvtx = load_nvtx_ranges(conn)

    metrics = pd.read_sql_query(
        """
        SELECT m.timestamp, m.metricId, m.value, i.metricName as metric_name
        FROM GPU_METRICS m
        JOIN TARGET_INFO_GPU_METRICS i ON m.metricId = i.metricId
        ORDER BY m.timestamp
        """,
        conn,
    )
    kernels = pd.read_sql_query(
        """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """,
        conn,
    )

    if exclude_warmup:
        kernels = filter_by_nvtx(kernels, nvtx, r":case$", timestamp_column="start", exclude_pattern=r":warmup$")
        metrics = filter_by_nvtx(metrics, nvtx, r":case$", timestamp_column="timestamp", exclude_pattern=r":warmup$")

    conn.close()

    return metrics, kernels


def plot_bandwidth_timeline(metrics: pd.DataFrame, ax: plt.Axes) -> None:
    bw_metrics = metrics[metrics["metric_name"].str.contains("DRAM", na=False)]

    if bw_metrics.empty:
        ax.text(0.5, 0.5, "No DRAM metrics", ha="center", va="center", transform=ax.transAxes)
        return

    t0 = bw_metrics["timestamp"].min()
    bw_metrics = bw_metrics.copy()
    bw_metrics["time_ms"] = (bw_metrics["timestamp"] - t0) / 1e6

    for metric_name in bw_metrics["metric_name"].unique():
        mdf = bw_metrics[bw_metrics["metric_name"] == metric_name]
        label = metric_name.replace(" [Throughput %]", "").replace("DRAM ", "")
        ax.plot(mdf["time_ms"], mdf["value"], label=label, alpha=0.7)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Throughput (%)")
    ax.set_title("DRAM Bandwidth Over Decode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_sm_timeline(metrics: pd.DataFrame, ax: plt.Axes) -> None:
    sm_metrics = metrics[metrics["metric_name"].str.contains("SMs Active", na=False)]

    if sm_metrics.empty:
        ax.text(0.5, 0.5, "No SM metrics", ha="center", va="center", transform=ax.transAxes)
        return

    t0 = sm_metrics["timestamp"].min()
    sm_metrics = sm_metrics.copy()
    sm_metrics["time_ms"] = (sm_metrics["timestamp"] - t0) / 1e6

    for metric_name in sm_metrics["metric_name"].unique():
        mdf = sm_metrics[sm_metrics["metric_name"] == metric_name]
        label = metric_name.replace(" [Throughput %]", "")
        ax.plot(mdf["time_ms"], mdf["value"], label=label, alpha=0.7)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Throughput (%)")
    ax.set_title("SM Utilization Over Decode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def visualize_nsys(
    path: Path, output: Path | None = None, exclude_warmup: bool = True, config: dict | None = None
) -> None:
    metrics, _ = load_nsys_metrics(path, exclude_warmup=exclude_warmup)

    if metrics.empty:
        print("No metrics found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_bandwidth_timeline(metrics, axes[0])
    plot_sm_timeline(metrics, axes[1])

    title = "LLM Decode Resource Utilization Over Time (NSYS)"
    if config:
        subtitle = format_config_title(config)
        if subtitle:
            title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
