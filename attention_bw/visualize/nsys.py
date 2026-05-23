import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_nvtx_ranges(conn: sqlite3.Connection) -> pd.DataFrame:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    table_names = tables["name"].tolist()

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


def filter_by_nvtx(
    df: pd.DataFrame,
    nvtx: pd.DataFrame,
    include_pattern: str,
    exclude_pattern: str | None = None,
    start_col: str = "start",
) -> pd.DataFrame:
    include_ranges = nvtx[nvtx["name"].str.contains(include_pattern, regex=True)]
    if include_ranges.empty:
        return df

    mask = pd.Series(False, index=df.index)
    for _, row in include_ranges.iterrows():
        mask |= (df[start_col] >= row["start"]) & (df[start_col] <= row["end"])

    if exclude_pattern:
        exclude_ranges = nvtx[nvtx["name"].str.contains(exclude_pattern, regex=True)]
        for _, row in exclude_ranges.iterrows():
            mask &= ~((df[start_col] >= row["start"]) & (df[start_col] <= row["end"]))

    return df[mask].copy()


def classify_by_nvtx(df: pd.DataFrame, nvtx: pd.DataFrame, start_col: str = "start") -> pd.Series:
    case_ranges = nvtx[nvtx["name"].str.endswith(":case")]
    kernel_types = pd.Series(index=df.index, dtype="object")
    for _, row in case_ranges.iterrows():
        mask = (df[start_col] >= row["start"]) & (df[start_col] <= row["end"])
        kernel_type = row["name"].split(":")[1]
        kernel_types.loc[mask] = kernel_type
    return kernel_types


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

    kernels["kernel_type"] = classify_by_nvtx(kernels, nvtx)
    metrics["kernel_type"] = classify_by_nvtx(metrics, nvtx, start_col="timestamp")

    if exclude_warmup:
        kernels = filter_by_nvtx(kernels, nvtx, r":case$", r":warmup$")
        metrics = filter_by_nvtx(metrics, nvtx, r":case$", r":warmup$", start_col="timestamp")

    conn.close()

    return metrics, kernels


def filter_metrics_by_kernel(metrics: pd.DataFrame, kernel_type: str) -> pd.DataFrame:
    return metrics[metrics["kernel_type"] == kernel_type].copy()


def plot_kernel_bandwidth(metrics: pd.DataFrame, kernel_type: str, ax: plt.Axes) -> None:
    filtered = filter_metrics_by_kernel(metrics, kernel_type)
    bw_metrics = filtered[filtered["metric_name"].str.contains("DRAM", na=False)]

    if bw_metrics.empty:
        ax.text(0.5, 0.5, f"No data for {kernel_type}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{kernel_type}")
        return

    t0 = bw_metrics["timestamp"].min()
    bw_metrics["time_us"] = (bw_metrics["timestamp"] - t0) / 1e3

    for metric_name in bw_metrics["metric_name"].unique():
        mdf = bw_metrics[bw_metrics["metric_name"] == metric_name]
        label = metric_name.replace(" [Throughput %]", "").replace("DRAM ", "")
        ax.plot(mdf["time_us"], mdf["value"], label=label, alpha=0.7)

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Throughput (%)")
    ax.set_title(f"{kernel_type}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_kernel_sm(metrics: pd.DataFrame, kernel_type: str, ax: plt.Axes) -> None:
    filtered = filter_metrics_by_kernel(metrics, kernel_type)
    sm_metrics = filtered[filtered["metric_name"].str.contains("SMs Active", na=False)]

    if sm_metrics.empty:
        ax.text(0.5, 0.5, f"No data for {kernel_type}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{kernel_type}")
        return

    t0 = sm_metrics["timestamp"].min()
    sm_metrics["time_us"] = (sm_metrics["timestamp"] - t0) / 1e3

    for metric_name in sm_metrics["metric_name"].unique():
        mdf = sm_metrics[sm_metrics["metric_name"] == metric_name]
        label = metric_name.replace(" [Throughput %]", "")
        ax.plot(mdf["time_us"], mdf["value"], label=label, alpha=0.7)

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Throughput (%)")
    ax.set_title(f"{kernel_type}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def visualize_nsys(path: Path, output: Path | None = None, exclude_warmup: bool = True) -> None:
    metrics, _ = load_nsys_metrics(path, exclude_warmup=exclude_warmup)
    kernel_types = [kt for kt in metrics["kernel_type"].dropna().unique()]

    if not kernel_types:
        print("No attention kernels found")
        return

    fig, axes = plt.subplots(2, len(kernel_types), figsize=(5 * len(kernel_types), 8))
    if len(kernel_types) == 1:
        axes = axes.reshape(-1, 1)

    for i, kernel_type in enumerate(kernel_types):
        plot_kernel_bandwidth(metrics, kernel_type, axes[0, i])
        plot_kernel_sm(metrics, kernel_type, axes[1, i])

    fig.suptitle("DRAM Bandwidth (top) and SM Utilization (bottom) by Kernel Type", fontsize=12)
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
