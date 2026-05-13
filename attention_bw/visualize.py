import re
import sqlite3
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ATTENTION_KERNEL_PATTERNS = [
    (r"flash_fwd", "sdpa_flash"),
    (r"fmha_cutlass", "sdpa_mem_efficient"),
    (r"softmax_warp_forward", "sdpa_math"),
    (r"cunn_SoftMax", "sdpa_math"),
]


def classify_kernel(name: str) -> str | None:
    for pattern, label in ATTENTION_KERNEL_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return None


def load_results(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    clean_lines = [line for line in lines if not line.startswith("==")]
    df = pd.read_csv(StringIO("\n".join(clean_lines)))
    df["Metric Value"] = df["Metric Value"].astype(str).str.replace(",", "").astype(float)
    pivoted = df.pivot_table(
        index=["ID", "Kernel Name"],
        columns="Metric Name",
        values="Metric Value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    pivoted["kernel_type"] = pivoted["Kernel Name"].apply(classify_kernel)
    pivoted = pivoted[pivoted["kernel_type"].notna()]
    return pivoted


def plot_bandwidth(df: pd.DataFrame, ax: plt.Axes) -> None:
    grouped = df.groupby("kernel_type").agg(
        bytes_read=("dram__bytes_read.sum", "sum"),
        bytes_write=("dram__bytes_write.sum", "sum"),
        duration_ns=("gpu__time_duration.sum", "sum"),
    )
    grouped["total_bytes"] = grouped["bytes_read"] + grouped["bytes_write"]
    grouped["bandwidth_gb_s"] = grouped["total_bytes"] / grouped["duration_ns"]  # bytes/ns = GB/s
    grouped["bandwidth_gb_s"].plot(kind="bar", ax=ax)
    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Measured Memory Bandwidth")
    ax.tick_params(axis="x", rotation=45)


def plot_dram_utilization(df: pd.DataFrame, ax: plt.Axes) -> None:
    col = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    grouped = df.groupby("kernel_type")[col].mean()
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("DRAM Utilization (%)")
    ax.set_title("Avg DRAM Throughput (% of Peak)")
    ax.tick_params(axis="x", rotation=45)


def plot_sm_utilization(df: pd.DataFrame, ax: plt.Axes) -> None:
    col = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    grouped = df.groupby("kernel_type")[col].mean()
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("SM Utilization (%)")
    ax.set_title("Avg SM Throughput (% of Peak)")
    ax.tick_params(axis="x", rotation=45)


def visualize(df: pd.DataFrame, output: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_bandwidth(df, axes[0])
    plot_dram_utilization(df, axes[1])
    plot_sm_utilization(df, axes[2])
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def load_nsys_metrics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(path)
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
    conn.close()

    kernels["kernel_type"] = kernels["name"].apply(classify_kernel)

    return metrics, kernels


def filter_metrics_by_kernel(
    metrics: pd.DataFrame, kernels: pd.DataFrame, kernel_type: str
) -> pd.DataFrame:
    type_kernels = kernels[kernels["kernel_type"] == kernel_type]
    if type_kernels.empty:
        return pd.DataFrame()

    mask = pd.Series(False, index=metrics.index)
    for _, row in type_kernels.iterrows():
        mask |= (metrics["timestamp"] >= row["start"]) & (metrics["timestamp"] <= row["end"])
    return metrics[mask].copy()


def plot_kernel_bandwidth(
    metrics: pd.DataFrame, kernels: pd.DataFrame, kernel_type: str, ax: plt.Axes
) -> None:
    filtered = filter_metrics_by_kernel(metrics, kernels, kernel_type)
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


def plot_kernel_sm(
    metrics: pd.DataFrame, kernels: pd.DataFrame, kernel_type: str, ax: plt.Axes
) -> None:
    filtered = filter_metrics_by_kernel(metrics, kernels, kernel_type)
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


def visualize_nsys(path: Path, output: Path | None = None) -> None:
    metrics, kernels = load_nsys_metrics(path)
    kernel_types = [kt for kt in ["sdpa_math", "sdpa_mem_efficient", "sdpa_flash"]
                    if kt in kernels["kernel_type"].values]

    if not kernel_types:
        print("No attention kernels found")
        return

    fig, axes = plt.subplots(2, len(kernel_types), figsize=(5 * len(kernel_types), 8))
    if len(kernel_types) == 1:
        axes = axes.reshape(-1, 1)

    for i, kernel_type in enumerate(kernel_types):
        plot_kernel_bandwidth(metrics, kernels, kernel_type, axes[0, i])
        plot_kernel_sm(metrics, kernels, kernel_type, axes[1, i])

    fig.suptitle("DRAM Bandwidth (top) and SM Utilization (bottom) by Kernel Type", fontsize=12)
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
