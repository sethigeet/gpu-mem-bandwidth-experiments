import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ATTENTION_KERNEL_PATTERNS = [
    (r"pytorch_flash::flash_fwd", "sdpa_flash"),
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


def plot_kernel_count(df: pd.DataFrame, ax: plt.Axes) -> None:
    counts = df.groupby("kernel_type").size()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("Invocation Count")
    ax.set_title("Kernel Invocations")
    ax.tick_params(axis="x", rotation=45)


def visualize(df: pd.DataFrame, output: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_bandwidth(df, axes[0, 0])
    plot_dram_utilization(df, axes[0, 1])
    plot_sm_utilization(df, axes[1, 0])
    plot_kernel_count(df, axes[1, 1])
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
