import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LLM_KERNEL_PATTERNS = [
    (r"flash_fwd|fmha|attention", "attention"),
    (r"gemm|matmul|cublas", "linear"),
    (r"layer_norm|rms_norm|rmsnorm", "normalization"),
    (r"embedding", "embedding"),
    (r"softmax", "softmax"),
    (r"elementwise|add|mul|gelu|silu", "activation"),
]


def classify_llm_kernel(name: str) -> str | None:
    for pattern, label in LLM_KERNEL_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "other"


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
    pivoted["layer_type"] = pivoted["Kernel Name"].apply(classify_llm_kernel)
    return pivoted


def plot_time_breakdown(df: pd.DataFrame, ax: plt.Axes) -> None:
    grouped = df.groupby("layer_type")["gpu__time_duration.sum"].sum()
    grouped = grouped.sort_values(ascending=False)
    grouped.plot(kind="pie", ax=ax, autopct="%1.1f%%")
    ax.set_ylabel("")
    ax.set_title("Time Breakdown by Layer Type")


def plot_bandwidth(df: pd.DataFrame, ax: plt.Axes) -> None:
    grouped = df.groupby("layer_type").agg(
        bytes_read=("dram__bytes_read.sum", "sum"),
        bytes_write=("dram__bytes_write.sum", "sum"),
        duration_ns=("gpu__time_duration.sum", "sum"),
    )
    grouped["total_bytes"] = grouped["bytes_read"] + grouped["bytes_write"]
    grouped["bandwidth_gb_s"] = grouped["total_bytes"] / grouped["duration_ns"]
    grouped["bandwidth_gb_s"].sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_xlabel("Layer Type")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Memory Bandwidth by Layer Type")
    ax.tick_params(axis="x", rotation=45)


def plot_dram_utilization(df: pd.DataFrame, ax: plt.Axes) -> None:
    col = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    grouped = df.groupby("layer_type")[col].mean().sort_values(ascending=False)
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Layer Type")
    ax.set_ylabel("DRAM Utilization (%)")
    ax.set_title("Avg DRAM Throughput (% of Peak)")
    ax.tick_params(axis="x", rotation=45)


def plot_sm_utilization(df: pd.DataFrame, ax: plt.Axes) -> None:
    col = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    grouped = df.groupby("layer_type")[col].mean().sort_values(ascending=False)
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Layer Type")
    ax.set_ylabel("SM Utilization (%)")
    ax.set_title("Avg SM Throughput (% of Peak)")
    ax.tick_params(axis="x", rotation=45)


def format_config_title(config: dict) -> str:
    if not config:
        return ""
    parts = []
    if "model" in config:
        parts.append(config["model"])
    if "dtype" in config:
        parts.append(config["dtype"])
    if "attention" in config:
        parts.append(f"attn={config['attention']}")
    if "prompt_length" in config:
        parts.append(f"prompt={config['prompt_length']}")
    if "batch_size" in config:
        parts.append(f"bs={config['batch_size']}")
    return " | ".join(parts)


def visualize_ncu(path: Path, output: Path | None = None, config: dict | None = None) -> None:
    df = load_results(path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_time_breakdown(df, axes[0, 0])
    plot_bandwidth(df, axes[0, 1])
    plot_dram_utilization(df, axes[1, 0])
    plot_sm_utilization(df, axes[1, 1])

    title = "LLM Decode Resource Utilization (NCU)"
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
