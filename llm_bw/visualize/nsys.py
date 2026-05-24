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
        kernels = filter_by_nvtx(kernels, nvtx, r":case$", r":warmup$")
        metrics = filter_by_nvtx(metrics, nvtx, r":case$", r":warmup$", start_col="timestamp")

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
