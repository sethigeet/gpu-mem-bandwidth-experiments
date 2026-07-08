from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from component_bw.config import STAGES

MEASURED_RANGE_PATTERN = r"component_bw:.*:case$"
WARMUP_RANGE_PATTERN = r"component_bw:.*:warmup$"
MAX_PLOT_METRIC_ROWS = 200_000


def visualize_summary(path: Path, output: Path | None = None) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print("No rows in component CSV")
        return

    stage_order = {stage: idx for idx, stage in enumerate(STAGES)}
    df = df.sort_values("stage", key=lambda s: s.map(stage_order).fillna(len(stage_order)))
    labels = [
        f"{stage}\n{layout}" for stage, layout in zip(df["stage"].astype(str), df["layout"].astype(str), strict=True)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].bar(labels, df["throughput_toks_s"])
    axes[0, 0].set_title("Decode Throughput")
    axes[0, 0].set_ylabel("tokens/s")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(labels, df["per_token_ms"])
    axes[0, 1].set_title("Latency Per Output Token")
    axes[0, 1].set_ylabel("ms/token")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(labels, df["physical_kv_read_bytes_per_output_token"] / 1024**2)
    axes[1, 0].set_title("Estimated Physical KV Bytes")
    axes[1, 0].set_ylabel("MiB/output token")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(labels, df["estimated_launches_per_token"])
    axes[1, 1].set_title("Estimated Kernel Launches")
    axes[1, 1].set_ylabel("launches/token")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Component Decode Ladder | model={df['model'].iloc[0]} | "
        f"prefix={df['prefix_len'].iloc[0]} | dtype={df['dtype'].iloc[0]}",
        fontsize=12,
    )
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    return set(tables["name"].tolist())


def _load_nvtx_ranges(conn: sqlite3.Connection) -> pd.DataFrame:
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
    exclude_pattern: str | None = None,
) -> pd.DataFrame:
    include_ranges = nvtx[nvtx["name"].str.contains(include_pattern, regex=True, na=False)]
    if include_ranges.empty:
        return df.copy()

    mask = pd.Series(False, index=df.index)
    for _, row in include_ranges.iterrows():
        mask |= (df[start_col] >= row["start"]) & (df[start_col] <= row["end"])

    if exclude_pattern:
        exclude_ranges = nvtx[nvtx["name"].str.contains(exclude_pattern, regex=True, na=False)]
        for _, row in exclude_ranges.iterrows():
            mask &= ~((df[start_col] >= row["start"]) & (df[start_col] <= row["end"]))
    return df[mask].copy()


def _align_metrics_to_nvtx(metrics: pd.DataFrame, nvtx: pd.DataFrame) -> pd.DataFrame:
    include_ranges = nvtx[nvtx["name"].str.contains(MEASURED_RANGE_PATTERN, regex=True, na=False)]
    if metrics.empty or include_ranges.empty:
        return metrics

    metric_min = float(metrics["timestamp"].min())
    metric_max = float(metrics["timestamp"].max())
    nvtx_min = float(include_ranges["start"].min())
    nvtx_max = float(include_ranges["end"].max())
    metric_span = metric_max - metric_min
    nvtx_span = nvtx_max - nvtx_min
    if metric_span <= 0 or nvtx_span <= 0:
        return metrics

    aligned = metrics.copy()
    aligned["timestamp"] = ((aligned["timestamp"] - metric_min) * (nvtx_span / metric_span) + nvtx_min).astype("int64")
    return aligned


def _load_nsys_metrics(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        table_names = _table_names(conn)
        if "GPU_METRICS" not in table_names or "TARGET_INFO_GPU_METRICS" not in table_names:
            return pd.DataFrame(columns=["timestamp", "metric_name", "value"])
        nvtx = _load_nvtx_ranges(conn)
        relevant_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM GPU_METRICS m
            JOIN TARGET_INFO_GPU_METRICS i ON m.metricId = i.metricId
            WHERE
                i.metricName LIKE '%DRAM%'
                OR i.metricName LIKE 'SMs Active%'
                OR i.metricName LIKE 'SM Issue%'
                OR i.metricName LIKE 'Tensor Active%'
            """
        ).fetchone()[0]
        stride = max(1, int(relevant_count) // MAX_PLOT_METRIC_ROWS)
        metrics = pd.read_sql_query(
            """
            SELECT m.timestamp, CAST(m.value AS REAL) as value, i.metricName as metric_name
            FROM GPU_METRICS m
            JOIN TARGET_INFO_GPU_METRICS i ON m.metricId = i.metricId
            WHERE
                (
                i.metricName LIKE '%DRAM%'
                OR i.metricName LIKE 'SMs Active%'
                OR i.metricName LIKE 'SM Issue%'
                OR i.metricName LIKE 'Tensor Active%'
                )
                AND (m.rowid % ? = 0)
            ORDER BY m.timestamp
            """,
            conn,
            params=[stride],
        )
        filtered = _filter_by_nvtx(metrics, nvtx, MEASURED_RANGE_PATTERN, "timestamp", WARMUP_RANGE_PATTERN)
        if not filtered.empty:
            return filtered
        aligned = _align_metrics_to_nvtx(metrics, nvtx)
        return _filter_by_nvtx(aligned, nvtx, MEASURED_RANGE_PATTERN, "timestamp", WARMUP_RANGE_PATTERN)
    finally:
        conn.close()


def _metric_subset(metrics: pd.DataFrame, contains: str) -> pd.DataFrame:
    return metrics[metrics["metric_name"].str.contains(contains, case=False, na=False)].copy()


def _plot_metric(metrics: pd.DataFrame, contains: str, title: str, ax: plt.Axes) -> None:
    subset = _metric_subset(metrics, contains)
    if subset.empty:
        ax.text(0.5, 0.5, f"No {contains} metrics", ha="center", va="center", transform=ax.transAxes)
        return
    t0 = subset["timestamp"].min()
    subset["time_ms"] = (subset["timestamp"] - t0) / 1e6
    for metric_name, mdf in subset.groupby("metric_name"):
        label = str(metric_name).replace(" [Throughput %]", "")
        ax.plot(mdf["time_ms"], mdf["value"], label=label, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def summarize_nsys(path: Path) -> list[dict[str, object]]:
    metrics = _load_nsys_metrics(path)
    if metrics.empty:
        return []
    window_ms = (metrics["timestamp"].max() - metrics["timestamp"].min()) / 1e6
    rows: list[dict[str, object]] = []
    for metric_name, mdf in metrics.groupby("metric_name"):
        values = mdf["value"].astype(float)
        p95 = float(values.quantile(0.95))
        rows.append(
            {
                "metric_name": metric_name,
                "window": "component_bw_case_excluding_warmup",
                "window_ms": window_ms,
                "samples": int(values.count()),
                "avg_pct": float(values.mean()),
                "p50_pct": float(values.quantile(0.50)),
                "p95_pct": p95,
                "max_pct": float(values.max()),
                "headroom_vs_p95_pct": max(0.0, 100.0 - p95),
            }
        )
    return rows


def _write_summary(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary to {output}")


def visualize_nsys(path: Path, output: Path | None = None, summary_output: Path | None = None) -> None:
    metrics = _load_nsys_metrics(path)
    if metrics.empty:
        print("No component_bw GPU metrics found")
        return

    rows = summarize_nsys(path)
    if summary_output:
        _write_summary(rows, summary_output)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    _plot_metric(metrics, "DRAM", "DRAM Bandwidth Over Component Decode", axes[0])
    _plot_metric(metrics, "SMs Active", "SM Activity Over Component Decode", axes[1])
    fig.suptitle("Component Decode Resource Utilization (NSYS)", fontsize=12)
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
