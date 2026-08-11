from __future__ import annotations

import glob
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from component_bw.config import STAGES

NCU_METRICS = {
    "dram__bytes_read.sum": "bytes_read",
    "dram__bytes_write.sum": "bytes_write",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__time_duration.sum": "duration_ns",
}

KERNEL_PATTERNS = [
    (r"flash|fmha|attention|scaled_dot_product", "attention"),
    (r"gemm|matmul|cublas|cutlass", "linear"),
    (r"index|gather|take|scatter", "paged_gather"),
    (r"layer_norm|rms_norm|rmsnorm|norm", "normalization"),
    (r"embedding", "embedding"),
    (r"softmax", "softmax"),
    (r"silu|gelu|activation|elementwise|add|mul", "activation"),
    (r"copy|memcpy|fill|zero", "memory_op"),
]


def _stage_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {stage: idx for idx, stage in enumerate(STAGES)}
    columns = ["stage"]
    if "batch_size" in df:
        columns.append("batch_size")
    return df.sort_values(
        columns,
        key=lambda s: s.map(order).fillna(len(order)) if s.name == "stage" else s,
    ).reset_index(drop=True)


def _clean_ncu_lines(path: Path) -> str:
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(line for line in lines if line and not line.startswith("=="))


def _stage_from_path(path: Path) -> str:
    stem = path.stem
    stages = [str(stage) for stage in STAGES]
    stages.sort(key=lambda value: len(value), reverse=True)
    for stage in stages:
        if re.search(rf"(^|_){re.escape(stage)}($|_)", stem):
            return stage
    raise ValueError(f"could not infer component stage from {path}")


def _batch_size_from_path(path: Path) -> int | None:
    match = re.search(r"(?:^|_)b(?:atch)?(\d+)(?:_|$)", path.stem)
    return int(match.group(1)) if match else None


def classify_kernel(name: str) -> str:
    for pattern, label in KERNEL_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "other"


def load_ncu_csv(path: Path) -> pd.DataFrame:
    text = _clean_ncu_lines(path)
    df = pd.read_csv(StringIO(text))
    if df.empty:
        return pd.DataFrame()

    df["Metric Value"] = (
        df["Metric Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    pivot = df.pivot_table(
        index=["ID", "Kernel Name"],
        columns="Metric Name",
        values="Metric Value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    stage = _stage_from_path(path)
    pivot["stage"] = stage
    batch_size = _batch_size_from_path(path)
    if batch_size is not None:
        pivot["batch_size"] = batch_size
    pivot["kernel_type"] = pivot["Kernel Name"].astype(str).apply(classify_kernel)
    for metric, column in NCU_METRICS.items():
        if metric in pivot:
            pivot[column] = pivot[metric].fillna(0.0)
        else:
            pivot[column] = 0.0
    pivot["total_bytes"] = pivot["bytes_read"] + pivot["bytes_write"]
    pivot["bandwidth_gb_s"] = pivot["total_bytes"] / pivot["duration_ns"].where(pivot["duration_ns"] > 0, pd.NA)
    return pivot


def load_ncu_glob(pattern: str) -> pd.DataFrame:
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    frames = []
    for path in paths:
        try:
            frames.append(load_ncu_csv(path))
        except Exception as exc:
            print(f"Skipping unreadable NCU CSV {path}: {exc}")
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _duration_weighted_average(df: pd.DataFrame, column: str) -> float:
    duration = df["duration_ns"].sum()
    if duration <= 0:
        return 0.0
    return float((df[column] * df["duration_ns"]).sum() / duration)


def summarize_ncu(kernels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if kernels.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = ["stage", *(["batch_size"] if "batch_size" in kernels else [])]
    for key, sdf in kernels.groupby(group_columns, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        group = dict(zip(group_columns, values, strict=True))
        duration_ns = float(sdf["duration_ns"].sum())
        total_bytes = float(sdf["total_bytes"].sum())
        rows.append(
            {
                **group,
                "kernel_count": int(len(sdf)),
                "ncu_duration_ms": duration_ns / 1e6,
                "ncu_total_bytes": total_bytes,
                "ncu_bandwidth_gb_s": total_bytes / duration_ns if duration_ns > 0 else 0.0,
                "ncu_dram_pct_weighted": _duration_weighted_average(sdf, "dram_pct"),
                "ncu_sm_pct_weighted": _duration_weighted_average(sdf, "sm_pct"),
                "ncu_dram_pct_max": float(sdf["dram_pct"].max()),
                "ncu_sm_pct_max": float(sdf["sm_pct"].max()),
            }
        )

    type_rows: list[dict[str, object]] = []
    type_group_columns = [*group_columns, "kernel_type"]
    for key, sdf in kernels.groupby(type_group_columns, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        group = dict(zip(type_group_columns, values, strict=True))
        duration_ns = float(sdf["duration_ns"].sum())
        total_bytes = float(sdf["total_bytes"].sum())
        type_rows.append(
            {
                **group,
                "kernel_count": int(len(sdf)),
                "ncu_duration_ms": duration_ns / 1e6,
                "ncu_total_bytes": total_bytes,
                "ncu_bandwidth_gb_s": total_bytes / duration_ns if duration_ns > 0 else 0.0,
                "ncu_dram_pct_weighted": _duration_weighted_average(sdf, "dram_pct"),
                "ncu_sm_pct_weighted": _duration_weighted_average(sdf, "sm_pct"),
            }
        )

    return _stage_order(pd.DataFrame(rows)), _stage_order(pd.DataFrame(type_rows))


def merge_throughput_and_ncu(throughput_csv: Path, ncu_summary: pd.DataFrame) -> pd.DataFrame:
    throughput = pd.read_csv(throughput_csv)
    throughput = _stage_order(throughput)
    if ncu_summary.empty:
        return throughput
    merge_columns = ["stage"]
    if "batch_size" in ncu_summary:
        merge_columns.append("batch_size")
    return throughput.merge(ncu_summary, on=merge_columns, how="left")


def plot_report(merged: pd.DataFrame, type_summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if merged["batch_size"].nunique() > 1:
        has_sm = "ncu_sm_pct_weighted" in merged and merged["ncu_sm_pct_weighted"].fillna(0).gt(0).any()
        panel_count = 3 if has_sm else 2
        fig, axes = plt.subplots(1, panel_count, figsize=(7.5 * panel_count, 6))
        for stage, sdf in merged.groupby("stage", sort=False):
            sdf = sdf.sort_values("batch_size")
            axes[0].plot(sdf["batch_size"], sdf["throughput_toks_s"], marker="o", label=stage)
            axes[1].plot(sdf["batch_size"], sdf["ncu_dram_pct_weighted"], marker="o", label=stage)
            if has_sm:
                axes[2].plot(sdf["batch_size"], sdf["ncu_sm_pct_weighted"], marker="o", label=stage)
        axes[0].set_title("Decode Throughput vs Batch Size")
        axes[0].set_ylabel("tokens/s")
        axes[0].set_yscale("log")
        axes[1].set_title("DRAM Bandwidth Utilization vs Batch Size")
        axes[1].set_ylabel("% of peak sustained")
        if has_sm:
            axes[2].set_title("SM Utilization vs Batch Size")
            axes[2].set_ylabel("% of peak sustained")
        for axis in axes:
            axis.set_xlabel("batch size")
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted(merged["batch_size"].unique()))
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axis.tick_params(axis="x", rotation=45)
            axis.grid(alpha=0.25)
        axes[-1].legend(fontsize=8, ncol=2)
        fig.suptitle("Phi-3-mini-shaped 10K Shared-prefix Component Batch Sweep", fontsize=12)
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close(fig)
        print(f"Saved figure to {output}")
        return

    labels = merged["stage"].astype(str).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(labels, merged["throughput_toks_s"])
    axes[0, 0].set_title("Decode Throughput")
    axes[0, 0].set_ylabel("tokens/s")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(labels, merged["ncu_bandwidth_gb_s"])
    axes[0, 1].set_title("NCU Effective DRAM Bandwidth")
    axes[0, 1].set_ylabel("GB/s")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(labels, merged["ncu_dram_pct_weighted"], label="DRAM")
    axes[1, 0].bar(labels, merged["ncu_sm_pct_weighted"], alpha=0.65, label="SM")
    axes[1, 0].set_title("NCU Weighted Utilization")
    axes[1, 0].set_ylabel("% of peak sustained")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend()

    if not type_summary.empty:
        pivot = type_summary.pivot_table(
            index="stage",
            columns="kernel_type",
            values="ncu_duration_ms",
            aggfunc="sum",
            fill_value=0.0,
        )
        pivot = pivot.reindex(labels).fillna(0.0)
        pivot.plot(kind="bar", stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title("NCU Kernel Time Breakdown")
        axes[1, 1].set_ylabel("ms")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No NCU kernel type data", ha="center", va="center")

    fig.suptitle("Phi-3-mini-shaped 10K Shared-prefix Component Ladder", fontsize=12)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {output}")


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    display = df[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display = display.fillna("")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_report(
    merged: pd.DataFrame,
    ncu_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    output: Path,
    plot_path: Path,
    throughput_csv: Path,
    ncu_glob: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    is_batch_sweep = merged["batch_size"].nunique() > 1
    missing_rows = merged[merged["ncu_dram_pct_weighted"].isna()]
    missing_ncu = [
        f"{row.stage} (batch {row.batch_size})" if is_batch_sweep else str(row.stage)
        for row in missing_rows.itertuples()
    ]
    missing_note = ""
    if missing_ncu:
        missing_note = (
            "\n- Missing NCU counters: "
            + ", ".join(f"`{stage}`" for stage in missing_ncu)
            + ". These stages have throughput rows but no usable NCU CSV in this report.\n"
        )
    title = (
        "Component Bandwidth 10K Shared-prefix Batch Sweep"
        if is_batch_sweep
        else "Component Bandwidth 10K Shared-prefix Report"
    )
    has_sm = "ncu_sm_pct_weighted" in merged and merged["ncu_sm_pct_weighted"].fillna(0).gt(0).any()
    sweep_counters = ["`dram__throughput.avg.pct_of_peak_sustained_elapsed`"]
    if has_sm:
        sweep_counters.append("`sm__throughput.avg.pct_of_peak_sustained_elapsed`")
    sweep_counters.append("`gpu__time_duration.sum`")
    counter_note = (
        f"- Counters: {', '.join(sweep_counters)}.\n"
        "- Aggregation: DRAM/SM utilization is a kernel-duration-weighted average for the measured decode token."
        if is_batch_sweep
        else "- Counters: `dram__bytes_read.sum`, `dram__bytes_write.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, and `gpu__time_duration.sum`.\n"
        "- Aggregation: effective GB/s is total read+write bytes divided by total profiled kernel duration. DRAM/SM percentages are duration-weighted averages across kernels in each stage."
    )
    stage_columns = ["stage", "batch_size", "throughput_toks_s", "per_token_ms"]
    if not is_batch_sweep:
        stage_columns.append("ncu_bandwidth_gb_s")
    stage_columns.extend(["ncu_dram_pct_weighted"])
    if has_sm:
        stage_columns.append("ncu_sm_pct_weighted")
    stage_columns.append("kernel_count")
    type_columns = [
        "stage",
        *(["batch_size"] if "batch_size" in type_summary else []),
        "kernel_type",
        "kernel_count",
        "ncu_duration_ms",
    ]
    if not is_batch_sweep:
        type_columns.append("ncu_bandwidth_gb_s")
    type_columns.append("ncu_dram_pct_weighted")
    if has_sm:
        type_columns.append("ncu_sm_pct_weighted")

    report = f"""# {title}

## Methodology

- Throughput source: `{throughput_csv}` from the component batch sweep. This measures wall-clock decode throughput with CUDA synchronization around each measured stage.
- Bandwidth source: per-stage Nsight Compute CSVs matched by `{ncu_glob}`. NCU profiles one measured decode token per stage with warmup excluded through `component_bw:<stage>:iter` NVTX filtering.
{counter_note}
{missing_note}

## Results

![Component bandwidth summary]({plot_path.name})

### Stage Summary

{_markdown_table(merged, stage_columns)}

### NCU Kernel-type Time Breakdown

{_markdown_table(type_summary, type_columns)}

## Artifacts

- Throughput CSV: `{throughput_csv}`
- NCU stage summary CSV: `{ncu_summary_path(output)}`
- NCU kernel-type summary CSV: `{ncu_type_summary_path(output)}`
- Plot: `{plot_path}`
"""
    output.write_text(report)
    print(f"Wrote report to {output}")


def ncu_summary_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_ncu_summary.csv")


def ncu_type_summary_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_ncu_kernel_types.csv")


def _merge_sm_summary(primary: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    if sm.empty:
        return primary
    keys = ["stage", *(["batch_size"] if "batch_size" in sm else [])]
    if "kernel_type" in primary and "kernel_type" in sm:
        keys.append("kernel_type")
    metrics = [column for column in ["ncu_sm_pct_weighted", "ncu_sm_pct_max"] if column in sm]
    sm_columns = [*keys, *metrics]
    base = primary.drop(columns=metrics, errors="ignore")
    return base.merge(sm[sm_columns], on=keys, how="left")


def generate_report(
    throughput_csv: Path,
    ncu_pattern: str,
    output: Path,
    plot_output: Path | None = None,
    sm_ncu_pattern: str | None = None,
) -> None:
    kernels = load_ncu_glob(ncu_pattern)
    ncu_summary, type_summary = summarize_ncu(kernels)
    if sm_ncu_pattern:
        sm_kernels = load_ncu_glob(sm_ncu_pattern)
        sm_summary, sm_type_summary = summarize_ncu(sm_kernels)
        ncu_summary = _merge_sm_summary(ncu_summary, sm_summary)
        type_summary = _merge_sm_summary(type_summary, sm_type_summary)
    merged = merge_throughput_and_ncu(throughput_csv, ncu_summary)

    summary_path = ncu_summary_path(output)
    type_path = ncu_type_summary_path(output)
    if not ncu_summary.empty:
        ncu_summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
    if not type_summary.empty:
        type_summary.to_csv(type_path, index=False)
        print(f"Wrote {type_path}")

    plot_path = plot_output or output.with_suffix(".png")
    plot_report(merged, type_summary, plot_path)
    write_report(merged, ncu_summary, type_summary, output, plot_path, throughput_csv, ncu_pattern)
