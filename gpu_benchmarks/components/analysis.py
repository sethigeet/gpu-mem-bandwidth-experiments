import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gpu_benchmarks.components.config import STAGES
from gpu_benchmarks.profiling.ncu import load_ncu_metrics

NCU_METRICS = {
    "dram__bytes_read.sum": "bytes_read",
    "dram__bytes_write.sum": "bytes_write",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__time_duration.sum": "duration_ns",
    "dram__sectors_read.sum": "dram_read_sectors",
    "dram__sectors_write.sum": "dram_write_sectors",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "lts__t_sectors_aperture_device_lookup_hit.sum": "l2_lookup_hit_sectors",
    "lts__t_sectors_aperture_device_lookup_miss.sum": "l2_lookup_miss_sectors",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": "sm_active_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed": "sm_instruction_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_pipe_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "fma_pipe_pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_elapsed": "issue_active_pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "long_scoreboard_stall_pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct": "lg_throttle_stall_pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct": "math_throttle_stall_pct",
}

WEIGHTED_METRICS = {
    "dram_pct": "ncu_dram_pct_weighted",
    "sm_pct": "ncu_sm_pct_weighted",
    "l1_hit_pct": "ncu_l1_hit_pct_weighted",
    "l2_hit_pct": "ncu_l2_hit_pct_weighted",
    "sm_active_pct": "ncu_sm_active_pct_weighted",
    "achieved_occupancy_pct": "ncu_achieved_occupancy_pct_weighted",
    "sm_instruction_pct": "ncu_sm_instruction_pct_weighted",
    "tensor_pipe_pct": "ncu_tensor_pipe_pct_weighted",
    "fma_pipe_pct": "ncu_fma_pipe_pct_weighted",
    "math_pipe_pct": "ncu_math_pipe_pct_weighted",
    "issue_active_pct": "ncu_issue_active_pct_weighted",
    "long_scoreboard_stall_pct": "ncu_long_scoreboard_stall_pct_weighted",
    "lg_throttle_stall_pct": "ncu_lg_throttle_stall_pct_weighted",
    "math_throttle_stall_pct": "ncu_math_throttle_stall_pct_weighted",
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
    pivot = load_ncu_metrics(path)
    if pivot.empty:
        return pd.DataFrame()

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
    pivot["hbm_bytes"] = 32 * (pivot["dram_read_sectors"] + pivot["dram_write_sectors"])
    pivot["l2_lookup_sectors"] = pivot["l2_lookup_hit_sectors"] + pivot["l2_lookup_miss_sectors"]
    pivot["l2_miss_pct"] = (
        100 * pivot["l2_lookup_miss_sectors"] / pivot["l2_lookup_sectors"].where(pivot["l2_lookup_sectors"] > 0, pd.NA)
    ).fillna(0.0)
    pivot["math_pipe_pct"] = pivot[["tensor_pipe_pct", "fma_pipe_pct"]].max(axis=1)
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


def _weighted_metric_summary(df: pd.DataFrame) -> dict[str, float]:
    return {output: _duration_weighted_average(df, source) for source, output in WEIGHTED_METRICS.items()}


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
                "ncu_hbm_bytes": float(sdf["hbm_bytes"].sum()),
                "ncu_l2_lookup_hit_sectors": float(sdf["l2_lookup_hit_sectors"].sum()),
                "ncu_l2_lookup_miss_sectors": float(sdf["l2_lookup_miss_sectors"].sum()),
                **_weighted_metric_summary(sdf),
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
                "ncu_hbm_bytes": float(sdf["hbm_bytes"].sum()),
                "ncu_l2_lookup_hit_sectors": float(sdf["l2_lookup_hit_sectors"].sum()),
                "ncu_l2_lookup_miss_sectors": float(sdf["l2_lookup_miss_sectors"].sum()),
                **_weighted_metric_summary(sdf),
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
    merged = throughput.merge(ncu_summary, on=merge_columns, how="left")
    if "ncu_hbm_bytes" in merged:
        merged["ncu_hbm_bytes_per_output_token"] = merged["ncu_hbm_bytes"] / merged["batch_size"]
    if "ncu_l2_lookup_miss_sectors" in merged:
        merged["ncu_l2_miss_sectors_per_output_token"] = merged["ncu_l2_lookup_miss_sectors"] / merged["batch_size"]
    return merged


def _workload_title(merged: pd.DataFrame, suffix: str) -> str:
    first = merged.iloc[0]
    model = str(first.get("model", "synthetic model")).replace("-", " ").title()
    prefix_len = int(first["prefix_len"])
    prefix = f"{prefix_len // 1000}K" if prefix_len >= 1000 and prefix_len % 1000 == 0 else str(prefix_len)
    sharing = str(first.get("prefix_sharing", first.get("layout", ""))).replace("_", " ").title()
    return f"{model}-shaped {prefix}-token {sharing}-prefix Component {suffix}"


def plot_analysis(merged: pd.DataFrame, type_summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if merged["batch_size"].nunique() > 1:
        panels = [
            ("throughput_toks_s", "Decode Throughput vs Batch Size", "tokens/s", True),
            ("ncu_dram_pct_weighted", "DRAM Bandwidth Utilization vs Batch Size", "% of peak sustained", False),
        ]
        if "ncu_math_pipe_pct_weighted" in merged and merged["ncu_math_pipe_pct_weighted"].fillna(0).gt(0).any():
            panels.append(
                (
                    "ncu_math_pipe_pct_weighted",
                    "Math-pipe Activity (Tensor/FMA) vs Batch Size",
                    "% of sustained peak",
                    False,
                )
            )
        elif "ncu_sm_pct_weighted" in merged and merged["ncu_sm_pct_weighted"].fillna(0).gt(0).any():
            panels.append(
                ("ncu_sm_pct_weighted", "Composite SM Throughput vs Batch Size", "% of sustained peak", False)
            )
        if "ncu_sm_active_pct_weighted" in merged and merged["ncu_sm_active_pct_weighted"].fillna(0).gt(0).any():
            panels.append(
                ("ncu_sm_active_pct_weighted", "SM Active Cycles vs Batch Size", "% of elapsed cycles", False)
            )

        if len(panels) == 4:
            fig, axes_grid = plt.subplots(2, 2, figsize=(15, 11))
            axes = list(axes_grid.flat)
        else:
            fig, axes_row = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6))
            axes = list(axes_row) if len(panels) > 1 else [axes_row]

        for stage, sdf in merged.groupby("stage", sort=False):
            sdf = sdf.sort_values("batch_size")
            for axis, (column, _title, _ylabel, _log_y) in zip(axes, panels, strict=True):
                axis.plot(sdf["batch_size"], sdf[column], marker="o", label=stage)
        for axis, (_column, title, ylabel, log_y) in zip(axes, panels, strict=True):
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            if log_y:
                axis.set_yscale("log")
            axis.set_xlabel("batch size")
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted(merged["batch_size"].unique()))
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axis.tick_params(axis="x", rotation=45)
            axis.grid(alpha=0.25)
        axes[-1].legend(fontsize=8, ncol=2)
        fig.suptitle(_workload_title(merged, "Batch Sweep"), fontsize=12)
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

    fig.suptitle(_workload_title(merged, "Ladder"), fontsize=12)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {output}")


def diagnostics_plot_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.stem}_diagnostics.png")


def plot_diagnostics(merged: pd.DataFrame, output: Path) -> None:
    required = [
        "ncu_l1_hit_pct_weighted",
        "ncu_l2_hit_pct_weighted",
        "ncu_hbm_bytes_per_output_token",
        "ncu_sm_pct_weighted",
        "ncu_sm_active_pct_weighted",
        "ncu_tensor_pipe_pct_weighted",
        "ncu_long_scoreboard_stall_pct_weighted",
        "ncu_math_throttle_stall_pct_weighted",
    ]
    if not all(column in merged for column in required):
        return
    if not merged[required].fillna(0).to_numpy().any():
        return

    panels = [
        ("ncu_l1_hit_pct_weighted", "L1/TEX Hit Rate", "%"),
        ("ncu_l2_hit_pct_weighted", "L2 Hit Rate", "%"),
        ("ncu_hbm_bytes_per_output_token", "HBM Traffic per Output Token", "bytes/token"),
        ("ncu_sm_pct_weighted", "SM Compute Throughput", "% of sustained peak"),
        ("ncu_sm_active_pct_weighted", "SM Active Cycles", "% of elapsed cycles"),
        ("ncu_math_pipe_pct_weighted", "Math-pipe Activity (Tensor/FMA)", "% of sustained peak"),
        ("ncu_long_scoreboard_stall_pct_weighted", "Memory-dependency Stall", "% cycles/active warp"),
        ("ncu_math_throttle_stall_pct_weighted", "Math-pipe Throttle Stall", "% cycles/active warp"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(30, 12))
    for axis, (column, title, ylabel) in zip(axes.flat, panels, strict=True):
        for stage, sdf in merged.groupby("stage", sort=False):
            sdf = sdf.sort_values("batch_size")
            axis.plot(sdf["batch_size"], sdf[column], marker="o", label=stage)
        axis.set_title(title)
        axis.set_xlabel("batch size")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    axes[0, 2].set_yscale("log")
    axes[-1, -1].legend(fontsize=8, ncol=2)
    fig.suptitle(_workload_title(merged, "Memory-hierarchy Diagnostics"), fontsize=12)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostics figure to {output}")


def stage_summary_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.stem}_stage_summary.csv")


def kernel_summary_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.stem}_kernel_types.csv")


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


def generate_analysis(
    throughput_csv: Path,
    ncu_pattern: str,
    output_prefix: Path,
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

    summary_path = stage_summary_path(output_prefix)
    type_path = kernel_summary_path(output_prefix)
    if not ncu_summary.empty:
        ncu_summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
    if not type_summary.empty:
        type_summary.to_csv(type_path, index=False)
        print(f"Wrote {type_path}")

    plot_path = plot_output or output_prefix.with_suffix(".png")
    plot_analysis(merged, type_summary, plot_path)
    plot_diagnostics(merged, diagnostics_plot_path(output_prefix))
