"""Resident-batch decode sweep for stock vLLM 0.22.1 (GPU host only)."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
WORKLOADS = ["128_unique", "10k_shared"]
PROFILE_SAMPLER_WARMUP_REQUESTS = 1024


class CapacityError(RuntimeError):
    """The requested batch cannot stay resident without preemption."""


def _ncu_csv_complete(path: Path) -> bool:
    if not path.exists():
        return False
    return '"Metric Name"' in path.read_text(errors="replace") and "gpu__time_duration.sum" in path.read_text(
        errors="replace"
    )


def add_saturation_args(parser):
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-batch", type=int, help="Optional smoke-test cap; not a memory limit")
    parser.add_argument("--no-ncu", action="store_true")


def worker(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    import random

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != "0.22.1":
        raise RuntimeError(f"Validated adapter requires vLLM 0.22.1, found {vllm.__version__}")
    model_runner_class = None
    original_dummy_sampler_run = None
    if args.profile:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        model_runner_class = cast(Any, GPUModelRunner)
        original_dummy_sampler_run = model_runner_class._dummy_sampler_run

        def capped_dummy_sampler_run(model_runner: Any, hidden_states: torch.Tensor) -> torch.Tensor:
            return original_dummy_sampler_run(
                model_runner,
                hidden_states[:PROFILE_SAMPLER_WARMUP_REQUESTS],
            )

        model_runner_class._dummy_sampler_run = capped_dummy_sampler_run
    try:
        llm = LLM(
            model=args.model,
            dtype="float16",
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=args.workload == "10k_shared",
            enable_chunked_prefill=True,
            max_model_len=16384,
            max_num_seqs=args.batch,
            max_num_batched_tokens=max(16384, args.batch),
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=0,
            swap_space=0,
            block_size=16,
            num_gpu_blocks_override=args.num_gpu_blocks,
            seed=0,
            disable_log_stats=True,
        )
    finally:
        if model_runner_class is not None and original_dummy_sampler_run is not None:
            model_runner_class._dummy_sampler_run = original_dummy_sampler_run
    engine = llm.llm_engine
    core = cast(Any, engine.engine_core).engine_core
    scheduler = core.scheduler
    config = core.vllm_config
    assert config.offload_config.uva.cpu_offload_gb == 0
    assert config.cache_config.kv_offloading_size is None
    assert config.kv_transfer_config is None
    rng = random.Random(0)
    prefix = [rng.randrange(100, 30000) for _ in range(10000)]
    if args.workload == "10k_shared":
        llm.generate([{"prompt_token_ids": prefix}], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    # No chat template or tokenization ambiguity. Unique ID digits precede random
    # tokens, ensuring distinct first blocks even for very large batches.
    for i in range(args.batch):
        suffix = [100 + i % 10000, 100 + i // 10000]
        count = 20 if args.workload == "10k_shared" else 128
        suffix += [rng.randrange(100, 30000) for _ in range(count - 2)]
        prompt = prefix + suffix if args.workload == "10k_shared" else suffix
        engine.add_request(
            str(i),
            {"prompt_token_ids": prompt},
            SamplingParams(
                max_tokens=6000,
                ignore_eos=True,
                temperature=0,
                detokenize=False,
            ),
        )
    original_schedule = scheduler.schedule
    state = {"full": False, "measuring": False}
    traces = []

    def schedule():
        output = original_schedule()
        if output.preempted_req_ids:
            raise CapacityError("KV capacity: scheduler preempted requests")
        counts = output.num_scheduled_tokens
        full = (
            len(counts) == args.batch
            and all(n == 1 for n in counts.values())
            and all(
                scheduler.requests[r].num_computed_tokens >= scheduler.requests[r].num_prompt_tokens for r in counts
            )
        )
        state["full"] = full
        if state["measuring"] and not full:
            raise CapacityError("Measured step did not execute the entire decode batch")
        return output

    scheduler.schedule = schedule
    for _ in range(6000):
        engine.step()
        if state["full"]:
            break
        if len(scheduler.requests) != args.batch:
            raise CapacityError("Requests finished before the full batch became resident")
    else:
        raise RuntimeError("Prefill did not converge within 6000 steps")
    state["measuring"] = True
    for _ in range(args.warmup_steps):
        engine.step()
    contexts = [r.num_computed_tokens for r in scheduler.requests.values()]
    request_ids = list(scheduler.requests)
    if len(request_ids) != args.batch:
        raise RuntimeError(f"Expected {args.batch} resident requests, found {len(request_ids)}")
    blocks = [scheduler.kv_cache_manager.get_block_ids(request_id)[0] for request_id in request_ids]
    if any(not request_blocks for request_blocks in blocks):
        raise RuntimeError("Resident request has no physical KV blocks")
    if args.workload == "10k_shared":
        shared_blocks = 10000 // 16
        if any(b[:shared_blocks] != blocks[0][:shared_blocks] for b in blocks):
            raise RuntimeError("10K prefix did not share physical KV blocks")
    elif len({b for ids in blocks for b in ids}) != sum(map(len, blocks)):
        raise RuntimeError("Unique requests unexpectedly share physical KV blocks")
    torch.cuda.synchronize()
    for step in range(args.decode_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.profile and step == 0:
            torch.cuda.nvtx.range_push("vllm_saturation_decode")
        engine.step()
        torch.cuda.synchronize()
        if args.profile and step == 0:
            torch.cuda.nvtx.range_pop()
        traces.append(time.perf_counter() - start)
    result = {
        "status": "ok",
        "model": args.model,
        "workload": args.workload,
        "batch_size": args.batch,
        "decode_steps": args.decode_steps,
        "decode_tok_s": args.batch * args.decode_steps / sum(traces),
        "step_seconds": traces,
        "context_min": min(contexts),
        "context_max": max(contexts),
        "physical_kv_blocks": len({b for ids in blocks for b in ids}),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "vllm_version": vllm.__version__,
        "model_revision": getattr(config.model_config.hf_config, "_commit_hash", None),
        "torch_version": torch.__version__,
        "profiled": args.profile,
        "num_gpu_blocks_override": args.num_gpu_blocks,
        "profile_block_accounting_version": 2,
        "profile_sampler_warmup_requests": PROFILE_SAMPLER_WARMUP_REQUESTS if args.profile else None,
        "engine_config": str(config),
        "preemptions": 0,
    }
    args.result.write_text(json.dumps(result, indent=2))
    core.shutdown()


def run_saturation(args):
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between 0 and 1")
    if args.decode_steps < 1 or args.warmup_steps < 0 or (args.max_batch is not None and args.max_batch < 1):
        raise ValueError("Invalid step or batch limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output has different settings; choose a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    from gpu_benchmarks.components.analysis import NCU_METRICS

    metrics = [m for m in NCU_METRICS if not m.startswith("dram__bytes_")]

    def point(model, workload, batch, profile=False):
        directory = args.output_dir / model.split("/")[-1] / workload
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"b{batch}{'_ncu' if profile else ''}"
        result = Path(f"{stem}.json")
        if result.exists():
            saved = json.loads(result.read_text())
            stale_profile_result = profile and saved.get("profile_block_accounting_version") != 2
            stale_sampler_warmup_failure = (
                profile
                and saved["status"] == "capacity"
                and "warming up sampler" in saved.get("error", "")
                and saved.get("profile_sampler_warmup_requests") != PROFILE_SAMPLER_WARMUP_REQUESTS
            )
            incomplete_profile_result = (
                profile and saved["status"] == "ok" and not _ncu_csv_complete(Path(f"{stem}.csv"))
            )
            if (
                not stale_profile_result
                and not stale_sampler_warmup_failure
                and not incomplete_profile_result
                and saved["status"] in ("ok", "capacity", "runtime_limit")
            ):
                return saved
        cmd = [
            sys.executable,
            "-m",
            __name__,
            "--model",
            model,
            "--workload",
            workload,
            "--batch",
            str(batch),
            "--result",
            str(result),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--decode-steps",
            str(1 if profile else args.decode_steps),
            "--warmup-steps",
            str(args.warmup_steps),
        ]
        if profile:
            timing = json.loads((directory / f"b{batch}.json").read_text())
            # Kernel replay backs up writable allocations. Avoid backing up tens
            # of GiB of unused cache, while preserving every active block plus
            # enough growth for the same decode horizon and admission pattern.
            if timing.get("profile_block_accounting_version") == 2:
                active_blocks = timing["physical_kv_blocks"]
            elif workload == "10k_shared":
                active_blocks = 625 + batch * ((timing["context_max"] - 10000 + 15) // 16)
            else:
                active_blocks = batch * ((timing["context_max"] + 15) // 16)
            growth_blocks = batch
            blocks = max(1024, active_blocks + growth_blocks)
            cmd += ["--profile", "--num-gpu-blocks", str(blocks)]
            cmd = [
                "ncu",
                "--target-processes",
                "all",
                "--replay-mode",
                "kernel",
                "--nvtx",
                "--nvtx-include",
                "vllm_saturation_decode/",
                "--metrics",
                ",".join(metrics),
                "--csv",
                "--log-file",
                f"{stem}.csv",
                *cmd,
            ]
        print(f"Running {model} {workload} batch={batch} profile={profile}", flush=True)
        with Path(f"{stem}.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            if result.exists():
                saved = json.loads(result.read_text())
                if saved["status"] in ("capacity", "runtime_limit"):
                    return saved
            raise RuntimeError(f"Point failed (not classified as capacity): {stem}.log")
        if not result.exists():
            raise RuntimeError(f"Missing worker result: {result}")
        return json.loads(result.read_text())

    profile_points = []
    for model in args.models:
        for workload in args.workloads:
            low, high, batch = 0, None, 1
            high_status = None
            successful = []
            sweep_successful = []
            while True:
                outcome = point(model, workload, batch)
                if outcome["status"] in ("capacity", "runtime_limit"):
                    high = batch
                    high_status = outcome["status"]
                    break
                successful.append(batch)
                sweep_successful.append(batch)
                low = batch
                if args.max_batch is not None and batch >= args.max_batch:
                    break
                batch = min(batch * 2, args.max_batch) if args.max_batch else batch * 2
            # Refine the resident limit to one request; profile these points too.
            while high is not None and high - low > 1:
                batch = (low + high) // 2
                outcome = point(model, workload, batch)
                if outcome["status"] == "ok":
                    low = batch
                    successful.append(batch)
                else:
                    high = batch
                    high_status = outcome["status"]
            directory = args.output_dir / model.split("/")[-1] / workload
            (directory / "boundary.json").write_text(
                json.dumps(
                    {
                        "largest_resident_batch": low,
                        "first_infeasible_batch": high,
                        "status": (
                            "memory_boundary"
                            if high_status == "capacity"
                            else "runtime_boundary"
                            if high_status == "runtime_limit"
                            else "user_cap"
                        ),
                        "limit_reason": high_status,
                    },
                    indent=2,
                )
            )
            profile_batches = sorted({*sweep_successful, low})
            profile_points.extend((model, workload, batch) for batch in profile_batches)
            analyze(args.output_dir)
    if not args.no_ncu:
        for model, workload, batch in profile_points:
            outcome = point(model, workload, batch, profile=True)
            if outcome["status"] != "ok":
                print(f"NCU capacity failure at batch {batch}; timing result retained", flush=True)
            analyze(args.output_dir)
    if args.max_batch is None and not args.no_ncu and args.models == MODELS and args.workloads == WORKLOADS:
        write_study(args.output_dir)
    return 0


def write_study(output_dir):
    """Prepare the canonical report remotely; fetch installs its final snapshot."""
    import shutil

    import pandas as pd

    summary = pd.read_csv(output_dir / "saturation_summary.csv")
    terminal_statuses = {"memory_boundary", "runtime_boundary"}
    if len(summary) != len(MODELS) * len(WORKLOADS) or not summary["status"].isin(terminal_statuses).all():
        raise ValueError("A final study report requires all four measured terminal boundaries")
    lines = [
        "## Measured results",
        "",
        f"Run artifacts: `{output_dir}`. Saturation is the first measured batch reaching 95% of observed peak throughput.",
        "",
        "### 128-token Unique-prefix Results",
        "",
        "![vLLM 128-token unique-prefix batch sweep](assets/vllm_saturation_128_unique.png)",
        "",
        "![vLLM 128-token unique-prefix memory-hierarchy diagnostics](assets/vllm_saturation_128_unique_diagnostics.png)",
        "",
        "### 10K-token Shared-prefix Results",
        "",
        "![vLLM 10K-token shared-prefix batch sweep](assets/vllm_saturation_10k_shared.png)",
        "",
        "![vLLM 10K-token shared-prefix memory-hierarchy diagnostics](assets/vllm_saturation_10k_shared_diagnostics.png)",
        "",
        "### Summary",
        "",
        "| Model | Workload | Saturation batch | Peak batch | Peak tok/s | Largest resident batch | First infeasible batch | Limit | NCU points / timing points |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model'].split('/')[-1]} | {row['workload']} | {row['saturation_batch']} | {row['peak_batch']} | "
            f"{row['peak_decode_tok_s']:,.2f} | {row['largest_resident_batch']} | {row['first_infeasible_batch']} | "
            f"{row['status']} | "
            f"{row['ncu_completed_points']} / {row['timing_points']} |"
        )
    lines += [
        "",
        "NCU capacity failures remain missing counters; they do not invalidate successful full-budget timing points.",
        "The CSVs contain context ranges and the full counter set for each batch.",
    ]
    document = Path("docs/vllm_saturation_study.md")
    text = document.read_text().split("\n## Measured results")[0]
    text = text.replace(
        "The measurement harness is under validation. Results and memory boundaries must\nnot be inferred from model sizes or from accepted client concurrency.",
        "The full sweep has completed. Measured boundaries and throughput appear below;\nmissing profiler points are explicitly counted.",
    )
    text = text.replace(
        "The full sweep is running remotely in tmux session `vllm_saturation_20260908`.\n"
        "Small-model throughput and NCU smoke checks passed for both workloads at batches\n"
        "1 and 2; the 7B model passed the corresponding throughput/residency smoke checks.\n"
        "Full memory boundaries and final counter curves are not yet available.",
        "The full sweep has completed. Measured boundaries and throughput appear below;\n"
        "missing profiler points are explicitly counted.",
    )
    text += "\n" + "\n".join(lines) + "\n"
    (output_dir / "study.md").write_text(text)
    document.write_text(text)
    Path("docs/assets").mkdir(parents=True, exist_ok=True)
    for workload in WORKLOADS:
        for suffix in ("", "_diagnostics"):
            name = f"saturation_{workload}{suffix}.png"
            shutil.copy2(output_dir / name, Path("docs/assets") / f"vllm_{name}")


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def _workload_plot_rows(df, output_dir, workload):
    """Keep doubling points and the exact resident endpoint, like the component sweep."""
    import pandas as pd

    groups = []
    for model in df.model.drop_duplicates():
        group = df[(df.model == model) & (df.workload == workload)].copy()
        if group.empty:
            continue
        boundary_path = output_dir / model.split("/")[-1] / workload / "boundary.json"
        endpoint = None
        if boundary_path.exists():
            endpoint = json.loads(boundary_path.read_text()).get("largest_resident_batch")
        keep = group.batch_size.map(_is_power_of_two)
        if endpoint is not None:
            keep |= group.batch_size.eq(int(endpoint))
        groups.append(group[keep].sort_values("batch_size"))
    return pd.concat(groups, ignore_index=True) if groups else df.iloc[0:0].copy()


def _plot_workload(df, output_dir, workload):
    import matplotlib.pyplot as plt

    plot_df = _workload_plot_rows(df, output_dir, workload)
    if plot_df.empty:
        return

    labels = {
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat": "Qwen1.5-MoE-A2.7B",
    }
    model_order = [model for model in MODELS if model in set(plot_df.model)]
    model_order.extend(model for model in plot_df.model.drop_duplicates() if model not in model_order)
    colors = dict(zip(model_order, plt.rcParams["axes.prop_cycle"].by_key()["color"], strict=False))
    workload_title = {
        "128_unique": "vLLM 128-token Unique-prefix",
        "10k_shared": "vLLM 10K-token Shared-prefix",
    }[workload]
    for column in (
        "dram_pct",
        "math_pipe_pct",
        "sm_active_pct",
        "l1_hit_pct",
        "l2_hit_pct",
        "dram_bytes_per_token",
        "sm_pct",
        "long_scoreboard_stall_pct",
        "math_throttle_stall_pct",
    ):
        if column not in plot_df:
            plot_df[column] = float("nan")

    panels = [
        ("decode_tok_s", "Decode Throughput vs Batch Size", "tokens/s", True),
        ("dram_pct", "DRAM Bandwidth Utilization vs Batch Size", "% of peak sustained", False),
        ("math_pipe_pct", "Math-pipe Activity (Tensor/FMA) vs Batch Size", "% of sustained peak", False),
        ("sm_active_pct", "SM Active Cycles vs Batch Size", "% of elapsed cycles", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for model in model_order:
        group = plot_df[plot_df.model == model].sort_values("batch_size")
        if group.empty:
            continue
        for axis, (column, _title, _ylabel, _log_y) in zip(axes.flat, panels, strict=True):
            axis.plot(
                group.batch_size,
                group[column],
                marker="o",
                label=labels.get(model, model.split("/")[-1]),
                color=colors[model],
            )
    ticks = sorted(batch for batch in plot_df.batch_size.unique() if _is_power_of_two(batch))
    for axis, (_column, title, ylabel, log_y) in zip(axes.flat, panels, strict=True):
        axis.set_title(title)
        axis.set_xlabel("batch size")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.tick_params(axis="x", rotation=45)
        if log_y:
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[-1, -1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{workload_title} Batch Sweep", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f"saturation_{workload}.png", dpi=150)
    plt.close(fig)

    diagnostics = [
        ("l1_hit_pct", "L1/TEX Hit Rate", "%"),
        ("l2_hit_pct", "L2 Hit Rate", "%"),
        ("dram_bytes_per_token", "HBM Traffic per Output Token", "bytes/token"),
        ("sm_pct", "SM Compute Throughput", "% of sustained peak"),
        ("sm_active_pct", "SM Active Cycles", "% of elapsed cycles"),
        ("math_pipe_pct", "Math-pipe Activity (Tensor/FMA)", "% of sustained peak"),
        ("long_scoreboard_stall_pct", "Memory-dependency Stall", "% cycles/active warp"),
        ("math_throttle_stall_pct", "Math-pipe Throttle Stall", "% cycles/active warp"),
    ]
    diagnostic_columns = [column for column, _title, _ylabel in diagnostics]
    if not plot_df[diagnostic_columns].notna().any().any():
        return
    fig, axes = plt.subplots(2, 4, figsize=(30, 12))
    for model in model_order:
        group = plot_df[plot_df.model == model].sort_values("batch_size")
        if group.empty:
            continue
        for axis, (column, _title, _ylabel) in zip(axes.flat, diagnostics, strict=True):
            axis.plot(
                group.batch_size,
                group[column],
                marker="o",
                label=labels.get(model, model.split("/")[-1]),
                color=colors[model],
            )
    for axis, (_column, title, ylabel) in zip(axes.flat, diagnostics, strict=True):
        axis.set_title(title)
        axis.set_xlabel("batch size")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    if plot_df["dram_bytes_per_token"].gt(0).any():
        axes[0, 2].set_yscale("log")
    axes[-1, -1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{workload_title} Memory-hierarchy Diagnostics", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f"saturation_{workload}_diagnostics.png", dpi=150)
    plt.close(fig)


def analyze(output_dir):
    import pandas as pd

    from gpu_benchmarks.components.analysis import NCU_METRICS
    from gpu_benchmarks.profiling.ncu import load_ncu_metrics

    rows = []
    for path in sorted(output_dir.glob("*/*/b*.json")):
        if "_ncu" in path.stem:
            continue
        data = json.loads(path.read_text())
        if data["status"] != "ok":
            continue
        row = {k: data[k] for k in ("model", "workload", "batch_size", "decode_tok_s", "context_min", "context_max")}
        ncu = path.with_name(path.stem + "_ncu.csv")
        validation = path.with_name(path.stem + "_ncu.json")
        ncu_complete = _ncu_csv_complete(ncu)
        row["ncu_status"] = json.loads(validation.read_text())["status"] if validation.exists() else "pending"
        if row["ncu_status"] == "ok" and not ncu_complete:
            row["ncu_status"] = "incomplete"
        if ncu_complete and validation.exists() and json.loads(validation.read_text())["status"] == "ok":
            profiled = json.loads(validation.read_text())
            if any(profiled[key] != data[key] for key in ("context_min", "context_max", "batch_size")):
                raise ValueError(f"Profiled workload differs from timing workload: {validation}")
            kernels = load_ncu_metrics(ncu)
            if kernels.empty:
                raise ValueError(f"No decode kernels captured: {ncu}")
            missing = set(NCU_METRICS) - {"dram__bytes_read.sum", "dram__bytes_write.sum"} - set(kernels.columns)
            if missing:
                raise ValueError(f"Missing NCU metrics in {ncu}: {missing}")
            required = set(NCU_METRICS) - {"dram__bytes_read.sum", "dram__bytes_write.sum"}
            if kernels[list(required)].isna().any().any():
                raise ValueError(f"NCU returned nonnumeric metric values: {ncu}")
            duration = kernels["gpu__time_duration.sum"]
            if duration.sum() <= 0:
                raise ValueError(f"NCU returned no positive kernel duration: {ncu}")
            for metric, name in NCU_METRICS.items():
                if metric in kernels:
                    row[name] = float((kernels[metric] * duration).sum() / duration.sum())
            math_pipe = kernels[[m for m in NCU_METRICS if "pipe_tensor" in m or "pipe_fma" in m]].max(axis=1)
            row["math_pipe_pct"] = float((math_pipe * duration).sum() / duration.sum())
            row["dram_bytes_per_token"] = float(
                32 * (kernels["dram__sectors_read.sum"] + kernels["dram__sectors_write.sum"]).sum() / data["batch_size"]
            )
            for metric, name in NCU_METRICS.items():
                if metric in kernels and ("sectors" in metric or name == "duration_ns"):
                    row[name] = float(kernels[metric].sum())
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["model", "workload", "batch_size"])
    df.to_csv(output_dir / "summary.csv", index=False)
    saturation_rows = []
    for _, group in df.groupby(["model", "workload"]):
        peak = group.loc[group.decode_tok_s.idxmax()]
        model, workload = str(peak["model"]), str(peak["workload"])
        boundary_path = output_dir / model.split("/")[-1] / workload / "boundary.json"
        boundary = json.loads(boundary_path.read_text()) if boundary_path.exists() else {"status": "in_progress"}
        saturation_rows.append(
            {
                "model": model,
                "workload": workload,
                "saturation_batch": int(group.loc[group.decode_tok_s >= 0.95 * peak.decode_tok_s, "batch_size"].min()),
                "peak_batch": int(peak.batch_size),
                "peak_decode_tok_s": float(peak.decode_tok_s),
                "ncu_completed_points": int((group.ncu_status == "ok").sum()),
                "timing_points": len(group),
                **boundary,
            }
        )
    pd.DataFrame(saturation_rows).to_csv(output_dir / "saturation_summary.csv", index=False)
    for workload in WORKLOADS:
        _plot_workload(df, output_dir, workload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-gpu-blocks", type=int)
    parsed = parser.parse_args()
    try:
        worker(parsed)
    except Exception as exc:
        # Arbitrary crashes and unsupported configurations must never masquerade as OOM.
        message = str(exc)
        capacity = (
            isinstance(exc, CapacityError)
            or "CUDA out of memory" in message
            or "No available memory for the cache blocks" in message
        )
        runtime_limit = "CUDA error: an illegal memory access was encountered" in message
        parsed.result.write_text(
            json.dumps(
                {
                    "status": "capacity" if capacity else "runtime_limit" if runtime_limit else "error",
                    "error": message,
                    "profile_block_accounting_version": 2 if parsed.profile else None,
                    "profile_sampler_warmup_requests": PROFILE_SAMPLER_WARMUP_REQUESTS if parsed.profile else None,
                },
                indent=2,
            )
        )
        raise
