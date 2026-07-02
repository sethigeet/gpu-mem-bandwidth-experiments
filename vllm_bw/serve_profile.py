"""Run a vLLM OpenAI server and drive it with `vllm bench serve`.

This module is intended to be launched under Nsight Systems. It emits an NVTX range
around the measured client benchmark so the exported SQLite can isolate the load
window from server startup and warmup.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from vllm_bw.models import resolve_model

_MEASURED_RANGE = "vllm_bw:serve:bench"


def _vllm_executable() -> str:
    return shutil.which("vllm") or "vllm"


@contextlib.contextmanager
def _nvtx_range(name: str):
    pushed = False
    try:
        import torch

        torch.cuda.nvtx.range_push(name)
        pushed = True
    except Exception as exc:
        print(f"warning: could not push NVTX range {name!r}: {exc}", flush=True)

    try:
        yield
    finally:
        if pushed:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except Exception as exc:
                print(f"warning: could not pop NVTX range {name!r}: {exc}", flush=True)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def _wait_for_health(url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc
        time.sleep(2)

    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"Timed out waiting for vLLM health endpoint {url}{detail}")


def _server_command(args: argparse.Namespace, model: str) -> list[str]:
    cmd = [
        _vllm_executable(),
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def _bench_command(args: argparse.Namespace, model: str, num_prompts: int) -> list[str]:
    cmd = [
        _vllm_executable(),
        "bench",
        "serve",
        "--backend",
        args.bench_backend,
        "--model",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--endpoint",
        args.endpoint,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        args.request_rate,
        "--seed",
        str(args.seed),
    ]
    if args.max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(args.max_concurrency)])
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    return cmd


def _run_and_log(cmd: list[str], log_path: Path, timeout_s: float | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        log.write(completed.stdout)
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd, output=completed.stdout)


def run_serve_profile(args: argparse.Namespace) -> int:
    model = resolve_model(args.model)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.log_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": model,
                "host": args.host,
                "port": args.port,
                "dtype": args.dtype,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "random_input_len": args.random_input_len,
                "random_output_len": args.random_output_len,
                "num_prompts": args.num_prompts,
                "request_rate": args.request_rate,
                "max_concurrency": args.max_concurrency,
                "nvtx_range": _MEASURED_RANGE,
            },
            indent=2,
        )
        + "\n"
    )

    server_log = (args.log_dir / "server.log").open("w")
    server_cmd = _server_command(args, model)
    print(f"$ {' '.join(server_cmd)}", flush=True)
    server = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )

    try:
        _wait_for_health(f"http://{args.host}:{args.port}/health", server, args.server_timeout_s)
        print("vLLM server is healthy", flush=True)

        if args.warmup_prompts > 0:
            warmup_cmd = _bench_command(args, model, args.warmup_prompts)
            _run_and_log(warmup_cmd, args.log_dir / "warmup.log", args.bench_timeout_s)

        bench_cmd = _bench_command(args, model, args.num_prompts)
        with _nvtx_range(_MEASURED_RANGE):
            _run_and_log(bench_cmd, args.log_dir / "bench.log", args.bench_timeout_s)
    finally:
        _terminate_process_group(server)
        server_log.close()

    return 0


def _run_checked(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_client_nsys_profile(args: argparse.Namespace) -> int:
    """Start the server normally, then profile only the request-load client.

    Nsight Systems GPU metrics are device-wide, so profiling the client process
    is enough to capture DRAM/SM timelines while avoiding server startup/model
    load and long-lived server shutdown issues.
    """
    model = resolve_model(args.model)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    server_log = (args.log_dir / "server.log").open("w")
    server_cmd = _server_command(args, model)
    print(f"$ {' '.join(server_cmd)}", flush=True)
    server = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )

    output_prefix = str(args.output_prefix)
    try:
        _wait_for_health(f"http://{args.host}:{args.port}/health", server, args.server_timeout_s)
        print("vLLM server is healthy", flush=True)

        if args.warmup_prompts > 0:
            warmup_cmd = _bench_command(args, model, args.warmup_prompts)
            _run_and_log(warmup_cmd, args.log_dir / "warmup.log", args.bench_timeout_s)

        bench_cmd = _bench_command(args, model, args.num_prompts)
        nsys_cmd = [
            "nsys",
            "profile",
            "--trace",
            args.nsys_trace,
            "--gpu-metrics-devices",
            args.nsys_gpu_metrics_devices,
            "--gpu-metrics-frequency",
            str(args.nsys_gpu_metrics_frequency),
            "--duration",
            "0",
            "--output",
            output_prefix,
            "--force-overwrite=true",
            *bench_cmd,
        ]
        _run_and_log(nsys_cmd, args.log_dir / "bench_nsys.log", args.bench_timeout_s)
    finally:
        _terminate_process_group(server)
        server_log.close()

    sqlite_path = f"{output_prefix}.sqlite"
    _run_checked(["nsys", "export", "--type=sqlite", f"--output={sqlite_path}", f"{output_prefix}.nsys-rep"])

    from vllm_bw.visualize import visualize_nsys

    visualize_nsys(
        Path(sqlite_path),
        Path(f"{output_prefix}.png"),
        summary_output=Path(f"{output_prefix}_summary.csv"),
        measured_only=False,
    )
    print(f"Exported {sqlite_path}")
    print(f"Wrote {output_prefix}.png and {output_prefix}_summary.csv")
    return 0


def add_serve_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="phi-3-mini", help="Registry key or raw HF model id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow vLLM to load model repositories with custom code",
    )
    parser.add_argument("--bench-backend", default="openai", help="Backend passed to `vllm bench serve`")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--random-input-len", type=int, default=2048)
    parser.add_argument("--random-output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force random-output-len generated tokens when the benchmark backend supports it",
    )
    parser.add_argument("--server-timeout-s", type=float, default=900)
    parser.add_argument("--bench-timeout-s", type=float, default=1800)
    parser.add_argument("--log-dir", type=Path, default=Path("results/vllm_bw_serve_logs"))


def add_client_nsys_profile_args(parser: argparse.ArgumentParser) -> None:
    add_serve_profile_args(parser)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--nsys-trace", default="nvtx")
    parser.add_argument("--nsys-gpu-metrics-devices", default="all")
    parser.add_argument("--nsys-gpu-metrics-frequency", type=int, default=50000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run vLLM serving under request load.")
    add_serve_profile_args(parser)
    return run_serve_profile(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
