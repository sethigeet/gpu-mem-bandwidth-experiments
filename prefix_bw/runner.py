"""Runs prefix-sharing workloads on vLLM and measures decode throughput.

vLLM is imported lazily so the CLI's visualize path stays importable on a machine
without a GPU (vLLM is a remote-only dependency, like torch's CUDA build).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from inspect import Parameter, signature

import torch

from prefix_bw.scheduler import build_request_batches
from prefix_bw.workload import Workload


@dataclass
class RunResult:
    """Outcome of running one workload, plus the swept parameters for plotting."""

    experiment: str
    series: str
    x_name: str
    x_value: float
    num_requests: int
    avg_prompt_len: float
    decode_tokens: int
    total_output_tokens: int
    wall_time_s: float
    throughput_toks_s: float
    decode_throughput_toks_s: float
    max_num_seqs: int
    schedule: str
    num_generate_calls: int
    max_request_batch_size: int
    min_prefix_batch_size: int


def build_llm(
    model_id: str,
    dtype: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool = True,
    enable_prefix_caching: bool = True,
):
    from vllm import LLM  # ty: ignore[unresolved-import]

    llm_kwargs = {
        "model": model_id,
        "dtype": dtype,
        "enable_prefix_caching": enable_prefix_caching,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        # Some vLLM/Phi model combinations reject prefix caching with sliding
        # windows enabled. Older vLLM releases do not expose this knob.
        "disable_sliding_window": True,
    }
    llm_signature = signature(LLM)
    has_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in llm_signature.parameters.values())
    supported = set(llm_signature.parameters)
    if has_kwargs:
        try:
            from vllm.engine.arg_utils import EngineArgs  # ty: ignore[unresolved-import]

            supported |= set(signature(EngineArgs).parameters)
        except Exception:
            return LLM(**llm_kwargs)

    llm_kwargs = {key: value for key, value in llm_kwargs.items() if key in supported}
    return LLM(**llm_kwargs)


def get_vocab_size(llm) -> int:
    return llm.get_tokenizer().vocab_size


def _decode_window_s(outputs: list) -> float | None:
    first_token_times = [
        o.metrics.first_token_time for o in outputs if o.metrics is not None and o.metrics.first_token_time is not None
    ]
    finished_times = [
        o.metrics.finished_time for o in outputs if o.metrics is not None and o.metrics.finished_time is not None
    ]
    if not first_token_times or not finished_times:
        return None
    window = max(finished_times) - min(first_token_times)
    return window if window > 0 else None


def _generate_token_ids(llm, token_id_batches: list[list[int]], sampling_params):
    generate_signature = signature(llm.generate)
    if "prompt_token_ids" in generate_signature.parameters:
        return llm.generate(
            prompt_token_ids=token_id_batches,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    return llm.generate(
        [{"prompt_token_ids": tokens} for tokens in token_id_batches],
        sampling_params,
        use_tqdm=False,
    )


def run_workload(
    llm,
    workload: Workload,
    experiment: str,
    x_name: str,
    x_value: float,
    decode_tokens: int,
    max_num_seqs: int,
    series: str = "main",
    warmup: bool = True,
    schedule: str = "single",
    prefix_batch_size: int | None = None,
    min_prefix_batch_size: int = 64,
    prefix_hash_chunk_size: int = 64,
    min_shared_prefix_len: int = 128,
    prefix_auto_min_fill: float = 0.20,
    offline_prefix_min_gain: float = 0.10,
    offline_prefix_min_fill: float = 0.20,
    offline_prefix_max_waves: int = 0,
) -> RunResult:
    from vllm import SamplingParams  # ty: ignore[unresolved-import]

    label = f"{experiment}:{series}={x_value}:schedule={schedule}"

    # Warmup: prefill the distinct shared prefixes so their KV blocks are resident in
    # the prefix cache before the measured run. This mirrors the paper's methodology of
    # holding per-token compute constant so differences come from memory access patterns.
    if warmup and workload.shared_prefixes:
        torch.cuda.nvtx.range_push(f"prefix_bw:{label}:warmup")
        try:
            warm_params = SamplingParams(max_tokens=1, temperature=0.0)
            _generate_token_ids(llm, workload.shared_prefixes, warm_params)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    sampling = SamplingParams(
        max_tokens=decode_tokens,
        min_tokens=decode_tokens,
        ignore_eos=True,
        temperature=0.0,
    )
    request_batches = build_request_batches(
        workload,
        schedule,
        max_num_seqs,
        decode_tokens,
        prefix_batch_size,
        min_prefix_batch_size,
        prefix_hash_chunk_size,
        min_shared_prefix_len,
        prefix_auto_min_fill,
        offline_prefix_min_gain,
        offline_prefix_min_fill,
        offline_prefix_max_waves,
    )

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"prefix_bw:{label}:case")
    start = time.perf_counter()
    outputs = []
    decode_windows: list[float] = []
    try:
        for batch_idx, request_batch in enumerate(request_batches):
            torch.cuda.nvtx.range_push(f"prefix_bw:{label}:batch={batch_idx}")
            try:
                batch_outputs = _generate_token_ids(llm, request_batch, sampling)
            finally:
                torch.cuda.nvtx.range_pop()
            outputs.extend(batch_outputs)
            decode_window = _decode_window_s(batch_outputs)
            if decode_window is not None:
                decode_windows.append(decode_window)
        torch.cuda.synchronize()
        wall_time = time.perf_counter() - start
    finally:
        torch.cuda.nvtx.range_pop()

    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Decode-only throughput from vLLM's per-request timestamps: total decode tokens over
    # the window between the earliest first-token and the latest completion. Falls back to
    # wall-clock if metrics are unavailable.
    decode_throughput = total_output / wall_time
    if decode_windows:
        decode_throughput = total_output / sum(decode_windows)

    return RunResult(
        experiment=experiment,
        series=series,
        x_name=x_name,
        x_value=x_value,
        num_requests=workload.num_requests,
        avg_prompt_len=workload.avg_prompt_len,
        decode_tokens=decode_tokens,
        total_output_tokens=total_output,
        wall_time_s=wall_time,
        throughput_toks_s=total_output / wall_time,
        decode_throughput_toks_s=decode_throughput,
        max_num_seqs=max_num_seqs,
        schedule=schedule,
        num_generate_calls=len(request_batches),
        max_request_batch_size=max((len(batch) for batch in request_batches), default=0),
        min_prefix_batch_size=min_prefix_batch_size,
    )


def result_to_row(result: RunResult) -> dict:
    return asdict(result)
