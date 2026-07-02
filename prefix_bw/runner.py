"""Runs prefix-sharing workloads on vLLM and measures decode throughput.

vLLM is imported lazily so the CLI's visualize path stays importable on a machine
without a GPU (vLLM is a remote-only dependency, like torch's CUDA build).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from inspect import signature

import torch

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
    supported = set(signature(LLM).parameters)
    llm_kwargs = {key: value for key, value in llm_kwargs.items() if key in supported}

    return LLM(**llm_kwargs)


def get_vocab_size(llm) -> int:
    return llm.get_tokenizer().vocab_size


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
    warmup_requests: list[list[int]] | None = None,
) -> RunResult:
    from vllm import SamplingParams  # ty: ignore[unresolved-import]

    label = f"{experiment}:{series}={x_value}"

    # Warmup: prefill the distinct shared prefixes so their KV blocks are resident in
    # the prefix cache before the measured run. This mirrors the paper's methodology of
    # holding per-token compute constant so differences come from memory access patterns.
    warmup_prompts = warmup_requests if warmup_requests is not None else workload.shared_prefixes
    if warmup and warmup_prompts:
        torch.cuda.nvtx.range_push(f"prefix_bw:{label}:warmup")
        try:
            warm_params = SamplingParams(max_tokens=1, temperature=0.0)
            llm.generate(
                [{"prompt_token_ids": p} for p in warmup_prompts],
                warm_params,
                use_tqdm=False,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    sampling = SamplingParams(
        max_tokens=decode_tokens,
        min_tokens=decode_tokens,
        ignore_eos=True,
        temperature=0.0,
    )
    prompts = [{"prompt_token_ids": tokens} for tokens in workload.requests]

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"prefix_bw:{label}:case")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - start
    torch.cuda.nvtx.range_pop()

    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Decode-only throughput from vLLM's per-request timestamps: total decode tokens over
    # the window between the earliest first-token and the latest completion. Falls back to
    # wall-clock if metrics are unavailable.
    decode_throughput = total_output / wall_time
    first_token_times = [
        o.metrics.first_token_time for o in outputs if o.metrics is not None and o.metrics.first_token_time is not None
    ]
    finished_times = [
        o.metrics.finished_time for o in outputs if o.metrics is not None and o.metrics.finished_time is not None
    ]
    if first_token_times and finished_times:
        decode_window = max(finished_times) - min(first_token_times)
        if decode_window > 0:
            decode_throughput = total_output / decode_window

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
    )


def result_to_row(result: RunResult) -> dict:
    return asdict(result)
