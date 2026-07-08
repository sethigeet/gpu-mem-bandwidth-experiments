from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from component_bw.config import (
    STAGES,
    StageName,
    SyntheticConfig,
    estimate_kv_cache_bytes,
    estimate_logical_kv_read_bytes,
    estimate_param_bytes,
    estimate_physical_kv_read_bytes,
    estimate_transient_bytes,
    layers_for_stage,
    layout_for_stage,
)
from component_bw.modules import (
    DecoderBlock,
    DecoderStack,
    DenseKVCache,
    KVCache,
    PagedKVCache,
    SyntheticAttention,
    SyntheticMLP,
    SyntheticModel,
    sdpa_decode,
    torch_dtype,
)


@dataclass
class RunResult:
    stage: str
    layout: str
    model: str
    batch_size: int
    prefix_len: int
    decode_tokens: int
    warmup_tokens: int
    dtype: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    block_size: int
    wall_time_s: float
    total_output_tokens: int
    throughput_toks_s: float
    per_token_ms: float
    logical_kv_read_bytes_per_iter: int
    logical_kv_read_bytes_per_output_token: float
    physical_kv_read_bytes_per_iter: int
    physical_kv_read_bytes_per_output_token: float
    kv_cache_bytes: int
    transient_bytes: int
    param_bytes: int
    estimated_launches_per_token: int
    output_checksum: float


def result_to_row(result: RunResult) -> dict[str, object]:
    return asdict(result)


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


def write_config(config: SyntheticConfig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_row(), indent=2) + "\n")


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on the remote GPU host.")


def _device_total_memory() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)


def _device_free_memory() -> int:
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _memory_budget(config: SyntheticConfig) -> int:
    total_budget = int(_device_total_memory() * config.gpu_memory_utilization - config.reserve_gb * 1024**3)
    free_budget = int(_device_free_memory() * config.gpu_memory_utilization)
    return max(0, min(total_budget, free_budget))


def _auto_batch_size_for_budget(stage: StageName, config: SyntheticConfig, budget: int) -> int:
    usable = budget - estimate_param_bytes(stage, config)
    if usable <= 0:
        return 1

    lo, hi = 1, max(1, config.max_auto_batch)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        working_set = estimate_kv_cache_bytes(stage, config, mid) + estimate_transient_bytes(stage, config, mid)
        if working_set <= usable:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def resolve_batch_size(stage: StageName, config: SyntheticConfig) -> int:
    if config.batch_size is not None:
        return config.batch_size
    return _auto_batch_size_for_budget(stage, config, _memory_budget(config))


def _check_memory_budget(stage: StageName, config: SyntheticConfig, batch_size: int) -> None:
    budget = _memory_budget(config)
    required = (
        estimate_param_bytes(stage, config)
        + estimate_kv_cache_bytes(stage, config, batch_size)
        + estimate_transient_bytes(stage, config, batch_size)
    )
    if required > max(0, budget):
        raise RuntimeError(
            f"estimated {stage} working set for batch={batch_size} is {required / 1024**3:.2f} GiB, "
            f"but the current free-memory budget is {max(0, budget) / 1024**3:.2f} GiB. "
            "Free the GPU or reduce --layers/--batch-size/--prefix-len."
        )


def _set_seed(config: SyntheticConfig) -> None:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def _make_dense_cache(config: SyntheticConfig, stage: StageName, batch_size: int) -> DenseKVCache:
    dtype = torch_dtype(config.dtype)
    layers = layers_for_stage(stage, config)
    layout = layout_for_stage(stage, config.layout)
    physical_batch = batch_size if layout == "dense_duplicate" else 1
    shape = (
        layers,
        physical_batch,
        config.num_key_value_heads,
        config.prefix_len,
        config.head_dim,
    )
    k = torch.randn(shape, device=config.device, dtype=dtype)
    v = torch.randn(shape, device=config.device, dtype=dtype)
    return DenseKVCache(k=k, v=v)


def _make_paged_cache(config: SyntheticConfig, stage: StageName, batch_size: int) -> PagedKVCache:
    dtype = torch_dtype(config.dtype)
    layers = layers_for_stage(stage, config)
    num_blocks = (config.prefix_len + config.block_size - 1) // config.block_size
    shape = (
        layers,
        num_blocks,
        config.num_key_value_heads,
        config.block_size,
        config.head_dim,
    )
    k_blocks = torch.randn(shape, device=config.device, dtype=dtype)
    v_blocks = torch.randn(shape, device=config.device, dtype=dtype)
    block_table = torch.arange(num_blocks, device=config.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return PagedKVCache(k_blocks=k_blocks, v_blocks=v_blocks, block_table=block_table, block_size=config.block_size)


def _make_cache(config: SyntheticConfig, stage: StageName, batch_size: int) -> KVCache | None:
    if layers_for_stage(stage, config) == 0:
        return None
    if layout_for_stage(stage, config.layout) == "paged":
        return _make_paged_cache(config, stage, batch_size)
    return _make_dense_cache(config, stage, batch_size)


def _estimated_launches(stage: StageName, config: SyntheticConfig) -> int:
    per_attention = 5
    per_mlp = 4
    per_block = per_attention + per_mlp + 4
    if stage == "attention_kernel":
        return 1
    if stage == "attention_layer":
        return per_attention
    if stage == "mlp":
        return per_mlp
    if stage == "block":
        return per_block
    if stage == "blocks":
        return config.num_layers * per_block
    if stage == "model":
        return config.num_layers * per_block + 4
    if stage == "paged_attention":
        return per_attention + 2
    if stage == "paged_model":
        return config.num_layers * (per_block + 2) + 4
    raise ValueError(f"unknown stage {stage}")


class _StageExecutor:
    def __init__(self, stage: StageName, config: SyntheticConfig, batch_size: int, cache: KVCache | None) -> None:
        self.stage = stage
        self.config = config
        self.batch_size = batch_size
        self.cache = cache
        self.dtype = torch_dtype(config.dtype)
        self.hidden = torch.randn(batch_size, config.hidden_size, device=config.device, dtype=self.dtype)
        self.token_ids = torch.randint(0, config.vocab_size, (batch_size, 1), device=config.device)

        if stage == "attention_kernel":
            self.q = torch.randn(
                batch_size,
                config.num_attention_heads,
                1,
                config.head_dim,
                device=config.device,
                dtype=self.dtype,
            )
            self.module = None
        elif stage in {"attention_layer", "paged_attention"}:
            self.module = SyntheticAttention(config).to(device=config.device, dtype=self.dtype).eval()
        elif stage == "mlp":
            self.module = SyntheticMLP(config).to(device=config.device, dtype=self.dtype).eval()
        elif stage == "block":
            self.module = DecoderBlock(config).to(device=config.device, dtype=self.dtype).eval()
        elif stage == "blocks":
            self.module = DecoderStack(config).to(device=config.device, dtype=self.dtype).eval()
        elif stage in {"model", "paged_model"}:
            self.module = SyntheticModel(config).to(device=config.device, dtype=self.dtype).eval()
        else:
            raise ValueError(f"unknown stage {stage}")

    def step(self) -> torch.Tensor:
        if self.stage == "attention_kernel":
            if self.cache is None:
                raise RuntimeError("attention kernel stage requires a cache")
            k_cache, v_cache = self.cache.layer(0, self.batch_size, self.config.prefix_len)
            out = sdpa_decode(self.q, k_cache, v_cache, self.config.num_attention_heads)
            self.q = out.reshape(self.batch_size, self.config.num_attention_heads, 1, self.config.head_dim)
            return out

        if self.stage in {"attention_layer", "paged_attention"}:
            if self.cache is None:
                raise RuntimeError(f"{self.stage} requires a cache")
            assert self.module is not None
            k_cache, v_cache = self.cache.layer(0, self.batch_size, self.config.prefix_len)
            self.hidden = self.module(self.hidden, k_cache, v_cache)
            return self.hidden

        if self.stage == "mlp":
            assert self.module is not None
            self.hidden = self.module(self.hidden)
            return self.hidden

        if self.stage == "block":
            if self.cache is None:
                raise RuntimeError("block stage requires a cache")
            assert self.module is not None
            k_cache, v_cache = self.cache.layer(0, self.batch_size, self.config.prefix_len)
            self.hidden = self.module(self.hidden, k_cache, v_cache)
            return self.hidden

        if self.stage == "blocks":
            if self.cache is None:
                raise RuntimeError("blocks stage requires a cache")
            assert self.module is not None
            self.hidden = self.module(self.hidden, self.cache, self.config.prefix_len)
            return self.hidden

        if self.stage in {"model", "paged_model"}:
            if self.cache is None:
                raise RuntimeError(f"{self.stage} requires a cache")
            assert self.module is not None
            logits = self.module(self.token_ids, self.cache, self.config.prefix_len)
            self.token_ids = logits.argmax(dim=-1, keepdim=True)
            return logits

        raise ValueError(f"unknown stage {self.stage}")


def _run_steps(executor: _StageExecutor, stage: StageName, step_count: int, range_kind: str) -> torch.Tensor:
    out = torch.empty((), device=executor.config.device)
    for _ in range(step_count):
        torch.cuda.nvtx.range_push(f"component_bw:{stage}:{range_kind}")
        try:
            out = executor.step()
        finally:
            torch.cuda.nvtx.range_pop()
    return out


def run_stage(stage: StageName, config: SyntheticConfig) -> RunResult:
    _ensure_cuda()
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    _set_seed(config)

    batch_size = resolve_batch_size(stage, config)
    _check_memory_budget(stage, config, batch_size)
    cache = _make_cache(config, stage, batch_size)
    executor = _StageExecutor(stage, config, batch_size, cache)

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"component_bw:{stage}:case")
    try:
        with torch.inference_mode():
            _run_steps(executor, stage, config.warmup_tokens, "warmup")
            torch.cuda.synchronize()

            start = time.perf_counter()
            out = _run_steps(executor, stage, config.decode_tokens, "iter")
            torch.cuda.synchronize()
            wall_time = time.perf_counter() - start
    finally:
        torch.cuda.nvtx.range_pop()

    if out.numel() == 0:
        raise RuntimeError("empty output")
    checksum = float(out.float().mean().item())
    total_output_tokens = batch_size * config.decode_tokens
    throughput = total_output_tokens / wall_time if wall_time > 0 else 0.0

    logical_read = estimate_logical_kv_read_bytes(stage, config, batch_size)
    physical_read = estimate_physical_kv_read_bytes(stage, config, batch_size)
    result = RunResult(
        stage=stage,
        layout=layout_for_stage(stage, config.layout),
        model=config.model,
        batch_size=batch_size,
        prefix_len=config.prefix_len,
        decode_tokens=config.decode_tokens,
        warmup_tokens=config.warmup_tokens,
        dtype=config.dtype,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        block_size=config.block_size,
        wall_time_s=wall_time,
        total_output_tokens=total_output_tokens,
        throughput_toks_s=throughput,
        per_token_ms=(wall_time / total_output_tokens * 1000.0) if total_output_tokens else 0.0,
        logical_kv_read_bytes_per_iter=logical_read,
        logical_kv_read_bytes_per_output_token=logical_read / batch_size if batch_size else 0.0,
        physical_kv_read_bytes_per_iter=physical_read,
        physical_kv_read_bytes_per_output_token=physical_read / batch_size if batch_size else 0.0,
        kv_cache_bytes=estimate_kv_cache_bytes(stage, config, batch_size),
        transient_bytes=estimate_transient_bytes(stage, config, batch_size),
        param_bytes=estimate_param_bytes(stage, config),
        estimated_launches_per_token=_estimated_launches(stage, config),
        output_checksum=checksum,
    )

    del executor, cache, out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    return result


def run_matrix(stages: list[StageName], config: SyntheticConfig) -> list[RunResult]:
    return [run_stage(stage, config) for stage in stages]
