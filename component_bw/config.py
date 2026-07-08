from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

StageName = Literal[
    "attention_kernel",
    "attention_layer",
    "mlp",
    "block",
    "blocks",
    "model",
    "paged_attention",
    "paged_model",
]
LayoutName = Literal["dense_duplicate", "shared", "paged"]

STAGES: tuple[StageName, ...] = (
    "attention_kernel",
    "attention_layer",
    "mlp",
    "block",
    "blocks",
    "model",
    "paged_attention",
    "paged_model",
)
LAYOUTS: tuple[LayoutName, ...] = ("dense_duplicate", "shared", "paged")

DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}


@dataclass(frozen=True)
class ModelPreset:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int


MODEL_PRESETS: dict[str, ModelPreset] = {
    # Phi-3-mini-4k-instruct shape. The synthetic benchmark can use a longer
    # prefix than the original trained context because it never loads positions.
    "phi-3-mini": ModelPreset(
        hidden_size=3072,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=96,
        intermediate_size=8192,
        vocab_size=32064,
    ),
    "llama-7b": ModelPreset(
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        intermediate_size=11008,
        vocab_size=32000,
    ),
    "mistral-7b": ModelPreset(
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=32000,
    ),
}


@dataclass(frozen=True)
class SyntheticConfig:
    model: str = "phi-3-mini"
    hidden_size: int = MODEL_PRESETS["phi-3-mini"].hidden_size
    num_layers: int = MODEL_PRESETS["phi-3-mini"].num_layers
    num_attention_heads: int = MODEL_PRESETS["phi-3-mini"].num_attention_heads
    num_key_value_heads: int = MODEL_PRESETS["phi-3-mini"].num_key_value_heads
    head_dim: int = MODEL_PRESETS["phi-3-mini"].head_dim
    intermediate_size: int = MODEL_PRESETS["phi-3-mini"].intermediate_size
    vocab_size: int = MODEL_PRESETS["phi-3-mini"].vocab_size
    prefix_len: int = 10_000
    decode_tokens: int = 64
    warmup_tokens: int = 5
    batch_size: int | None = None
    max_auto_batch: int = 512
    dtype: str = "fp16"
    layout: LayoutName = "dense_duplicate"
    block_size: int = 16
    gpu_memory_utilization: float = 0.85
    reserve_gb: float = 8.0
    seed: int = 0
    device: str = "cuda"

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def with_model_preset(config: SyntheticConfig, model: str) -> SyntheticConfig:
    preset = MODEL_PRESETS[model]
    return replace(
        config,
        model=model,
        hidden_size=preset.hidden_size,
        num_layers=preset.num_layers,
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_key_value_heads,
        head_dim=preset.head_dim,
        intermediate_size=preset.intermediate_size,
        vocab_size=preset.vocab_size,
    )


def layers_for_stage(stage: StageName, config: SyntheticConfig) -> int:
    if stage in {"blocks", "model", "paged_model"}:
        return config.num_layers
    if stage in {"attention_kernel", "attention_layer", "block", "paged_attention"}:
        return 1
    return 0


def layout_for_stage(stage: StageName, layout: LayoutName) -> LayoutName:
    if stage in {"paged_attention", "paged_model"}:
        return "paged"
    return layout


def estimate_layer_param_count(
    config: SyntheticConfig, include_attention: bool = True, include_mlp: bool = True
) -> int:
    attention = 0
    if include_attention:
        attention = (
            config.hidden_size * config.attention_width
            + 2 * config.hidden_size * config.kv_width
            + config.attention_width * config.hidden_size
        )

    mlp = 0
    if include_mlp:
        mlp = 3 * config.hidden_size * config.intermediate_size

    norms = 2 * config.hidden_size
    return attention + mlp + norms


def estimate_param_bytes(stage: StageName, config: SyntheticConfig) -> int:
    if stage == "attention_kernel":
        return 0
    if stage in {"attention_layer", "paged_attention"}:
        return estimate_layer_param_count(config, include_mlp=False) * config.dtype_bytes
    if stage == "mlp":
        return estimate_layer_param_count(config, include_attention=False) * config.dtype_bytes
    if stage == "block":
        return estimate_layer_param_count(config) * config.dtype_bytes
    if stage == "blocks":
        return estimate_layer_param_count(config) * config.num_layers * config.dtype_bytes
    if stage in {"model", "paged_model"}:
        block_params = estimate_layer_param_count(config) * config.num_layers
        embedding_params = config.vocab_size * config.hidden_size
        lm_head_params = config.hidden_size * config.vocab_size
        final_norm_params = config.hidden_size
        return (block_params + embedding_params + lm_head_params + final_norm_params) * config.dtype_bytes
    raise ValueError(f"unknown stage {stage}")


def estimate_kv_cache_bytes(stage: StageName, config: SyntheticConfig, batch_size: int) -> int:
    layers = layers_for_stage(stage, config)
    if layers == 0:
        return 0

    layout = layout_for_stage(stage, config.layout)
    if layout == "dense_duplicate":
        physical_tokens = batch_size * (config.prefix_len + config.decode_tokens)
    else:
        physical_tokens = config.prefix_len + batch_size * config.decode_tokens

    return physical_tokens * layers * config.num_key_value_heads * config.head_dim * 2 * config.dtype_bytes


def estimate_transient_bytes(stage: StageName, config: SyntheticConfig, batch_size: int) -> int:
    layout = layout_for_stage(stage, config.layout)
    if layout != "paged":
        return 0

    # The PyTorch paged approximation gathers one layer's blocks into a dense
    # [B, KV heads, prefix, head_dim] K/V pair before SDPA. The stored cache may
    # be physically shared, but this transient is batch-sized.
    dense_pair = batch_size * config.prefix_len * config.num_key_value_heads * config.head_dim * 2 * config.dtype_bytes
    # Advanced indexing plus reshape can temporarily hold multiple dense K/V
    # copies. Keep auto batch sizing conservative so the full paged model runs.
    return 4 * dense_pair


def estimate_logical_kv_read_bytes(stage: StageName, config: SyntheticConfig, batch_size: int) -> int:
    layers = layers_for_stage(stage, config)
    if layers == 0:
        return 0
    return (
        batch_size * config.prefix_len * layers * config.num_key_value_heads * config.head_dim * 2 * config.dtype_bytes
    )


def estimate_physical_kv_read_bytes(stage: StageName, config: SyntheticConfig, batch_size: int) -> int:
    layers = layers_for_stage(stage, config)
    if layers == 0:
        return 0
    layout = layout_for_stage(stage, config.layout)
    physical_batch = batch_size if layout == "dense_duplicate" else 1
    return (
        physical_batch
        * config.prefix_len
        * layers
        * config.num_key_value_heads
        * config.head_dim
        * 2
        * config.dtype_bytes
    )


def auto_batch_size(stage: StageName, config: SyntheticConfig, total_memory_bytes: int) -> int:
    usable = int(total_memory_bytes * config.gpu_memory_utilization - config.reserve_gb * 1024**3)
    usable -= estimate_param_bytes(stage, config)
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
