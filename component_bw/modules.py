from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from component_bw.config import SyntheticConfig


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype {name}")


@dataclass
class DenseKVCache:
    k: torch.Tensor
    v: torch.Tensor

    def layer(self, layer_idx: int, batch_size: int, _prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k[layer_idx]
        v = self.v[layer_idx]
        if k.shape[0] == 1 and batch_size > 1:
            k = k.expand(batch_size, -1, -1, -1)
            v = v.expand(batch_size, -1, -1, -1)
        return k, v


@dataclass
class PagedKVCache:
    k_blocks: torch.Tensor
    v_blocks: torch.Tensor
    block_table: torch.Tensor
    block_size: int

    def layer(self, layer_idx: int, _batch_size: int, prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # This intentionally materializes the gathered view: it approximates the
        # block-table indirection and loss of contiguous KV layout before SDPA.
        k = self.k_blocks[layer_idx][self.block_table]
        v = self.v_blocks[layer_idx][self.block_table]
        batch, blocks, kv_heads, block_size, head_dim = k.shape
        k = k.permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, blocks * block_size, head_dim)
        v = v.permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, blocks * block_size, head_dim)
        return k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]


KVCache = DenseKVCache | PagedKVCache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(input_dtype) * self.weight


def _repeat_kv(x: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    kv_heads = x.shape[1]
    if kv_heads == num_attention_heads:
        return x
    if num_attention_heads % kv_heads != 0:
        raise ValueError(f"attention heads ({num_attention_heads}) must be divisible by KV heads ({kv_heads})")
    return x.repeat_interleave(num_attention_heads // kv_heads, dim=1)


def sdpa_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    k = _repeat_kv(k, num_attention_heads)
    v = _repeat_kv(v, num_attention_heads)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False).squeeze(2)


class SyntheticAttention(nn.Module):
    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.attention_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.o_proj = nn.Linear(config.attention_width, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        q = self.q_proj(hidden).view(batch, self.config.num_attention_heads, 1, self.config.head_dim)

        # Project the current token so this stage includes the same QKV projection
        # work as a decode attention layer. The synthetic long-prefix attention uses
        # the preallocated prefix cache to keep every iteration shape-stable.
        self.k_proj(hidden)
        self.v_proj(hidden)

        out = sdpa_decode(q, k_cache, v_cache, self.config.num_attention_heads)
        out = out.reshape(batch, self.config.attention_width)
        return self.o_proj(out)


class SyntheticMLP(nn.Module):
    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = SyntheticAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = SyntheticMLP(config)

    def forward(self, hidden: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), k_cache, v_cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DecoderStack(nn.Module):
    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])

    def forward(self, hidden: torch.Tensor, cache: KVCache, prefix_len: int) -> torch.Tensor:
        batch = hidden.shape[0]
        for layer_idx, layer in enumerate(self.layers):
            k_cache, v_cache = cache.layer(layer_idx, batch, prefix_len)
            hidden = layer(hidden, k_cache, v_cache)
        return hidden


class SyntheticModel(nn.Module):
    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = DecoderStack(config)
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, cache: KVCache, prefix_len: int) -> torch.Tensor:
        hidden = self.embed_tokens(token_ids).squeeze(1)
        hidden = self.layers(hidden, cache, prefix_len)
        return self.lm_head(self.norm(hidden))
