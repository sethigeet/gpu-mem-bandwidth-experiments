from dataclasses import dataclass

import torch
from transformers import (  # ty: ignore[unresolved-import]  # Optional GPU-host dependency.
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from gpu_benchmarks.models import resolve_model

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class ModelConfig:
    model_id: str
    dtype: torch.dtype
    attention_impl: str
    device: str = "cuda"


def load_model(
    model_name: str,
    dtype: str = "fp16",
    attention_impl: str = "sdpa",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_id = resolve_model(model_name)
    torch_dtype = DTYPE_MAP[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=torch_dtype,
        attn_implementation=attention_impl,
        device_map="cuda",
    )
    model.eval()

    return model, tokenizer
