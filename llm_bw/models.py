from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

MODEL_REGISTRY = {
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

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
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_id = MODEL_REGISTRY[model_name]
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
