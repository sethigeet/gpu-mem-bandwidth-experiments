"""Shared model aliases used by all benchmark suites."""

MODEL_REGISTRY = {
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}


def resolve_model(name: str) -> str:
    """Resolve a short project alias, preserving full Hugging Face model IDs."""
    return MODEL_REGISTRY.get(name, name)
