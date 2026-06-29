"""Model aliases for prefix_bw.

Keep this independent from llm_bw.models so prefix_bw does not import
transformers just to resolve a short model name.
"""

MODEL_REGISTRY = {
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}
