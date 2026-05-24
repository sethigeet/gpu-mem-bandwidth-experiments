import gc

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def run_decode_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    prompt_length: int,
    decode_tokens: int,
    warmup_tokens: int,
    batch_size: int = 1,
) -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    prompt = "The quick brown fox " * (prompt_length // 4)
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_length,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    torch.cuda.nvtx.range_push(f"llm_bw:{model_name}:case")
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = next_token
            torch.cuda.synchronize()

            for _ in range(warmup_tokens):
                torch.cuda.nvtx.range_push(f"llm_bw:{model_name}:warmup")
                try:
                    new_attention_mask = torch.ones(
                        (batch_size, attention_mask.shape[1] + generated.shape[1]),
                        device=model.device,
                        dtype=attention_mask.dtype,
                    )
                    outputs = model(
                        next_token,
                        attention_mask=new_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                finally:
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()

            for _ in range(decode_tokens):
                torch.cuda.nvtx.range_push(f"llm_bw:{model_name}:iter")
                try:
                    new_attention_mask = torch.ones(
                        (batch_size, attention_mask.shape[1] + generated.shape[1]),
                        device=model.device,
                        dtype=attention_mask.dtype,
                    )
                    outputs = model(
                        next_token,
                        attention_mask=new_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                finally:
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()

    finally:
        torch.cuda.nvtx.range_pop()

    del input_ids, attention_mask, past_key_values, outputs, generated, next_token
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
