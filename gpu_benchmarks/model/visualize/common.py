from collections.abc import Mapping


def format_config_title(config: Mapping[str, object] | None) -> str:
    if not config:
        return ""
    labels = {
        "model": "",
        "dtype": "",
        "attention": "attn=",
        "prompt_length": "prompt=",
        "batch_size": "bs=",
    }
    return " | ".join(f"{prefix}{config[key]}" for key, prefix in labels.items() if config.get(key) is not None)
