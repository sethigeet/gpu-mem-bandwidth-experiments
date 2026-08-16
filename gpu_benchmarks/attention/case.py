from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    cache_seq: int
    dim: int
    dtype: str
