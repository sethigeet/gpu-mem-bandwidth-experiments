from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    seq: int
    dim: int
    dtype: str
    causal: bool
