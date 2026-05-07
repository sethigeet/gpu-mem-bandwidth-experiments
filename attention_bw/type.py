from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    seq: int
    dim: int
    dtype: str
    causal: bool


@dataclass
class Result:
    kernel: str
    batch: int
    heads: int
    seq: int
    dim: int
    dtype: str
    causal: bool
    median_ms: float
    p20_ms: float
    p80_ms: float
    effective_gb_s: float
    utilization_pct_of_peak: float | None
    estimated_bytes: int
    tflops: float
