import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from attention_bw.type import Result


def write_outputs(results: list[Result], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    if out.suffix == ".json":
        out.write_text(json.dumps(rows, indent=2) + "\n")
        return
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results(results: list[Result]) -> None:
    cols = [
        "kernel",
        "batch",
        "heads",
        "seq",
        "dim",
        "dtype",
        "causal",
        "median_ms",
        "effective_gb_s",
        "utilization_pct_of_peak",
        "tflops",
    ]
    print(pd.DataFrame([asdict(r) for r in results])[cols].to_markdown(index=False, floatfmt=".3f"))
