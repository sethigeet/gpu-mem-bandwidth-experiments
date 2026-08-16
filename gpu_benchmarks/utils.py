import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


def write_dict_rows(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    """Write homogeneous dictionary rows to CSV, creating parent directories."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
