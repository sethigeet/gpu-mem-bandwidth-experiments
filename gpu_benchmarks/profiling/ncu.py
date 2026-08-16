from io import StringIO
from pathlib import Path

import pandas as pd


def load_ncu_metrics(path: Path) -> pd.DataFrame:
    """Read an Nsight Compute CSV and pivot one row per captured kernel."""
    lines = path.read_text(errors="replace").splitlines()
    csv_text = "\n".join(line for line in lines if line and not line.startswith("=="))
    metrics = pd.read_csv(StringIO(csv_text))
    if metrics.empty:
        return pd.DataFrame()

    metrics["Metric Value"] = (
        metrics["Metric Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    result = metrics.pivot_table(
        index=["ID", "Kernel Name"],
        columns="Metric Name",
        values="Metric Value",
        aggfunc="first",
    ).reset_index()
    result.columns.name = None
    return result
