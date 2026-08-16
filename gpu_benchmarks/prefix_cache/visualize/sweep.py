from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_X_LABELS = {
    "beta": "Fraction of requests on prefix A (beta)",
    "share_fraction": "Shared prefix fraction (p)",
    "num_groups": "Number of prefix groups",
    "batch_size": "Max batch size (max_num_seqs)",
}

_TITLES = {
    "homogeneity_fraction": "Prefix Homogeneity vs Decode Throughput (Fig 4)",
    "shared_length": "Shared Prefix Length vs Decode Throughput (Fig 5)",
    "num_groups": "Number of Prefix Groups vs Decode Throughput (Fig 6)",
    "batch_size": "Batch Size vs Throughput: Homogeneous vs Heterogeneous (Figs 8-9)",
}


def visualize_sweep(path: Path, output: Path | None = None) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print("No rows in sweep CSV")
        return

    experiment = str(df["experiment"].iloc[0])
    x_name = str(df["x_name"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 6))
    for series, sdf in df.groupby("series"):
        sdf = sdf.sort_values("x_value")
        label = None if series == "main" else str(series)
        ax.plot(
            sdf["x_value"],
            sdf["decode_throughput_toks_s"],
            marker="o",
            label=label,
        )

    ax.set_xlabel(_X_LABELS.get(x_name, x_name))
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_title(_TITLES.get(experiment, f"{experiment}: throughput vs {x_name}"))
    ax.grid(True, alpha=0.3)
    if df["series"].nunique() > 1:
        ax.legend(title="Workload")

    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()
