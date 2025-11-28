"""Plot grouped reward statistics per dataset and batch size."""

from __future__ import annotations

import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["Minari", "Expert", "Minari + Expert"]
MODEL_ORDER = ["CNN", "MLP", "VIT"]
MODEL_COLORS = {
    "CNN": "#f0a202",
    "MLP": "#5dade2",
    "VIT": "#17a589",
}


def _prepare_frame(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    df["Dataset"] = df["Dataset"].str.strip().str.title()
    df["Model"] = df["Model"].str.strip().str.upper()
    df["Batch Size"] = df["Batch Size"].astype(int)
    df.sort_values(["Dataset", "Model", "Batch Size"], inplace=True)
    return df


def _sorted_batch_sizes(df: pd.DataFrame) -> list[int]:
    sizes = sorted(df["Batch Size"].unique())
    return sizes


def _plot_dataset_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dataset: str,
    batch_sizes: Iterable[int],
    add_legend_labels: bool = False,
) -> None:
    dataset_df = df[df["Dataset"] == dataset]
    x_positions = np.arange(len(batch_sizes))
    bar_width = 0.25

    for offset, model in enumerate(MODEL_ORDER):
        model_df = (
            dataset_df[dataset_df["Model"] == model]
            .set_index("Batch Size")
            .reindex(batch_sizes)
        )

        mean = model_df["Mean reward"]
        std = model_df["Standard deviation"]
        min_vals = model_df["Reward min"]
        max_vals = model_df["Reward max"]

        positions = x_positions + (offset - 1) * bar_width
        ax.bar(
            positions,
            mean,
            width=bar_width,
            color=MODEL_COLORS[model],
            label=model if add_legend_labels else None,
            yerr=std,
            capsize=5,
            edgecolor="black",
            linewidth=0.8,
        )

        # Indicate min/max with short horizontal segments for quick reference.
        valid_min = ~min_vals.isna()
        ax.hlines(
            min_vals[valid_min],
            positions[valid_min] - bar_width / 2,
            positions[valid_min] + bar_width / 2,
            colors="red",
            linewidth=2,
        )

        valid_max = ~max_vals.isna()
        ax.hlines(
            max_vals[valid_max],
            positions[valid_max] - bar_width / 2,
            positions[valid_max] + bar_width / 2,
            colors="green",
            linewidth=2,
        )

    ax.set_title(dataset)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(size) for size in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_rewards(csv_path: str | pathlib.Path) -> None:
    csv_path = pathlib.Path(csv_path)
    df = _prepare_frame(csv_path)
    batches = _sorted_batch_sizes(df)

    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Combined plot
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(15, 5), sharey=True)

    for idx, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER)):
        _plot_dataset_panel(ax, df, dataset, batches, add_legend_labels=(idx == 0))

    axes[0].set_ylabel("Reward Medio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Reward distribution by dataset and batch size", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path = output_dir / "reward_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)

    # Individual plots for each dataset
    for dataset in DATASET_ORDER:
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        _plot_dataset_panel(ax_single, df, dataset, batches, add_legend_labels=True)

        ax_single.set_ylabel("Reward Medio")
        ax_single.legend(loc="upper right", frameon=True)
        ax_single.set_title(f"Reward distribution - {dataset}")
        fig_single.tight_layout()

        # Save individual plot
        filename = f"reward_distribution_{dataset.lower().replace(' + ', '_')}.png"
        output_path_single = output_dir / filename
        fig_single.savefig(output_path_single, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path_single}")

        plt.close(fig_single)


if __name__ == "__main__":
    default_csv = pathlib.Path(__file__).with_name("reward.csv")
    plot_rewards(default_csv)
