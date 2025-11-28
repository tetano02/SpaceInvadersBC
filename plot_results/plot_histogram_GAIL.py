"""Plot grouped reward statistics before and after GAIL training per dataset."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["Minari", "Expert", "Minari + Expert"]
MODEL_ORDER = ["CNN", "MLP", "VIT"]
PHASE_COLORS = {
    "Before GAIL": "#f5c16c",  # orange
    "After GAIL": "#85c1e9",  # light blue
}


def _prepare_frame(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df["Dataset"] = df["Dataset"].str.strip()
    df["Model"] = df["Model"].str.strip().str.upper()
    df.sort_values(["Dataset", "Model"], inplace=True)
    return df


def _plot_dataset_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dataset: str,
    add_legend_labels: bool = False,
) -> None:
    dataset_df = df[df["Dataset"] == dataset]
    x_positions = np.arange(len(MODEL_ORDER))
    bar_width = 0.35

    for idx, model in enumerate(MODEL_ORDER):
        model_df = dataset_df[dataset_df["Model"] == model]

        if model_df.empty:
            continue

        # Get before and after GAIL data
        before_mean = model_df["Reward_mean_before_GAIL"].values[0]
        before_std = model_df["Std_dev_before_GAIL"].values[0]
        after_mean = model_df["Reward_mean_after_GAIL"].values[0]
        after_std = model_df["Std_dev_after_GAIL"].values[0]

        # Plot before GAIL (left bar)
        ax.bar(
            x_positions[idx] - bar_width / 2,
            before_mean,
            width=bar_width,
            color=PHASE_COLORS["Before GAIL"],
            label="Before GAIL" if add_legend_labels and idx == 0 else None,
            yerr=before_std,
            capsize=5,
            edgecolor="black",
            linewidth=0.8,
        )

        # Plot after GAIL (right bar)
        ax.bar(
            x_positions[idx] + bar_width / 2,
            after_mean,
            width=bar_width,
            color=PHASE_COLORS["After GAIL"],
            label="After GAIL" if add_legend_labels and idx == 0 else None,
            yerr=after_std,
            capsize=5,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_title(dataset)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_rewards(csv_path: str | pathlib.Path) -> None:
    csv_path = pathlib.Path(csv_path)
    df = _prepare_frame(csv_path)

    output_dir = pathlib.Path(__file__).parent / "plots" / "gail"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined plot
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(15, 5), sharey=True)

    for idx, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER)):
        _plot_dataset_panel(ax, df, dataset, add_legend_labels=(idx == 0))

    axes[0].set_ylabel("Mean Reward")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path = output_dir / "reward_gail.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)

    # Individual plots for each dataset
    for dataset in DATASET_ORDER:
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        _plot_dataset_panel(ax_single, df, dataset, add_legend_labels=True)

        ax_single.set_ylabel("Mean Reward")
        ax_single.legend(loc="upper right", frameon=True)
        ax_single.set_title(f"Reward comparison - {dataset}")
        fig_single.tight_layout()

        # Save individual plot
        filename_map = {
            "Minari": "mina",
            "Expert": "exp",
            "Minari + Expert": "mina_exp",
        }
        filename = f"reward_gail_{filename_map[dataset]}.png"
        output_path_single = output_dir / filename
        fig_single.savefig(output_path_single, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path_single}")

        plt.close(fig_single)


if __name__ == "__main__":
    default_csv = pathlib.Path(__file__).parent / "data_training" / "reward_GAIL.csv"
    plot_rewards(default_csv)
