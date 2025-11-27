"""Analyzes demonstration files by counting how many times each action is executed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from data_manager import DataManager

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}


def prompt_for_demo_file(data_manager: DataManager) -> Path | None:
    files = data_manager.list_demonstrations()
    if not files:
        print("\n⚠ No demonstrations found in data/demonstrations")
        return None

    if len(files) == 1:
        print(f"\nI will analyze the file: {files[0]}")
        return files[0]

    print("\nAvailable demonstrations:")
    for idx, file in enumerate(files, start=1):
        info = data_manager.get_demonstrations_info(file)
        print(
            f"  [{idx}] {file.name} | episodi: {info['num_episodes']} | steps: {info['total_steps']} | source: {info['source']}"
        )

    while True:
        choice = input("Select the file number to analyze: ").strip()
        if not choice.isdigit():
            print("Enter a valid number")
            continue
        choice_idx = int(choice)
        if 1 <= choice_idx <= len(files):
            return files[choice_idx - 1]
        print("Index out of range, try again")


def determine_action_dim(demonstrations: Iterable[dict]) -> int:
    max_action = 0
    for episode in demonstrations:
        actions = episode.get("actions")
        if actions is None or len(actions) == 0:
            continue
        max_action = max(max_action, int(np.max(actions)))
    return max_action + 1


def action_label(action_id: int) -> str:
    return ACTION_NAMES.get(action_id, f"Action {action_id}")


def print_episode_stats(episode_idx: int, counts: np.ndarray, reward: float = 0.0):
    total = int(np.sum(counts))
    print(f"\nEpisode {episode_idx + 1}: {total} steps | Reward: {reward}")
    for action_id, count in enumerate(counts):
        percentage = (count / total * 100) if total else 0.0
        print(
            f"  - {action_id:>2} ({action_label(action_id):<10}): {count:>5}  ({percentage:5.1f}%)"
        )


def print_global_stats(
    total_counts: np.ndarray, total_reward: float = 0.0, avg_reward: float = 0.0
):
    total_steps = int(np.sum(total_counts))
    print("\n===== TOTAL SUMMARY =====")
    print(f"Total steps: {total_steps}")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    for action_id, count in enumerate(total_counts):
        percentage = (count / total_steps * 100) if total_steps else 0.0
        print(
            f"  - {action_id:>2} ({action_label(action_id):<10}): {count:>6}  ({percentage:5.1f}%)"
        )


def analyze_demonstrations_file(
    filepath: Path,
    *,
    summary_only: bool = False,
    data_manager: DataManager | None = None,
):
    """Performs analysis of a single demonstration file."""
    data_manager = data_manager or DataManager()
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    demonstrations = data_manager.load_demonstrations(filepath)
    print(f"\nAnalyzing demonstrations from: {filepath}")
    num_episodes = len(demonstrations)
    print(f"Episodes present: {num_episodes}")

    action_dim = determine_action_dim(demonstrations)
    episode_counts: list[np.ndarray] = []
    episode_rewards: list[float] = []
    for episode in demonstrations:
        actions = episode.get("actions", np.array([], dtype=np.int32))
        actions = np.asarray(actions, dtype=np.int64)
        counts = np.bincount(actions, minlength=action_dim)
        episode_counts.append(counts)

        # Calculate total reward for the episode
        rewards = episode.get("rewards", np.array([], dtype=np.float32))
        total_reward = float(np.sum(rewards))
        episode_rewards.append(total_reward)

    total_counts = np.sum(np.vstack(episode_counts), axis=0)
    total_reward = sum(episode_rewards)
    avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0.0

    if not summary_only:
        for idx, (counts, reward) in enumerate(zip(episode_counts, episode_rewards)):
            print_episode_stats(idx, counts, reward)

    print_global_stats(total_counts, total_reward, avg_reward)

    return {
        "episode_counts": episode_counts,
        "total_counts": total_counts,
        "num_episodes": num_episodes,
        "action_dim": action_dim,
    }


def main() -> int:
    data_manager = DataManager()

    selected = prompt_for_demo_file(data_manager)

    if selected is None:
        return 1

    if not selected.exists():
        print(f"\n⚠ File not found: {selected}")
        return 1

    analyze_demonstrations_file(
        selected,
        summary_only=False,
        data_manager=data_manager,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
