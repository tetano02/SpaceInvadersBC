"""Analizza i file di dimostrazioni contando quante volte viene eseguita ogni azione."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conta la frequenza delle azioni presenti nelle dimostrazioni salvate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file", type=str, help="Percorso del file di dimostrazioni da analizzare"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Mostra solo il riepilogo aggregato (senza dettaglio per episodio)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Esporta le statistiche in formato CSV (specifica il percorso)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Mostra l'elenco dei file disponibili e termina",
    )
    return parser.parse_args()


def prompt_for_demo_file(data_manager: DataManager) -> Path | None:
    files = data_manager.list_demonstrations()
    if not files:
        print("\n⚠ Nessuna dimostrazione trovata in data/demonstrations")
        return None

    if len(files) == 1:
        print(f"\nAnalizzerò il file: {files[0]}")
        return files[0]

    print("\nDimostrazioni disponibili:")
    for idx, file in enumerate(files, start=1):
        info = data_manager.get_demonstrations_info(file)
        print(
            f"  [{idx}] {file.name} | episodi: {info['num_episodes']} | steps: {info['total_steps']} | source: {info['source']}"
        )

    while True:
        choice = input("Seleziona il numero del file da analizzare: ").strip()
        if not choice.isdigit():
            print("Inserisci un numero valido")
            continue
        choice_idx = int(choice)
        if 1 <= choice_idx <= len(files):
            return files[choice_idx - 1]
        print("Indice fuori range, riprova")


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
    print(f"\nEpisodio {episode_idx + 1}: {total} passi | Reward: {reward}")
    for action_id, count in enumerate(counts):
        percentage = (count / total * 100) if total else 0.0
        print(
            f"  - {action_id:>2} ({action_label(action_id):<10}): {count:>5}  ({percentage:5.1f}%)"
        )


def print_global_stats(
    total_counts: np.ndarray, total_reward: float = 0.0, avg_reward: float = 0.0
):
    total_steps = int(np.sum(total_counts))
    print("\n===== RIEPILOGO TOTALE =====")
    print(f"Steps complessivi: {total_steps}")
    print(f"Reward totale: {total_reward}")
    print(f"Reward medio per episodio: {avg_reward:.2f}")
    for action_id, count in enumerate(total_counts):
        percentage = (count / total_steps * 100) if total_steps else 0.0
        print(
            f"  - {action_id:>2} ({action_label(action_id):<10}): {count:>6}  ({percentage:5.1f}%)"
        )


def export_csv(path: Path, episode_counts: list[np.ndarray], total_counts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "action_id", "action_name", "count", "percentage"])
        for idx, counts in enumerate(episode_counts, start=1):
            total = int(np.sum(counts)) or 1
            for action_id, count in enumerate(counts):
                percentage = count / total * 100
                writer.writerow(
                    [
                        idx,
                        action_id,
                        action_label(action_id),
                        count,
                        f"{percentage:.4f}",
                    ]
                )
        grand_total = int(np.sum(total_counts)) or 1
        for action_id, count in enumerate(total_counts):
            percentage = count / grand_total * 100
            writer.writerow(
                [
                    "TOTAL",
                    action_id,
                    action_label(action_id),
                    count,
                    f"{percentage:.4f}",
                ]
            )
    print(f"\n✓ Statistiche esportate in {path}")


def analyze_demonstrations_file(
    filepath: Path,
    *,
    summary_only: bool = False,
    csv_path: Path | None = None,
    data_manager: DataManager | None = None,
):
    """Esegue l'analisi di un singolo file di dimostrazioni."""
    data_manager = data_manager or DataManager()
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")

    demonstrations = data_manager.load_demonstrations(filepath)
    print(f"\nAnalisi dimostrazioni da: {filepath}")
    num_episodes = len(demonstrations)
    print(f"Episodi presenti: {num_episodes}")

    action_dim = determine_action_dim(demonstrations)
    episode_counts: list[np.ndarray] = []
    episode_rewards: list[float] = []
    for episode in demonstrations:
        actions = episode.get("actions", np.array([], dtype=np.int32))
        actions = np.asarray(actions, dtype=np.int64)
        counts = np.bincount(actions, minlength=action_dim)
        episode_counts.append(counts)

        # Calcola il reward totale dell'episodio
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

    if csv_path:
        export_csv(csv_path, episode_counts, total_counts)

    return {
        "episode_counts": episode_counts,
        "total_counts": total_counts,
        "num_episodes": num_episodes,
        "action_dim": action_dim,
    }


def main() -> int:
    args = parse_args()
    data_manager = DataManager()

    if args.list:
        files = data_manager.list_demonstrations()
        if not files:
            print("\nNessuna dimostrazione trovata.")
            return 0
        print("\nFile disponibili:")
        for file in files:
            info = data_manager.get_demonstrations_info(file)
            print(
                f"- {file.name} | episodi: {info['num_episodes']} | steps: {info['total_steps']} | source: {info['source']}"
            )
        return 0

    if args.file:
        selected = Path(args.file)
    else:
        selected = prompt_for_demo_file(data_manager)

    if selected is None:
        return 1

    if not selected.exists():
        print(f"\n⚠ File non trovato: {selected}")
        return 1

    csv_path = Path(args.csv) if args.csv else None
    analyze_demonstrations_file(
        selected,
        summary_only=args.summary_only,
        csv_path=csv_path,
        data_manager=data_manager,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
