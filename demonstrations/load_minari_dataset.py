"""
Script to load and verify expert datasets from Minari.
"""

import os
from pathlib import Path

# Set a local default path for Minari datasets inside the repo.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MINARI_DIR = PROJECT_ROOT / "data" / "minari_datasets"
if not os.environ.get("MINARI_DATASETS_PATH"):
    DEFAULT_MINARI_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MINARI_DATASETS_PATH"] = str(DEFAULT_MINARI_DIR)

import minari
import numpy as np
from data_manager import DataManager


def list_available_datasets():
    """Lists all available datasets in Minari."""
    print("\n" + "=" * 60)
    print("AVAILABLE MINARI DATASETS")
    print("=" * 60)

    try:
        # List all locally available datasets
        local_datasets = minari.list_local_datasets()
        print(f"\nLocal datasets ({len(local_datasets)}):")
        for dataset_id in local_datasets:
            print(f"  - {dataset_id}")

        # List remote datasets available for Space Invaders
        print("\nRemote datasets available for Atari/Space Invaders:")
        remote_datasets = minari.list_remote_datasets()
        space_invaders_datasets = [
            d
            for d in remote_datasets
            if "spaceinvaders" in d.lower() or "space-invaders" in d.lower()
        ]

        if space_invaders_datasets:
            for dataset_id in space_invaders_datasets:
                print(f"  - {dataset_id}")
        else:
            print("  No specific dataset for Space Invaders found.")
            print("\nGeneric Atari datasets:")
            atari_datasets = [d for d in remote_datasets if "atari" in d.lower()]
            for dataset_id in atari_datasets[:10]:  # Show first 10
                print(f"  - {dataset_id}")

        return local_datasets, (
            space_invaders_datasets if space_invaders_datasets else atari_datasets
        )

    except Exception as e:
        print(f"Error retrieving datasets: {e}")
        return [], []


def check_dataset_info(dataset_id):
    """Checks information about a specific dataset."""
    print(f"\n" + "=" * 60)
    print(f"DATASET INFORMATION: {dataset_id}")
    print("=" * 60)

    try:
        # Try to load the dataset
        dataset = minari.load_dataset(dataset_id)

        print(f"\nID: {dataset.id}")
        print(f"Total episodes: {dataset.total_episodes}")
        print(f"Total steps: {dataset.total_steps}")

        # Information about spaces
        print(f"\nObservation space: {dataset.observation_space}")
        print(f"Action space: {dataset.action_space}")

        # Metadata if available
        if hasattr(dataset, "spec") and dataset.spec:
            print(f"\nMetadata:")
            if hasattr(dataset.spec, "env_spec"):
                print(f"  Environment: {dataset.spec.env_spec}")

        # Statistics on a sample episode
        print(f"\nLoading sample episode...")
        episode_data = dataset[0]

        # Correct access to episode data
        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards

        print(f"  Steps in episode 0: {len(actions)}")
        print(f"  Observation shape: {observations[0].shape}")
        print(
            f"  Actions shape: {actions.shape if hasattr(actions, 'shape') else len(actions)}"
        )
        print(f"  Total reward: {np.sum(rewards):.2f}")

        return dataset

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print(f"\nTry downloading the dataset with:")
        print(f"  minari.download_dataset('{dataset_id}')")
        return None


def download_dataset(dataset_id):
    """Downloads a dataset from Minari."""
    print(f"\n" + "=" * 60)
    print(f"DATASET DOWNLOAD: {dataset_id}")
    print("=" * 60)

    try:
        print("\nDownload in progress...")
        minari.download_dataset(dataset_id)
        print(f"✓ Dataset '{dataset_id}' downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def convert_minari_to_demonstrations(dataset, max_episodes=None):
    """Converts Minari dataset to the format used for BC."""
    print(f"\n" + "=" * 60)
    print("DATASET CONVERSION")
    print("=" * 60)

    demonstrations = []
    num_episodes = (
        dataset.total_episodes
        if max_episodes is None
        else min(max_episodes, dataset.total_episodes)
    )

    print(f"\nConverting {num_episodes} episodes...")

    for i in range(num_episodes):
        episode_data = dataset[i]

        # Correct access to Minari episode data
        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards

        # Convert to BC format
        demo_episode = {
            "observations": np.array(observations, dtype=np.uint8),
            "actions": np.array(actions, dtype=np.int8),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.zeros(len(actions), dtype=np.bool_),
        }
        # Mark the last step as done
        demo_episode["dones"][-1] = True

        demonstrations.append(demo_episode)

        if (i + 1) % 10 == 0:
            print(f"  Converted {i + 1}/{num_episodes} episodes...")

    print(f"\n✓ Conversion completed!")
    print(f"  Total episodes: {len(demonstrations)}")
    total_steps = sum(len(ep["actions"]) for ep in demonstrations)
    print(f"  Total steps: {total_steps}")

    return demonstrations


def save_demonstrations(
    demonstrations, dataset_name, source=None, custom_id=None, data_manager=None
):
    """Saves converted demonstrations using DataManager."""
    data_manager = data_manager or DataManager()
    resolved_source = source or f"minari_{dataset_name}"
    return data_manager.save_demonstrations(
        demonstrations=demonstrations,
        filename=None,
        source=resolved_source,
        custom_id=custom_id,
    )


def demonstration_exists(custom_id, data_manager=None):
    """Checks if demonstrations with the specified ID already exist."""
    data_manager = data_manager or DataManager()
    pattern = f"dem_*_{custom_id}.pkl"
    existing_files = sorted(data_manager.demonstrations_dir.glob(pattern))
    if existing_files:
        print(f"\n⚠ Demonstrations with ID '{custom_id}' already present:")
        for file in existing_files:
            print(f"  - {file}")
    return existing_files


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("LOADING MINARI DATASET FOR BEHAVIORAL CLONING")
    print("=" * 60)

    dataset_id = "atari/spaceinvaders/expert-v0"
    dataset_name = "spaceinvaders_expert"

    print(f"\nTarget dataset: {dataset_id}")
    local_datasets, _ = list_available_datasets()

    if dataset_id not in local_datasets:
        print(f"\n⚠ Dataset not found locally. Starting automatic download...")
        if not download_dataset(dataset_id):
            print("\n❌ Unable to complete the flow without the required dataset.")
            return
    else:
        print("\n✓ Dataset already available locally.")

    data_manager = DataManager()
    if demonstration_exists("minari", data_manager=data_manager):
        print("\n✓ Demonstrations already converted. Skipping reconversion.")
        return

    dataset = check_dataset_info(dataset_id)
    if dataset is None:
        print("\n❌ Unable to convert the dataset.")
        return

    demonstrations = convert_minari_to_demonstrations(dataset)
    save_path, _ = save_demonstrations(
        demonstrations,
        dataset_name,
        source="minari",
        custom_id="minari",
        data_manager=data_manager,
    )

    if save_path:
        print(f"\n✓ Demonstrations saved in: {save_path}")
    print("\nOperation completed! You can now use this data for training.")


if __name__ == "__main__":
    main()
