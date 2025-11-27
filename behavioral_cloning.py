"""
Behavioral Cloning implementation for Space Invaders.
Trains a neural network to imitate human demonstrations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import math
from data_manager import DataManager


class BCDataset(Dataset):
    """Dataset for Behavioral Cloning."""

    def __init__(self, demonstrations, frame_mode: str = "single"):
        """Initializes the dataset.

        Args:
            demonstrations: list of episodes with observations and actions
            frame_mode: "single" to use a single frame, "stacked" to concatenate
                previous and current frame along the channel
        """
        self.frame_mode = (frame_mode or "single").lower()
        if self.frame_mode not in {"single", "stacked"}:
            raise ValueError("frame_mode must be 'single' or 'stacked'")

        self.samples = []

        # Extract all (prev_obs, obs, action) tuples maintaining temporal sequence
        for episode in demonstrations:
            prev_obs = None
            for obs, action in zip(episode["observations"], episode["actions"]):
                if prev_obs is None:
                    prev_obs = obs
                self.samples.append({
                    "current": obs,
                    "previous": prev_obs,
                    "action": action,
                })
                prev_obs = obs

        print(f"Dataset created with {len(self.samples)} samples (frame_mode={self.frame_mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        current = torch.FloatTensor(sample["current"]).permute(2, 0, 1) / 255.0
        action = torch.LongTensor([sample["action"]])

        if self.frame_mode == "stacked":
            previous = torch.FloatTensor(sample["previous"]).permute(2, 0, 1) / 255.0
            obs = torch.cat([previous, current], dim=0)
        else:
            obs = current

        return obs, action


class BCPolicy(nn.Module):
    """DQN-style neural network (CNN + MLP) for Behavioral Cloning."""

    def __init__(self, num_actions=6, in_channels=3):
        super(BCPolicy, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.feature_size = 64 * 22 * 16

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BCMLPPolicy(nn.Module):
    """Multi-layer perceptron working on flattened image."""

    def __init__(self, num_actions=6, in_channels=3):
        super().__init__()
        input_dim = in_channels * 210 * 160
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x)


class BCVisionTransformer(nn.Module):
    """Compact Vision Transformer for Atari observations."""

    def __init__(
        self,
        num_actions=6,
        in_channels=3,
        image_size=(210, 160),
        patch_size=(14, 16),
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError("patch_size must divide image_size exactly for height and width")

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_actions)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dqn": {
        "label": "DQN CNN",
        "description": "CNN with three conv layers and fully-connected head (default)",
        "builder": lambda num_actions, **kwargs: BCPolicy(num_actions=num_actions, **kwargs),
    },
    "mlp": {
        "label": "Multi-Layer Perceptron",
        "description": "Fully-connected network on flattened observation",
        "builder": lambda num_actions, **kwargs: BCMLPPolicy(num_actions=num_actions, **kwargs),
    },
    "vit": {
        "label": "Vision Transformer",
        "description": "Compact Transformer with patch embedding (more demanding on VRAM)",
        "builder": lambda num_actions, **kwargs: BCVisionTransformer(num_actions=num_actions, **kwargs),
    },
}

DEFAULT_MODEL_TYPE = "dqn"


def get_available_model_types():
    """Returns the catalog of registered models."""
    return MODEL_REGISTRY


def build_policy(model_type: str, num_actions: int = 6, **model_kwargs):
    """Instantiates the requested model, propagating any kwargs."""
    key = (model_type or DEFAULT_MODEL_TYPE).lower()
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_type}' not supported. Options: {valid}")
    return spec["builder"](num_actions=num_actions, **model_kwargs)


def prompt_model_type(default: str = DEFAULT_MODEL_TYPE):
    """Allows the user to choose the model type to use."""
    options = list(MODEL_REGISTRY.items())
    print("\n=== BC MODEL SELECTION ===")
    for idx, (key, spec) in enumerate(options, start=1):
        default_tag = " (default)" if key == default else ""
        print(f"{idx}. {spec['label']} [{key}]{default_tag}\n   {spec['description']}")
    print("Choose by entering the number or press ENTER for default.")

    while True:
        choice = input("Modello: ").strip()
        if not choice:
            return default
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        print("Invalid input. Try again.")


def prompt_frame_mode(default: str = "single"):
    """Asks whether to use a single frame or two concatenated frames."""
    choices = [("single", "Single frame (3 channels)"), ("stacked", "Two consecutive frames (6 channels)")]
    print("\n=== INPUT TYPE SELECTION ===")
    for idx, (key, label) in enumerate(choices, start=1):
        default_tag = " (default)" if key == default else ""
        print(f"{idx}. {label}{default_tag}")

    while True:
        choice = input("Desired input: ").strip()
        if not choice:
            return default
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        print("Invalid input. Try again.")


class BCTrainer:
    """Trainer for Behavioral Cloning."""

    def __init__(
        self,
        policy,
        model_type=DEFAULT_MODEL_TYPE,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.model_type = (model_type or DEFAULT_MODEL_TYPE).lower()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Initialize DataManager
        self.data_manager = DataManager()

        # Store timestamp and id for synchronization with plots
        self.last_save_timestamp = None
        self.last_save_id = None

        # Run information
        self.run_timestamp, self.run_id = self.data_manager.create_run_identifier()
        self.metrics_csv_path = self.data_manager.get_metrics_filepath(
            self.run_timestamp, self.run_id
        )
        self.run_metadata = {"model_type": self.model_type}
        self.run_metadata["model_label"] = MODEL_REGISTRY.get(self.model_type, {}).get(
            "label", self.model_type
        )
        self.run_start_time = None
        self.run_end_time = None

        print(f"Using device: {device}")

    def configure_run(self, metadata):
        """Configures run information (environment, dataset, etc.)."""
        if metadata is None:
            metadata = {}
        self.run_metadata.update(metadata)
        self.run_metadata.setdefault("model_type", self.model_type)
        self.run_metadata.setdefault(
            "model_label", MODEL_REGISTRY.get(self.model_type, {}).get("label", self.model_type)
        )

    def train_epoch(self, train_loader):
        """Trains for one epoch."""
        self.policy.train()
        total_loss = 0
        correct = 0
        total = 0
        total_batches = len(train_loader)
        progress_interval = max(1, math.ceil(total_batches * 0.05)) if total_batches else 1

        for batch_idx, (observations, actions) in enumerate(train_loader):
            observations = observations.to(self.device)
            actions = actions.squeeze(-1).to(self.device)

            # Forward pass
            predictions = self.policy(observations)
            loss = self.criterion(predictions, actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

            if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                model_tag = self.model_type.upper()
                print(
                    f"[{model_tag}][Batch {batch_idx + 1}/{total_batches}] "
                    f"Loss: {loss.item():.4f} | Avg: {avg_loss_so_far:.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validates the model."""
        self.policy.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for observations, actions in val_loader:
                observations = observations.to(self.device)
                actions = actions.squeeze(-1).to(self.device)

                predictions = self.policy(observations)
                loss = self.criterion(predictions, actions)

                total_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs=50):
        """Trains the model for multiple epochs."""
        print(f"\n{'='*50}")
        print("TRAINING START")
        print(f"{'='*50}")
        self.run_start_time = datetime.now()

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Print progress
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")

        print(f"\n{'='*50}")
        print("TRAINING COMPLETED")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*50}\n")
        self.run_end_time = datetime.now()
        # Save final model with the same timestamp/id as the run
        self.save_model()

        self._export_training_metrics(num_epochs)

        # Plot results (uses timestamp and id of last save)
        self.plot_training_history()

    def _export_training_metrics(self, num_epochs):
        """Saves run metadata and metrics to CSV."""
        if self.run_start_time is None or self.run_end_time is None:
            return
        duration = (self.run_end_time - self.run_start_time).total_seconds()
        demo_files = [
            Path(f).name if isinstance(f, (str, Path)) else str(f)
            for f in self.run_metadata.get("demonstration_files", [])
        ]
        metadata_rows = {
            "run_id": f"{self.run_timestamp}_{self.run_id}",
            "run_timestamp": self.run_timestamp,
            "training_datetime": self.run_start_time.isoformat(),
            "training_end_datetime": self.run_end_time.isoformat(),
            "training_duration_seconds": round(duration, 2),
            "environment_name": self.run_metadata.get("environment_name", "unknown"),
            "training_type": self.run_metadata.get(
                "training_type", "behavioral_cloning"
            ),
            "model_type": self.run_metadata.get("model_type", self.model_type),
            "num_epochs": num_epochs,
            "batch_size": self.run_metadata.get("batch_size"),
            "total_dataset_samples": self.run_metadata.get("total_samples"),
            "num_training_samples": self.run_metadata.get("train_samples"),
            "num_validation_samples": self.run_metadata.get("val_samples"),
            "val_split": self.run_metadata.get("val_split"),
            "num_demonstration_files": len(demo_files),
            "num_demonstrations": self.run_metadata.get("num_demonstrations"),
            "demonstration_files": " | ".join(demo_files) if demo_files else "N/A",
            "model_checkpoint": f"mod_{self.run_timestamp}_{self.run_id}.pth",
            "metrics_csv_path": str(self.metrics_csv_path),
        }
        epoch_metrics = []
        for idx in range(num_epochs):
            if idx >= len(self.train_losses):
                break
            epoch_metrics.append(
                {
                    "epoch": idx + 1,
                    "train_loss": round(self.train_losses[idx], 6),
                    "train_accuracy": round(self.train_accuracies[idx], 3),
                    "val_loss": round(self.val_losses[idx], 6),
                    "val_accuracy": round(self.val_accuracies[idx], 3),
                }
            )
        self.metrics_csv_path = self.data_manager.save_training_csv(
            run_timestamp=self.run_timestamp,
            run_id=self.run_id,
            metadata=metadata_rows,
            epoch_metrics=epoch_metrics,
            target_path=self.metrics_csv_path,
        )

    def plot_training_history(self):
        """Plots loss and accuracy during training using DataManager.
        Uses timestamp and ID of last saved model for synchronization."""
        return self.data_manager.save_training_plot(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_accuracies=self.train_accuracies,
            val_accuracies=self.val_accuracies,
            model_timestamp=self.last_save_timestamp,
            model_id=self.last_save_id,
            metadata=self.run_metadata,
        )

    def save_model(self, filename=None):
        """Saves the model using DataManager.
        Stores timestamp and ID for synchronization with plots."""
        metadata = {
            "run_timestamp": self.run_timestamp,
            "run_id": self.run_id,
            "metrics_csv_path": str(self.metrics_csv_path),
            "model_type": self.run_metadata.get("model_type", self.model_type),
        }
        save_kwargs = dict(
            model_state_dict=self.policy.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_accuracies=self.train_accuracies,
            val_accuracies=self.val_accuracies,
            metadata=metadata,
        )
        if filename is None:
            save_kwargs.update(
                {"custom_timestamp": self.run_timestamp, "custom_id": self.run_id}
            )
        else:
            save_kwargs["filename"] = filename
        filepath, timestamp, model_id = self.data_manager.save_model(**save_kwargs)

        # Save timestamp and id for synchronization with plots
        if timestamp is not None and model_id is not None:
            self.last_save_timestamp = timestamp
            self.last_save_id = model_id

        return filepath

    def load_model(self, filename):
        """Loads the model using DataManager."""
        checkpoint = self.data_manager.load_model(filename, device=self.device)

        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
            self.train_accuracies = checkpoint["train_accuracies"]
            self.val_accuracies = checkpoint["val_accuracies"]


def get_available_devices():
    """Returns a list of available devices with their names."""
    devices = []
    
    # CPU is always available
    devices.append({"name": "CPU", "device": "cpu"})
    
    # Check GPU availability
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            devices.append({"name": f"GPU {i}: {gpu_name}", "device": f"cuda:{i}"})
    
    return devices


def select_device():
    """Asks the user to select a device for training."""
    devices = get_available_devices()
    
    print("\nAvailable devices:")
    for idx, device_info in enumerate(devices, 1):
        print(f"  {idx}. {device_info['name']}")
    
    while True:
        choice = input(f"\nSelect device (1-{len(devices)}) [default: 1]: ").strip()
        
        # Default to CPU (option 1)
        if not choice:
            choice = "1"
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(devices):
                selected_device = devices[choice_idx]
                print(f"Selected device: {selected_device['name']}")
                return selected_device['device']
            else:
                print(f"⚠ Invalid choice! Select a number between 1 and {len(devices)}.")
        except ValueError:
            print(f"⚠ Invalid input! Enter a number between 1 and {len(devices)}.")


def load_demonstrations(filepath):
    """Loads demonstrations from file using DataManager."""
    data_manager = DataManager()
    return data_manager.load_demonstrations(filepath)


def select_demonstration_files(demo_files):
    """Allows the user to choose one or more demonstration files."""
    sorted_files = sorted(demo_files, key=lambda p: p.stat().st_mtime, reverse=True)

    print("\nAvailable demonstration files:")
    for idx, demo_path in enumerate(sorted_files, 1):
        timestamp = datetime.fromtimestamp(demo_path.stat().st_mtime)
        size_mb = demo_path.stat().st_size / (1024 * 1024)
        print(
            f"  {idx}. {demo_path.name}"
            f" | modified on {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            f" | {size_mb:.2f} MB"
        )

    print(
        "\nYou can enter a single number (e.g. 1) or multiple numbers separated by comma"
        " (e.g. 1,3,4) to aggregate demonstrations."
    )

    while True:
        raw_choice = input(
            f"Select file (1-{len(sorted_files)}) [default: 1]: "
        ).strip()
        if not raw_choice:
            raw_choice = "1"

        parts = [part.strip() for part in raw_choice.split(",") if part.strip()]
        indices = []
        try:
            for part in parts:
                idx = int(part) - 1
                if 0 <= idx < len(sorted_files):
                    indices.append(idx)
                else:
                    raise ValueError
        except ValueError:
            print(
                f"⚠ Invalid input! Enter numbers between 1 and {len(sorted_files)}"
                " separated by comma."
            )
            continue

        unique_indices = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        if not unique_indices:
            print("⚠ No valid index detected, try again.")
            continue

        selected = [sorted_files[idx] for idx in unique_indices]
        if len(selected) == 1:
            print(f"You will use the file: {selected[0]}\n")
        else:
            print("You will use the files:")
            for path in selected:
                print(f"  - {path}")
            print()
        return selected


def train_bc_model(
    demonstrations_files,
    num_epochs=50,
    batch_size=32,
    val_split=0.2,
    device=None,
    model_type=DEFAULT_MODEL_TYPE,
    frame_mode: str = "single",
):
    """Main function to train the BC model.

    Args:
        demonstrations_files: list of paths to demonstration files
        num_epochs: number of training epochs
        batch_size: batch size for DataLoaders
        val_split: fraction dedicated to validation
        device: device to train on
        model_type: key in MODEL_REGISTRY
        frame_mode: "single" (3 channels) or "stacked" (two concatenated frames)
    """
    print("Loading demonstrations from:")
    demonstrations = []
    for demo_file in demonstrations_files:
        print(f"  - {demo_file}")
        demonstrations.extend(load_demonstrations(demo_file))

    # Create dataset
    full_dataset = BCDataset(demonstrations, frame_mode=frame_mode)

    # Split train/validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model and trainer
    input_channels = 6 if frame_mode == "stacked" else 3
    policy = build_policy(model_type=model_type, num_actions=6, in_channels=input_channels)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = BCTrainer(policy, model_type=model_type, device=device)
    environment_name = "ALE/SpaceInvaders-v5"
    demo_file_paths = [str(Path(path)) for path in demonstrations_files]
    dataset_label = Path(demonstrations_files[0]).name if demonstrations_files else "Unknown"
    if len(demonstrations_files) > 1:
        dataset_label = f"{dataset_label} (+{len(demonstrations_files) - 1})"

    trainer.configure_run(
        {
            "environment_name": environment_name,
            "training_type": "behavioral_cloning",
            "model_type": model_type,
            "frame_mode": frame_mode,
            "demonstration_files": demo_file_paths,
            "training_dataset_name": dataset_label,
            "num_demonstrations": len(demonstrations),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "val_split": val_split,
            "train_samples": train_size,
            "val_samples": val_size,
            "total_samples": len(full_dataset),
        }
    )

    # Train (the trainer will save the final model automatically)
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    return trainer.policy


def main(selected_model_type=None):
    """Main function."""
    # Find most recent demonstration file
    demo_dir = Path("data/demonstrations")
    if not demo_dir.exists():
        print("Error: Directory 'data/demonstrations' not found!")
        print("Run 'collect_demonstrations.py' first to collect data.")
        return

    demo_files = list(demo_dir.glob("dem_*.pkl"))
    if not demo_files:
        print("Error: No demonstration files found!")
        print("Run 'collect_demonstrations.py' first to collect data.")
        return

    # Allow user to choose one or more files to use (default: most recent)
    selected_demo_files = select_demonstration_files(demo_files)
    print("You will use the following demonstrations:")
    for path in selected_demo_files:
        print(f"  - {path}")
    print()

    model_type = selected_model_type or prompt_model_type()
    frame_mode = prompt_frame_mode()

    # Training parameters
    num_epochs = int(input("Number of epochs [default: 50]: ") or "50")
    batch_size = int(input("Batch size [default: 32]: ") or "32")
    
    # Device selection
    device = select_device()

    # Train model
    train_bc_model(
        selected_demo_files,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        model_type=model_type,
        frame_mode=frame_mode,
    )

    print("\nTraining completed! Use 'evaluate_bc.py' to test the model.")

if __name__ == "__main__":
    main()