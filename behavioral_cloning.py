"""
Implementazione di Behavioral Cloning per Space Invaders.
Addestra una rete neurale a imitare le dimostrazioni umane.
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
    """Dataset per Behavioral Cloning."""

    def __init__(self, demonstrations, frame_mode: str = "single"):
        """Inizializza il dataset.

        Args:
            demonstrations: lista di episodi con observations e actions
            frame_mode: "single" per usare un solo frame, "stacked" per concatenare
                frame precedente e corrente lungo il canale
        """
        self.frame_mode = (frame_mode or "single").lower()
        if self.frame_mode not in {"single", "stacked"}:
            raise ValueError("frame_mode deve essere 'single' o 'stacked'")

        self.samples = []

        # Estrai tutte le tuple (prev_obs, obs, action) mantenendo la sequenza temporale
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

        print(f"Dataset creato con {len(self.samples)} campioni (frame_mode={self.frame_mode})")

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
    """Rete neurale stile DQN (CNN + MLP) per Behavioral Cloning."""

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
    """Multi-layer perceptron che lavora sull'immagine flattenata."""

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
    """Vision Transformer compatto per osservazioni Atari."""

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
            raise ValueError("patch_size deve dividere esattamente image_size per altezza e larghezza")

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
        "description": "CNN con tre conv e testa fully-connected (default)",
        "builder": lambda num_actions, **kwargs: BCPolicy(num_actions=num_actions, **kwargs),
    },
    "mlp": {
        "label": "Multi-Layer Perceptron",
        "description": "Rete fully-connected su osservazione flattenata",
        "builder": lambda num_actions, **kwargs: BCMLPPolicy(num_actions=num_actions, **kwargs),
    },
    "vit": {
        "label": "Vision Transformer",
        "description": "Transformer compatto con patch embedding (più esigente in VRAM)",
        "builder": lambda num_actions, **kwargs: BCVisionTransformer(num_actions=num_actions, **kwargs),
    },
}

DEFAULT_MODEL_TYPE = "dqn"


def get_available_model_types():
    """Restituisce il catalogo dei modelli registrati."""
    return MODEL_REGISTRY


def build_policy(model_type: str, num_actions: int = 6, **model_kwargs):
    """Istanzia il modello richiesto, propagando eventuali kwargs."""
    key = (model_type or DEFAULT_MODEL_TYPE).lower()
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_type}' non supportato. Opzioni: {valid}")
    return spec["builder"](num_actions=num_actions, **model_kwargs)


def prompt_model_type(default: str = DEFAULT_MODEL_TYPE):
    """Permette all'utente di scegliere il tipo di modello da usare."""
    options = list(MODEL_REGISTRY.items())
    print("\n=== SELEZIONE MODELLO BC ===")
    for idx, (key, spec) in enumerate(options, start=1):
        default_tag = " (default)" if key == default else ""
        print(f"{idx}. {spec['label']} [{key}]{default_tag}\n   {spec['description']}")
    print("Scegli inserendo il numero o premi ENTER per il default.")

    while True:
        choice = input("Modello: ").strip()
        if not choice:
            return default
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        print("Input non valido. Riprova.")


def prompt_frame_mode(default: str = "single"):
    """Chiede se usare un solo frame o due frame concatenati."""
    choices = [("single", "Frame singolo (3 canali)"), ("stacked", "Due frame consecutivi (6 canali)")]
    print("\n=== SELEZIONE TIPO DI INPUT ===")
    for idx, (key, label) in enumerate(choices, start=1):
        default_tag = " (default)" if key == default else ""
        print(f"{idx}. {label}{default_tag}")

    while True:
        choice = input("Input desiderato: ").strip()
        if not choice:
            return default
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        print("Input non valido. Riprova.")


class BCTrainer:
    """Trainer per Behavioral Cloning."""

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

        # Inizializza DataManager
        self.data_manager = DataManager()

        # Memorizza timestamp e id per sincronizzazione con plot
        self.last_save_timestamp = None
        self.last_save_id = None

        # Informazioni della run
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

        print(f"Usando device: {device}")

    def configure_run(self, metadata):
        """Configura le informazioni della run (ambiente, dataset, ecc.)."""
        if metadata is None:
            metadata = {}
        self.run_metadata.update(metadata)
        self.run_metadata.setdefault("model_type", self.model_type)
        self.run_metadata.setdefault(
            "model_label", MODEL_REGISTRY.get(self.model_type, {}).get("label", self.model_type)
        )

    def train_epoch(self, train_loader):
        """Addestra per una epoch."""
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

            # Statistiche
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
        """Valida il modello."""
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
        """Addestra il modello per multiple epochs."""
        print(f"\n{'='*50}")
        print("INIZIO TRAINING")
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

            # Stampa progresso
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Salva miglior modello
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")

        print(f"\n{'='*50}")
        print("TRAINING COMPLETATO")
        print(f"Miglior validation loss: {best_val_loss:.4f}")
        print(f"{'='*50}\n")
        self.run_end_time = datetime.now()
        # Salva il modello finale con lo stesso timestamp/id della run
        self.save_model()

        self._export_training_metrics(num_epochs)

        # Plot risultati (usa timestamp e id dell'ultimo salvataggio)
        self.plot_training_history()

    def _export_training_metrics(self, num_epochs):
        """Salva su CSV metadati e metriche della run."""
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
        """Plotta loss e accuracy durante training usando DataManager.
        Usa timestamp e ID dell'ultimo modello salvato per sincronizzazione."""
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
        """Salva il modello usando DataManager.
        Memorizza timestamp e ID per sincronizzazione con i plot."""
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

        # Salva timestamp e id per sincronizzazione con plot
        if timestamp is not None and model_id is not None:
            self.last_save_timestamp = timestamp
            self.last_save_id = model_id

        return filepath

    def load_model(self, filename):
        """Carica il modello usando DataManager."""
        checkpoint = self.data_manager.load_model(filename, device=self.device)

        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
            self.train_accuracies = checkpoint["train_accuracies"]
            self.val_accuracies = checkpoint["val_accuracies"]


def load_demonstrations(filepath):
    """Carica dimostrazioni da file usando DataManager."""
    data_manager = DataManager()
    return data_manager.load_demonstrations(filepath)


def train_bc_model(
    demonstrations_file,
    num_epochs=50,
    batch_size=32,
    val_split=0.2,
    model_type=DEFAULT_MODEL_TYPE,
    frame_mode: str = "single",
):
    """Funzione principale per addestrare il modello BC.

    Args:
        demonstrations_files: lista di percorsi ai file di dimostrazioni
        num_epochs: numero di epoche di training
        batch_size: dimensione batch per i DataLoader
        val_split: frazione dedicata alla validation
        device: dispositivo su cui addestrare
        model_type: chiave nel MODEL_REGISTRY
        frame_mode: "single" (3 canali) oppure "stacked" (due frame concatenati)
    """
    print("Caricamento dimostrazioni da:")
    demonstrations = []
    for demo_file in demonstrations_files:
        print(f"  - {demo_file}")
        demonstrations.extend(load_demonstrations(demo_file))

    # Crea dataset
    full_dataset = BCDataset(demonstrations, frame_mode=frame_mode)

    # Split train/validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Crea dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Crea modello e trainer
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

    # Addestra (il trainer salverà il modello finale automaticamente)
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    return trainer.policy


def main(selected_model_type=None):
    """Funzione principale."""
    # Trova file di dimostrazioni più recente
    demo_dir = Path("data/demonstrations")
    if not demo_dir.exists():
        print("Errore: Directory 'data/demonstrations' non trovata!")
        print("Esegui prima 'collect_demonstrations.py' per raccogliere dati.")
        return

    demo_files = list(demo_dir.glob("dem_*.pkl"))
    if not demo_files:
        print("Errore: Nessun file di dimostrazioni trovato!")
        print("Esegui prima 'collect_demonstrations.py' per raccogliere dati.")
        return

    # Usa file più recente
    latest_demo_file = min(demo_files, key=lambda p: p.stat().st_mtime)
    print(f"Usando dimostrazioni da: {latest_demo_file}\n")

    model_type = selected_model_type or prompt_model_type()
    frame_mode = prompt_frame_mode()

    # Parametri training
    num_epochs = int(input("Numero di epochs [default: 50]: ") or "50")
    batch_size = int(input("Batch size [default: 32]: ") or "32")

    # Addestra modello
    policy = train_bc_model(
        latest_demo_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_type=model_type,
        frame_mode=frame_mode,
    )

    print("\nTraining completato! Usa 'evaluate_bc.py' per testare il modello.")


if __name__ == "__main__":
    main()