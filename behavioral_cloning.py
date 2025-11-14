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
from data_manager import DataManager


class BCDataset(Dataset):
    """Dataset per Behavioral Cloning."""
    
    def __init__(self, demonstrations):
        """
        Args:
            demonstrations: Lista di episodi con observations e actions
        """
        self.observations = []
        self.actions = []
        
        # Estrai tutte le coppie (observation, action) da tutti gli episodi
        for episode in demonstrations:
            for obs, action in zip(episode['observations'], episode['actions']):
                self.observations.append(obs)
                self.actions.append(action)
        
        print(f"Dataset creato con {len(self.observations)} campioni")
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        # Normalizza osservazioni (immagini 0-255 -> 0-1)
        obs = torch.FloatTensor(obs).permute(2, 0, 1) / 255.0
        action = torch.LongTensor([action])
        
        return obs, action


class BCPolicy(nn.Module):
    """Rete neurale per Behavioral Cloning con immagini."""
    
    def __init__(self, num_actions=6):
        super(BCPolicy, self).__init__()
        
        # CNN per estrarre features dalle immagini
        self.cnn = nn.Sequential(
            # Input: 3 x 210 x 160
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 32 x 51 x 39
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 64 x 24 x 18
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # 64 x 22 x 16
        )
        
        # Calcola dimensione output CNN
        self.feature_size = 64 * 22 * 16
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        # Estrai features con CNN
        x = self.cnn(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classificazione azione
        x = self.fc(x)
        return x


class BCTrainer:
    """Trainer per Behavioral Cloning."""
    
    def __init__(self, policy, learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.policy = policy.to(device)
        self.device = device
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
        self.metrics_csv_path = self.data_manager.get_metrics_filepath(self.run_timestamp, self.run_id)
        self.run_metadata = {}
        self.run_start_time = None
        self.run_end_time = None
        
        print(f"Usando device: {device}")

    def configure_run(self, metadata):
        """Configura le informazioni della run (ambiente, dataset, ecc.)."""
        if metadata is None:
            metadata = {}
        self.run_metadata.update(metadata)
    
    def train_epoch(self, train_loader):
        """Addestra per una epoch."""
        self.policy.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for observations, actions in train_loader:
            observations = observations.to(self.device)
            actions = actions.squeeze().to(self.device)
            
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
                actions = actions.squeeze().to(self.device)
                
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
        
        best_val_loss = float('inf')
        
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
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Salva miglior modello
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETATO")
        print(f"Miglior validation loss: {best_val_loss:.4f}")
        print(f"{'='*50}\n")
        self.run_end_time = datetime.now()
        self._export_training_metrics(num_epochs)
        
        # Plot risultati (usa timestamp e id dell'ultimo salvataggio)
        self.plot_training_history()

    def _export_training_metrics(self, num_epochs):
        """Salva su CSV metadati e metriche della run."""
        if self.run_start_time is None or self.run_end_time is None:
            return
        duration = (self.run_end_time - self.run_start_time).total_seconds()
        demo_files = [Path(f).name if isinstance(f, (str, Path)) else str(f) for f in self.run_metadata.get('demonstration_files', [])]
        metadata_rows = {
            'run_id': f"{self.run_timestamp}_{self.run_id}",
            'run_timestamp': self.run_timestamp,
            'training_datetime': self.run_start_time.isoformat(),
            'training_end_datetime': self.run_end_time.isoformat(),
            'training_duration_seconds': round(duration, 2),
            'environment_name': self.run_metadata.get('environment_name', 'unknown'),
            'training_type': self.run_metadata.get('training_type', 'behavioral_cloning'),
            'num_epochs': num_epochs,
            'batch_size': self.run_metadata.get('batch_size'),
            'total_dataset_samples': self.run_metadata.get('total_samples'),
            'num_training_samples': self.run_metadata.get('train_samples'),
            'num_validation_samples': self.run_metadata.get('val_samples'),
            'val_split': self.run_metadata.get('val_split'),
            'num_demonstration_files': len(demo_files),
            'num_demonstrations': self.run_metadata.get('num_demonstrations'),
            'demonstration_files': ' | '.join(demo_files) if demo_files else 'N/A',
            'model_checkpoint': f"mod_{self.run_timestamp}_{self.run_id}.pth",
            'metrics_csv_path': str(self.metrics_csv_path)
        }
        epoch_metrics = []
        for idx in range(num_epochs):
            if idx >= len(self.train_losses):
                break
            epoch_metrics.append({
                'epoch': idx + 1,
                'train_loss': round(self.train_losses[idx], 6),
                'train_accuracy': round(self.train_accuracies[idx], 3),
                'val_loss': round(self.val_losses[idx], 6),
                'val_accuracy': round(self.val_accuracies[idx], 3)
            })
        self.metrics_csv_path = self.data_manager.save_training_csv(
            run_timestamp=self.run_timestamp,
            run_id=self.run_id,
            metadata=metadata_rows,
            epoch_metrics=epoch_metrics,
            target_path=self.metrics_csv_path
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
            model_id=self.last_save_id
        )
    
    def save_model(self, filename=None):
        """Salva il modello usando DataManager.
        Memorizza timestamp e ID per sincronizzazione con i plot."""
        metadata = {
            'run_timestamp': self.run_timestamp,
            'run_id': self.run_id,
            'metrics_csv_path': str(self.metrics_csv_path)
        }
        save_kwargs = dict(
            model_state_dict=self.policy.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_accuracies=self.train_accuracies,
            val_accuracies=self.val_accuracies,
            metadata=metadata
        )
        if filename is None:
            save_kwargs.update({
                'custom_timestamp': self.run_timestamp,
                'custom_id': self.run_id
            })
        else:
            save_kwargs['filename'] = filename
        filepath, timestamp, model_id = self.data_manager.save_model(**save_kwargs)
        
        # Salva timestamp e id per sincronizzazione con plot
        if timestamp is not None and model_id is not None:
            self.last_save_timestamp = timestamp
            self.last_save_id = model_id
        
        return filepath
    
    def load_model(self, filename):
        """Carica il modello usando DataManager."""
        checkpoint = self.data_manager.load_model(filename, device=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']


def load_demonstrations(filepath):
    """Carica dimostrazioni da file usando DataManager."""
    data_manager = DataManager()
    return data_manager.load_demonstrations(filepath)


def train_bc_model(demonstrations_file, num_epochs=50, batch_size=32, val_split=0.2):
    """Funzione principale per addestrare il modello BC."""
    # Carica dimostrazioni
    print(f"Caricamento dimostrazioni da: {demonstrations_file}")
    demonstrations = load_demonstrations(demonstrations_file)
    
    # Crea dataset
    full_dataset = BCDataset(demonstrations)
    
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
    policy = BCPolicy(num_actions=6)
    trainer = BCTrainer(policy)
    environment_name = 'ALE/SpaceInvaders-v5'
    trainer.configure_run({
        'environment_name': environment_name,
        'training_type': 'behavioral_cloning',
        'demonstration_files': [str(Path(demonstrations_file))],
        'num_demonstrations': len(demonstrations),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'val_split': val_split,
        'train_samples': train_size,
        'val_samples': val_size,
        'total_samples': len(full_dataset)
    })
    
    # Addestra
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Salva modello finale (senza filename, userà il nuovo formato automatico)
    trainer.save_model()
    
    return trainer.policy


def main():
    """Funzione principale."""
    # Trova file di dimostrazioni più recente
    demo_dir = Path('data/demonstrations')
    if not demo_dir.exists():
        print("Errore: Directory 'data/demonstrations' non trovata!")
        print("Esegui prima 'collect_demonstrations.py' per raccogliere dati.")
        return
    
    demo_files = list(demo_dir.glob('demonstrations_*.pkl'))
    if not demo_files:
        print("Errore: Nessun file di dimostrazioni trovato!")
        print("Esegui prima 'collect_demonstrations.py' per raccogliere dati.")
        return
    
    # Usa file più recente
    latest_demo_file = max(demo_files, key=lambda p: p.stat().st_mtime)
    print(f"Usando dimostrazioni da: {latest_demo_file}\n")
    
    # Parametri training
    num_epochs = int(input("Numero di epochs [default: 50]: ") or "50")
    batch_size = int(input("Batch size [default: 32]: ") or "32")
    
    # Addestra modello
    policy = train_bc_model(latest_demo_file, num_epochs=num_epochs, batch_size=batch_size)
    
    print("\nTraining completato! Usa 'evaluate_bc.py' per testare il modello.")


if __name__ == "__main__":
    main()
