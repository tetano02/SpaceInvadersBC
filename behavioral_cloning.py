"""
Implementazione di Behavioral Cloning per Space Invaders.
Addestra una rete neurale a imitare le dimostrazioni umane.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from utils import get_next_model_id, get_next_plot_id


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
        
        print(f"Usando device: {device}")
    
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
        
        # Plot risultati
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plotta loss e accuracy durante training."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Salva plot
        plot_dir = Path('data/plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_id = get_next_plot_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(plot_dir / f'training_history_{plot_id:03d}_{timestamp}.png')
        print(f"Plot salvato in: {plot_dir / f'training_history_{plot_id:03d}_{timestamp}.png'}")
        plt.show()
    
    def save_model(self, filename):
        """Salva il modello."""
        model_dir = Path('data/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = model_dir / filename
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        
        print(f"Modello salvato in: {filepath}")
    
    def load_model(self, filename):
        """Carica il modello."""
        filepath = Path('data/models') / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"Modello caricato da: {filepath}")


def load_demonstrations(filepath):
    """Carica dimostrazioni da file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['demonstrations']


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
    
    # Addestra
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Salva modello finale
    model_id = get_next_model_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_model(f'bc_model_{model_id:03d}_{timestamp}.pth')
    
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
