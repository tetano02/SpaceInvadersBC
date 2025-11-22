"""
Data Manager - Gestisce il salvataggio di tutti i dati del progetto.
Centralizza la logica di salvataggio per dimostrazioni, modelli e plot.
"""
import pickle
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import csv


class DataManager:
    """Classe per gestire il salvataggio dei dati del progetto."""
    
    def __init__(self, base_dir='data'):
        """
        Inizializza il DataManager.
        
        Args:
            base_dir: Directory base per tutti i dati (default: 'data')
        """
        self.base_dir = Path(base_dir)
        self.demonstrations_dir = self.base_dir / 'demonstrations'
        self.models_dir = self.base_dir / 'models'
        self.plots_dir = self.base_dir / 'plots'
        self.metrics_dir = self.base_dir / 'metrics'
        
        # Crea le directory se non esistono
        self._create_directories()
    
    def _generate_unique_id(self, length=5):
        """
        Genera un ID alfanumerico univoco.
        
        Args:
            length: Lunghezza dell'ID (default: 5)
        
        Returns:
            str: ID alfanumerico univoco
        """
        # Caratteri disponibili: lettere maiuscole e minuscole + numeri
        characters = string.ascii_letters + string.digits
        
        # Genera ID finché non ne trova uno univoco
        max_attempts = 1000
        for _ in range(max_attempts):
            new_id = ''.join(random.choices(characters, k=length))
            
            # Verifica che non esista già in nessuna directory
            if not self._id_exists(new_id):
                return new_id
        
        # Se dopo molti tentativi non trova un ID univoco, aumenta la lunghezza
        return self._generate_unique_id(length + 1)
    
    def _id_exists(self, id_str):
        """
        Verifica se un ID esiste già in uno dei file salvati.
        
        Args:
            id_str: ID da verificare
        
        Returns:
            bool: True se l'ID esiste già, False altrimenti
        """
        # Cerca in dimostrazioni
        for file in self.demonstrations_dir.glob('dem_*_*_*.pkl'):
            if id_str in file.stem:
                return True
        
        # Cerca in modelli
        for file in self.models_dir.glob('mod_*_*_*.pth'):
            if id_str in file.stem:
                return True
        
        # Cerca in plot
        for file in self.plots_dir.glob('plot_*_*_*.png'):
            if id_str in file.stem:
                return True
        
        # Cerca in CSV di metriche
        for file in self.metrics_dir.glob('metrics_*.csv'):
            if id_str in file.stem:
                return True
        
        return False

    def create_run_identifier(self):
        """Genera timestamp e ID per una nuova esecuzione."""
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_id = self._generate_unique_id()
        return timestamp, run_id

    def get_metrics_filepath(self, run_timestamp, run_id):
        """Costruisce il percorso del CSV di metriche per il run specificato."""
        filename = f"metrics_{run_timestamp}_{run_id}.csv"
        return self.metrics_dir / filename

    def _ensure_metrics_header(self, filepath):
        """Garantisce la presenza dell'header nel file CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists() and filepath.stat().st_size > 0:
            return
        with open(filepath, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Section", "Name", "Value", "Extra1", "Extra2", "Extra3", "Extra4"])

    def save_training_csv(self, run_timestamp, run_id, metadata, epoch_metrics, target_path=None):
        """Salva su CSV i metadati della run e le metriche per-epoch."""
        filepath = Path(target_path) if target_path else self.get_metrics_filepath(run_timestamp, run_id)
        self._ensure_metrics_header(filepath)
        with open(filepath, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in metadata.items():
                writer.writerow(["metadata", key, value, "", "", "", ""])
            writer.writerow([])
            writer.writerow(["section_header", "epoch_metrics", "Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
            for metrics in epoch_metrics:
                writer.writerow([
                    "epoch",
                    metrics.get('epoch'),
                    metrics.get('train_loss'),
                    metrics.get('train_accuracy'),
                    metrics.get('val_loss'),
                    metrics.get('val_accuracy'),
                    ""
                ])
        return filepath

    def save_gail_metrics_csv(self, run_timestamp, run_id, metadata, iteration_metrics, target_path=None):
        """Salva CSV con metadati e metriche per-iterazione del training GAIL."""
        filepath = Path(target_path) if target_path else self.get_metrics_filepath(run_timestamp, run_id)
        self._ensure_metrics_header(filepath)
        with open(filepath, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in metadata.items():
                writer.writerow(["gail_metadata", key, value, "", "", "", ""])
            writer.writerow([])
            writer.writerow([
                "section_header",
                "gail_metrics",
                "Iteration",
                "DiscLoss",
                "ExpertAcc",
                "AgentAcc",
                "PolicyLoss"
            ])
            for metrics in iteration_metrics:
                writer.writerow([
                    "gail_iteration",
                    metrics.get('iteration'),
                    round(metrics.get('disc_loss', 0.0), 6),
                    round(metrics.get('expert_acc', 0.0), 6),
                    round(metrics.get('agent_acc', 0.0), 6),
                    round(metrics.get('policy_loss', 0.0), 6),
                    ""
                ])
                writer.writerow([
                    "gail_iteration_details",
                    metrics.get('iteration'),
                    metrics.get('policy_mode'),
                    metrics.get('steps'),
                    metrics.get('epsilon'),
                    metrics.get('disc_updates'),
                    metrics.get('policy_updates')
                ])
                writer.writerow([
                    "gail_iteration_return",
                    metrics.get('iteration'),
                    round(metrics.get('mean_return', 0.0), 6),
                    metrics.get('gail_reward_mean'),
                    "",
                    "",
                    ""
                ])
            writer.writerow([])
        return filepath

    def append_evaluation_results(self, csv_path, evaluation_summary, episode_metrics, metadata=None):
        """Aggiunge i risultati della valutazione al CSV esistente."""
        csv_path = Path(csv_path)
        self._ensure_metrics_header(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([])
            if metadata:
                for key, value in metadata.items():
                    writer.writerow(["evaluation_metadata", key, value, "", "", "", ""])
            writer.writerow(["section_header", "evaluation_summary", "Metric", "Value", "", "", ""])
            for key, value in evaluation_summary.items():
                writer.writerow(["evaluation_summary", key, value, "", "", "", ""])
            writer.writerow(["section_header", "evaluation_episodes", "Episode", "Reward", "Steps", "", ""])
            for idx, episode in enumerate(episode_metrics, start=1):
                writer.writerow([
                    "evaluation_episode",
                    idx,
                    episode.get('reward'),
                    episode.get('steps'),
                    "",
                    "",
                    ""
                ])
        return csv_path
    
    def _create_directories(self):
        """Crea tutte le directory necessarie."""
        self.demonstrations_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def save_demonstrations(self, demonstrations, filename=None, source='manual', custom_id=None):
        """
        Salva le dimostrazioni su file.
        
        Args:
            demonstrations: Lista di episodi con observations, actions, rewards, dones
            filename: Nome file personalizzato (opzionale)
            source: Sorgente dei dati (default: 'manual', può essere 'minari:dataset_name')
            custom_id: ID personalizzato (opzionale, altrimenti generato automaticamente)
        
        Returns:
            tuple: (Path del file salvato, ID generato)
        """
        if not demonstrations:
            print("Nessuna dimostrazione da salvare!")
            return None, None
        
        # Nome file: dem_YYMMDD_hhmmss_id.pkl (id alfanumerico 5 caratteri)
        if filename is None:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            if custom_id is None:
                demo_id = self._generate_unique_id()
            else:
                demo_id = custom_id
            
            filename = f"dem_{timestamp}_{demo_id}.pkl"
        else:
            demo_id = None
        
        filepath = self.demonstrations_dir / filename
        
        # Prepara i dati da salvare
        data = {
            'demonstrations': demonstrations,
            'num_episodes': len(demonstrations),
            'total_steps': sum(len(ep['actions']) for ep in demonstrations),
            'timestamp': datetime.now().isoformat(),
            'source': source
        }
        
        # Salva i dati
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Dimostrazioni salvate in: {filepath}")
        print(f"  Episodi: {data['num_episodes']}")
        print(f"  Steps totali: {data['total_steps']}")
        print(f"  Sorgente: {source}")
        
        return filepath, demo_id
    
    def load_demonstrations(self, filepath):
        """
        Carica dimostrazioni da file.
        
        Args:
            filepath: Percorso del file da caricare
        
        Returns:
            list: Lista di dimostrazioni
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✓ Dimostrazioni caricate da: {filepath}")
        print(f"  Episodi: {data.get('num_episodes', 'N/A')}")
        print(f"  Steps totali: {data.get('total_steps', 'N/A')}")
        print(f"  Sorgente: {data.get('source', 'N/A')}")
        
        return data['demonstrations']
    
    def save_model(self, model_state_dict, optimizer_state_dict=None, 
                   train_losses=None, val_losses=None, 
                   train_accuracies=None, val_accuracies=None,
                   filename=None, custom_id=None, custom_timestamp=None,
                   metadata=None):
        """
        Salva un modello PyTorch.
        
        Args:
            model_state_dict: State dict del modello
            optimizer_state_dict: State dict dell'optimizer (opzionale)
            train_losses: Lista di loss durante training (opzionale)
            val_losses: Lista di loss durante validazione (opzionale)
            train_accuracies: Lista di accuracy durante training (opzionale)
            val_accuracies: Lista di accuracy durante validazione (opzionale)
            filename: Nome file personalizzato (opzionale, se specificato ignora timestamp e id)
            custom_id: ID personalizzato (opzionale)
            custom_timestamp: Timestamp personalizzato in formato yymmdd_HHMMSS (opzionale)
        
        Returns:
            tuple: (Path del file salvato, timestamp, ID)
        """
        # Nome file: mod_YYMMDD_hhmmss_id.pth (id alfanumerico 5 caratteri)
        if filename is None:
            if custom_timestamp is None:
                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            else:
                timestamp = custom_timestamp
            
            if custom_id is None:
                model_id = self._generate_unique_id()
            else:
                model_id = custom_id
            
            filename = f"mod_{timestamp}_{model_id}.pth"
        else:
            # Se filename è specificato, estrai timestamp e id se possibile
            timestamp = None
            model_id = None
        
        filepath = self.models_dir / filename
        
        # Prepara i dati da salvare
        checkpoint = {
            'model_state_dict': model_state_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        # Aggiungi dati opzionali se forniti
        if optimizer_state_dict is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state_dict
        if train_losses is not None:
            checkpoint['train_losses'] = train_losses
        if val_losses is not None:
            checkpoint['val_losses'] = val_losses
        if train_accuracies is not None:
            checkpoint['train_accuracies'] = train_accuracies
        if val_accuracies is not None:
            checkpoint['val_accuracies'] = val_accuracies
        if metadata is not None:
            checkpoint['run_metadata'] = metadata
            checkpoint['run_id'] = metadata.get('run_id')
            checkpoint['run_timestamp'] = metadata.get('run_timestamp')
            checkpoint['metrics_csv_path'] = metadata.get('metrics_csv_path')
        
        # Salva il modello
        torch.save(checkpoint, filepath)
        
        print(f"\n✓ Modello salvato in: {filepath}")
        
        # Restituisci anche timestamp e id per sincronizzare con i plot
        if filename is None or (timestamp is not None and model_id is not None):
            return filepath, timestamp, model_id
        else:
            return filepath, None, None
    
    def load_model(self, filepath, device='cpu'):
        """
        Carica un modello PyTorch.
        
        Args:
            filepath: Percorso del file da caricare
            device: Device su cui caricare il modello (default: 'cpu')
        
        Returns:
            dict: Checkpoint contenente model_state_dict e altri dati
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        print(f"\n✓ Modello caricato da: {filepath}")
        
        return checkpoint
    
    def save_training_plot(self, train_losses, val_losses, 
                          train_accuracies, val_accuracies,
                          filename=None, model_timestamp=None, model_id=None,
                          metadata=None):
        """
        Salva un plot della storia del training.
        
        Args:
            train_losses: Lista di loss durante training
            val_losses: Lista di loss durante validazione
            train_accuracies: Lista di accuracy durante training
            val_accuracies: Lista di accuracy durante validazione
            filename: Nome file personalizzato (opzionale)
            model_timestamp: Timestamp del modello associato (per sincronizzazione)
            model_id: ID del modello associato (per sincronizzazione)
        
        Returns:
            Path: Percorso del file salvato
        """
        # Crea il plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(train_accuracies, label='Train Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Titolo descrittivo con modello, dataset, epochs e batch
        title = self._build_plot_title(metadata, len(train_losses))
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Nome file: plot_YYMMDD_hhmmss_id.png (id alfanumerico, sincronizzato con modello)
        if filename is None:
            if model_timestamp is not None and model_id is not None:
                # Usa timestamp e id del modello per sincronizzazione
                timestamp = model_timestamp
                plot_id = model_id
            else:
                # Genera nuovi timestamp e id
                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                plot_id = self._generate_unique_id()
            
            filename = f"plot_{timestamp}_{plot_id}.png"
        
        filepath = self.plots_dir / filename
        
        # Salva il plot in PNG
        plt.savefig(filepath)
        print(f"\n✓ Plot salvato in: {filepath}")

        # Salva anche la figura intera (pickle .fig) per editing futuro
        figure_path = filepath.with_suffix('.fig')
        with open(figure_path, 'wb') as fig_file:
            pickle.dump(fig, fig_file)
        print(f"✓ Figura salvata in: {figure_path}")
        
        # Mostra il plot
        plt.show()
        
        return filepath

    def _build_plot_title(self, metadata, observed_epochs):
        """Costruisce il titolo del plot con dettagli della run."""
        metadata = metadata or {}

        model_name = metadata.get('model_label') or metadata.get('model_type') or 'Unknown model'
        dataset_label = metadata.get('training_dataset_name')

        if not dataset_label:
            demo_files = metadata.get('demonstration_files')
            if isinstance(demo_files, (list, tuple)) and demo_files:
                first = Path(demo_files[0]).name
                if len(demo_files) > 1:
                    dataset_label = f"{first} (+{len(demo_files) - 1})"
                else:
                    dataset_label = first

        if not dataset_label:
            dataset_label = metadata.get('environment_name')
        if not dataset_label:
            dataset_label = 'Unknown dataset'

        epoch_count = metadata.get('num_epochs') or observed_epochs or 'N/A'
        batch_size = metadata.get('batch_size') or metadata.get('train_batch_size') or 'N/A'
        frame_mode = metadata.get('frame_mode')
        if frame_mode == 'stacked':
            frame_label = 'Input: 2 frames'
        elif frame_mode == 'single':
            frame_label = 'Input: 1 frame'
        elif frame_mode:
            frame_label = f"Input: {frame_mode}"
        else:
            frame_label = None

        parts = []
        parts.append(f"Model: {model_name}")
        parts.append(f"Dataset: {dataset_label}")
        parts.append(f"Epochs: {epoch_count}")
        parts.append(f"Batch: {batch_size}")
        if frame_label:
            parts.append(frame_label)

        return " | ".join(parts)
    
    def get_latest_demonstrations(self):
        """
        Trova il file di dimostrazioni più recente.
        
        Returns:
            Path o None: Percorso del file più recente, o None se non trovato
        """
        demo_files = list(self.demonstrations_dir.glob('demonstrations_*.pkl'))
        if not demo_files:
            return None
        return max(demo_files, key=lambda p: p.stat().st_mtime)
    
    def get_latest_model(self):
        """
        Trova il file di modello più recente.
        
        Returns:
            Path o None: Percorso del file più recente, o None se non trovato
        """
        model_files = list(self.models_dir.glob('*.pth'))
        if not model_files:
            return None
        return max(model_files, key=lambda p: p.stat().st_mtime)
    
    def list_demonstrations(self):
        """
        Lista tutti i file di dimostrazioni disponibili.
        
        Returns:
            list: Lista di Path ai file di dimostrazioni
        """
        # Supporta sia il nuovo formato (dem_*) che il vecchio (demonstrations_*)
        new_format = list(self.demonstrations_dir.glob('dem_*.pkl'))
        old_format = list(self.demonstrations_dir.glob('demonstrations_*.pkl'))
        return sorted(new_format + old_format)
    
    def list_models(self):
        """
        Lista tutti i modelli disponibili.
        
        Returns:
            list: Lista di Path ai file dei modelli
        """
        # Supporta sia il nuovo formato (mod_*) che i vecchi formati
        new_format = list(self.models_dir.glob('mod_*.pth'))
        old_format = list(self.models_dir.glob('bc_model_*.pth'))
        special_files = list(self.models_dir.glob('best_model.pth'))
        return sorted(new_format + old_format + special_files)
    
    def list_plots(self):
        """
        Lista tutti i plot disponibili.
        
        Returns:
            list: Lista di Path ai file dei plot
        """
        # Supporta sia il nuovo formato (plot_*) che il vecchio (training_history_*)
        new_format = list(self.plots_dir.glob('plot_*.png'))
        old_format = list(self.plots_dir.glob('training_history_*.png'))
        return sorted(new_format + old_format)
    
    def get_demonstrations_info(self, filepath):
        """
        Ottiene informazioni su un file di dimostrazioni senza caricarlo completamente.
        
        Args:
            filepath: Percorso del file
        
        Returns:
            dict: Informazioni sul file (num_episodes, total_steps, source, timestamp)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return {
            'num_episodes': data.get('num_episodes', 'N/A'),
            'total_steps': data.get('total_steps', 'N/A'),
            'source': data.get('source', 'N/A'),
            'timestamp': data.get('timestamp', 'N/A'),
            'filepath': filepath
        }
