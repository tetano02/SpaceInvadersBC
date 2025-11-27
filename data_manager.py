"""
Data Manager - Manages the saving of all project data.
Centralizes saving logic for demonstrations, models and plots.
"""
import pickle
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import random
import string
import csv


class DataManager:
    """Class to manage project data saving."""
    
    def __init__(self, base_dir='data'):
        """
        Initializes the DataManager.
        
        Args:
            base_dir: Base directory for all data (default: 'data')
        """
        self.base_dir = Path(base_dir)
        self.demonstrations_dir = self.base_dir / 'demonstrations'
        self.models_dir = self.base_dir / 'models'
        self.plots_dir = self.base_dir / 'plots'
        self.metrics_dir = self.base_dir / 'metrics'
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _generate_unique_id(self, length=5):
        """
        Generates a unique alphanumeric ID.
        
        Args:
            length: Length of the ID (default: 5)
        
        Returns:
            str: Unique alphanumeric ID
        """
        # Available characters: uppercase and lowercase letters + numbers
        characters = string.ascii_letters + string.digits
        
        # Generate ID until a unique one is found
        max_attempts = 1000
        for _ in range(max_attempts):
            new_id = ''.join(random.choices(characters, k=length))
            
            # Verify that it doesn't already exist in any directory
            if not self._id_exists(new_id):
                return new_id
        
        # If after many attempts no unique ID is found, increase the length
        return self._generate_unique_id(length + 1)
    
    def _id_exists(self, id_str):
        """
        Checks if an ID already exists in one of the saved files.
        
        Args:
            id_str: ID to verify
        
        Returns:
            bool: True if the ID already exists, False otherwise
        """
        # Search in demonstrations
        for file in self.demonstrations_dir.glob('dem_*_*_*.pkl'):
            if id_str in file.stem:
                return True
        
        # Search in models
        for file in self.models_dir.glob('mod_*_*_*.pth'):
            if id_str in file.stem:
                return True
        
        # Search in plots
        for file in self.plots_dir.glob('plot_*_*_*.png'):
            if id_str in file.stem:
                return True
        
        # Search in metrics CSV
        for file in self.metrics_dir.glob('metrics_*.csv'):
            if id_str in file.stem:
                return True
        
        return False

    def create_run_identifier(self):
        """Generates timestamp and ID for a new run."""
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_id = self._generate_unique_id()
        return timestamp, run_id

    def get_metrics_filepath(self, run_timestamp, run_id):
        """Constructs the path of the metrics CSV for the specified run."""
        filename = f"metrics_{run_timestamp}_{run_id}.csv"
        return self.metrics_dir / filename

    def _ensure_metrics_header(self, filepath):
        """Ensures the presence of the header in the CSV file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists() and filepath.stat().st_size > 0:
            return
        with open(filepath, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Section", "Name", "Value", "Extra1", "Extra2", "Extra3", "Extra4"])

    def save_training_csv(self, run_timestamp, run_id, metadata, epoch_metrics, target_path=None):
        """Saves run metadata and per-epoch metrics to CSV."""
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
        """Saves CSV with metadata and per-iteration metrics from GAIL training."""
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
        """Appends evaluation results to the existing CSV."""
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
        """Creates all necessary directories."""
        self.demonstrations_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def save_demonstrations(self, demonstrations, filename=None, source='manual', custom_id=None):
        """
        Saves demonstrations to file.
        
        Args:
            demonstrations: List of episodes with observations, actions, rewards, dones
            filename: Custom filename (optional)
            source: Data source (default: 'manual', can be 'minari:dataset_name')
            custom_id: Custom ID (optional, otherwise automatically generated)
        
        Returns:
            tuple: (Path of saved file, generated ID)
        """
        if not demonstrations:
            print("No demonstrations to save!")
            return None, None
        
        # Filename: dem_YYMMDD_hhmmss_id.pkl (5-character alphanumeric id)
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
        
        # Prepare data to save
        data = {
            'demonstrations': demonstrations,
            'num_episodes': len(demonstrations),
            'total_steps': sum(len(ep['actions']) for ep in demonstrations),
            'timestamp': datetime.now().isoformat(),
            'source': source
        }
        
        # Save data
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Demonstrations saved in: {filepath}")
        print(f"  Episodes: {data['num_episodes']}")
        print(f"  Total steps: {data['total_steps']}")
        print(f"  Source: {source}")
        
        return filepath, demo_id
    
    def load_demonstrations(self, filepath):
        """
        Loads demonstrations from file.
        
        Args:
            filepath: Path to the file to load
        
        Returns:
            list: List of demonstrations
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✓ Demonstrations loaded from: {filepath}")
        print(f"  Episodes: {data.get('num_episodes', 'N/A')}")
        print(f"  Total steps: {data.get('total_steps', 'N/A')}")
        print(f"  Source: {data.get('source', 'N/A')}")
        
        return data['demonstrations']
    
    def save_model(self, model_state_dict, optimizer_state_dict=None, 
                   train_losses=None, val_losses=None, 
                   train_accuracies=None, val_accuracies=None,
                   filename=None, custom_id=None, custom_timestamp=None,
                   metadata=None):
        """
        Saves a PyTorch model.
        
        Args:
            model_state_dict: Model state dict
            optimizer_state_dict: Optimizer state dict (optional)
            train_losses: List of losses during training (optional)
            val_losses: List of losses during validation (optional)
            train_accuracies: List of accuracies during training (optional)
            val_accuracies: List of accuracies during validation (optional)
            filename: Custom filename (optional, if specified ignores timestamp and id)
            custom_id: Custom ID (optional)
            custom_timestamp: Custom timestamp in yymmdd_HHMMSS format (optional)
        
        Returns:
            tuple: (Path of saved file, timestamp, ID)
        """
        # Filename: mod_YYMMDD_hhmmss_id.pth (5-character alphanumeric id)
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
            # If filename is specified, extract timestamp and id if possible
            timestamp = None
            model_id = None
        
        filepath = self.models_dir / filename
        
        # Prepare data to save
        checkpoint = {
            'model_state_dict': model_state_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optional data if provided
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
        
        # Save the model
        torch.save(checkpoint, filepath)
        
        print(f"\n✓ Model saved in: {filepath}")
        
        # Also return timestamp and id for synchronization with plots
        if filename is None or (timestamp is not None and model_id is not None):
            return filepath, timestamp, model_id
        else:
            return filepath, None, None
    
    def load_model(self, filepath, device='cpu'):
        """
        Loads a PyTorch model.
        
        Args:
            filepath: Path to the file to load
            device: Device to load the model on (default: 'cpu')
        
        Returns:
            dict: Checkpoint containing model_state_dict and other data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        print(f"\n✓ Model loaded from: {filepath}")
        
        return checkpoint
    
    def save_training_plot(self, train_losses, val_losses, 
                          train_accuracies, val_accuracies,
                          filename=None, model_timestamp=None, model_id=None,
                          metadata=None):
        """
        Saves a plot of training history.
        
        Args:
            train_losses: List of losses during training
            val_losses: List of losses during validation
            train_accuracies: List of accuracies during training
            val_accuracies: List of accuracies during validation
            filename: Custom filename (optional)
            model_timestamp: Timestamp of the associated model (for synchronization)
            model_id: ID of the associated model (for synchronization)
        
        Returns:
            Path: Path of the saved file
        """
        # Create the plot
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
        
        # Descriptive title with model, dataset, epochs and batch
        title = self._build_plot_title(metadata, len(train_losses))
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Filename: plot_YYMMDD_hhmmss_id.png (alphanumeric id, synchronized with model)
        if filename is None:
            if model_timestamp is not None and model_id is not None:
                # Use model timestamp and id for synchronization
                timestamp = model_timestamp
                plot_id = model_id
            else:
                # Generate new timestamp and id
                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                plot_id = self._generate_unique_id()
            
            filename = f"plot_{timestamp}_{plot_id}.png"
        
        filepath = self.plots_dir / filename
        
        # Save the plot as PNG
        plt.savefig(filepath)
        print(f"\n✓ Plot saved in: {filepath}")

        # Also save the entire figure (pickle .fig) for future editing
        figure_path = filepath.with_suffix('.fig')
        with open(figure_path, 'wb') as fig_file:
            pickle.dump(fig, fig_file)
        print(f"✓ Figure saved in: {figure_path}")
        
        # Show the plot
        plt.show()
        
        return filepath

    def _build_plot_title(self, metadata, observed_epochs):
        """Constructs the plot title with run details."""
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
        Finds the most recent demonstrations file.
        
        Returns:
            Path or None: Path to the most recent file, or None if not found
        """
        demo_files = list(self.demonstrations_dir.glob('demonstrations_*.pkl'))
        if not demo_files:
            return None
        return max(demo_files, key=lambda p: p.stat().st_mtime)
    
    def get_latest_model(self):
        """
        Finds the most recent model file.
        
        Returns:
            Path or None: Path to the most recent file, or None if not found
        """
        model_files = list(self.models_dir.glob('*.pth'))
        if not model_files:
            return None
        return max(model_files, key=lambda p: p.stat().st_mtime)
    
    def list_demonstrations(self):
        """
        Lists all available demonstration files.
        
        Returns:
            list: List of Paths to demonstration files
        """
        # Supports both new format (dem_*) and old format (demonstrations_*)
        new_format = list(self.demonstrations_dir.glob('dem_*.pkl'))
        old_format = list(self.demonstrations_dir.glob('demonstrations_*.pkl'))
        return sorted(new_format + old_format)
    
    def list_models(self):
        """
        Lists all available models.
        
        Returns:
            list: List of Paths to model files
        """
        # Supports both new format (mod_*) and old formats
        new_format = list(self.models_dir.glob('mod_*.pth'))
        old_format = list(self.models_dir.glob('bc_model_*.pth'))
        special_files = list(self.models_dir.glob('best_model.pth'))
        return sorted(new_format + old_format + special_files)
    
    def list_plots(self):
        """
        Lists all available plots.
        
        Returns:
            list: List of Paths to plot files
        """
        # Supports both new format (plot_*) and old format (training_history_*)
        new_format = list(self.plots_dir.glob('plot_*.png'))
        old_format = list(self.plots_dir.glob('training_history_*.png'))
        return sorted(new_format + old_format)
    
    def get_demonstrations_info(self, filepath):
        """
        Gets information about a demonstrations file without loading it completely.
        
        Args:
            filepath: Path to the file
        
        Returns:
            dict: Information about the file (num_episodes, total_steps, source, timestamp)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return {
            'num_episodes': data.get('num_episodes', 'N/A'),
            'total_steps': data.get('total_steps', 'N/A'),
            'source': data.get('source', 'N/A'),
            'timestamp': data.get('timestamp', 'N/A'),
            'filepath': filepath
        }
    
    def merge_demonstrations(self, demo_filepaths, output_filename=None):
        """
        Merges multiple demonstration files into a single file.
        Original files are not modified or deleted.
        
        Args:
            demo_filepaths: List of paths to demonstration files to merge
            output_filename: Output filename (optional, otherwise automatically generated)
        
        Returns:
            tuple: (Path of merged file, generated ID)
        """
        if not demo_filepaths:
            print("No files to merge!")
            return None, None
        
        all_demonstrations = []
        sources = []
        
        print(f"\n📂 Merging {len(demo_filepaths)} demonstration files...")
        
        # Load and combine all demonstrations
        for filepath in demo_filepaths:
            filepath = Path(filepath)
            if not filepath.exists():
                print(f"⚠ File not found: {filepath} - Skipped")
                continue
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            demonstrations = data['demonstrations']
            num_episodes = len(demonstrations)
            total_steps = sum(len(ep['actions']) for ep in demonstrations)
            source = data.get('source', 'unknown')
            
            all_demonstrations.extend(demonstrations)
            sources.append(source)
            
            print(f"  ✓ Loaded {filepath.name}: {num_episodes} episodes, {total_steps} steps (source: {source})")
        
        if not all_demonstrations:
            print("No valid demonstrations found!")
            return None, None
        
        # Create combined source string
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while maintaining order
        combined_source = f"merged:{'+'.join(unique_sources)}"
        
        # Save the merged file
        merged_filepath, merged_id = self.save_demonstrations(
            demonstrations=all_demonstrations,
            filename=output_filename,
            source=combined_source
        )
        
        print(f"\n✅ Merge completed!")
        print(f"  Original files kept: {len(demo_filepaths)}")
        print(f"  Total merged episodes: {len(all_demonstrations)}")
        
        return merged_filepath, merged_id
