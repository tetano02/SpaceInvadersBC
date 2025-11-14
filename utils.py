"""
Utility functions per il progetto di Behavioral Cloning.
"""
from pathlib import Path
import re


def get_next_id(directory, pattern):
    """
    Ottiene il prossimo ID incrementale per i file in una directory.
    
    Args:
        directory: Path alla directory
        pattern: Pattern regex per estrarre l'ID dai nomi file
    
    Returns:
        int: Prossimo ID disponibile
    """
    directory = Path(directory)
    if not directory.exists():
        return 1
    
    max_id = 0
    for file in directory.iterdir():
        match = re.search(pattern, file.name)
        if match:
            file_id = int(match.group(1))
            max_id = max(max_id, file_id)
    
    return max_id + 1


def get_next_demonstration_id():
    """Ottiene il prossimo ID per le dimostrazioni."""
    return get_next_id('data/demonstrations', r'demonstrations_(\d+)_')


def get_next_model_id():
    """Ottiene il prossimo ID per i modelli."""
    return get_next_id('data/models', r'bc_model_(\d+)_')


def get_next_plot_id():
    """Ottiene il prossimo ID per i plot."""
    return get_next_id('data/plots', r'training_history_(\d+)_')
