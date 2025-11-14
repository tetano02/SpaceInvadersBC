"""
Script per caricare e verificare dataset esperti da Minari.
"""
import minari
import numpy as np
from pathlib import Path
from data_manager import DataManager


def list_available_datasets():
    """Elenca tutti i dataset disponibili in Minari."""
    print("\n" + "="*60)
    print("DATASET MINARI DISPONIBILI")
    print("="*60)
    
    try:
        # Lista tutti i dataset disponibili localmente
        local_datasets = minari.list_local_datasets()
        print(f"\nDataset locali ({len(local_datasets)}):")
        for dataset_id in local_datasets:
            print(f"  - {dataset_id}")
        
        # Lista dataset remoti disponibili per Space Invaders
        print("\nDataset remoti disponibili per Atari/Space Invaders:")
        remote_datasets = minari.list_remote_datasets()
        space_invaders_datasets = [d for d in remote_datasets if 'spaceinvaders' in d.lower() or 'space-invaders' in d.lower()]
        
        if space_invaders_datasets:
            for dataset_id in space_invaders_datasets:
                print(f"  - {dataset_id}")
        else:
            print("  Nessun dataset specifico per Space Invaders trovato.")
            print("\nDataset Atari generici:")
            atari_datasets = [d for d in remote_datasets if 'atari' in d.lower()]
            for dataset_id in atari_datasets[:10]:  # Mostra primi 10
                print(f"  - {dataset_id}")
        
        return local_datasets, space_invaders_datasets if space_invaders_datasets else atari_datasets
    
    except Exception as e:
        print(f"Errore nel recupero dei dataset: {e}")
        return [], []


def check_dataset_info(dataset_id):
    """Verifica informazioni su un dataset specifico."""
    print(f"\n" + "="*60)
    print(f"INFORMAZIONI DATASET: {dataset_id}")
    print("="*60)
    
    try:
        # Prova a caricare il dataset
        dataset = minari.load_dataset(dataset_id)
        
        print(f"\nID: {dataset.id}")
        print(f"Totale episodi: {dataset.total_episodes}")
        print(f"Totale steps: {dataset.total_steps}")
        
        # Informazioni sugli spazi
        print(f"\nObservation space: {dataset.observation_space}")
        print(f"Action space: {dataset.action_space}")
        
        # Metadata se disponibili
        if hasattr(dataset, 'spec') and dataset.spec:
            print(f"\nMetadata:")
            if hasattr(dataset.spec, 'env_spec'):
                print(f"  Environment: {dataset.spec.env_spec}")
        
        # Statistiche su un episodio campione
        print(f"\nCaricamento episodio campione...")
        episode_data = dataset[0]
        
        # Accesso corretto ai dati dell'episodio
        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards
        
        print(f"  Steps nell'episodio 0: {len(actions)}")
        print(f"  Observation shape: {observations[0].shape}")
        print(f"  Actions shape: {actions.shape if hasattr(actions, 'shape') else len(actions)}")
        print(f"  Reward totale: {np.sum(rewards):.2f}")
        
        return dataset
    
    except Exception as e:
        print(f"\n❌ Errore nel caricamento del dataset: {e}")
        print(f"\nProva a scaricare il dataset con:")
        print(f"  minari.download_dataset('{dataset_id}')")
        return None


def download_dataset(dataset_id):
    """Scarica un dataset da Minari."""
    print(f"\n" + "="*60)
    print(f"DOWNLOAD DATASET: {dataset_id}")
    print("="*60)
    
    try:
        print("\nDownload in corso...")
        minari.download_dataset(dataset_id)
        print(f"✓ Dataset '{dataset_id}' scaricato con successo!")
        return True
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        return False


def convert_minari_to_demonstrations(dataset, max_episodes=None):
    """Converte dataset Minari nel formato usato per BC."""
    print(f"\n" + "="*60)
    print("CONVERSIONE DATASET")
    print("="*60)
    
    demonstrations = []
    num_episodes = dataset.total_episodes if max_episodes is None else min(max_episodes, dataset.total_episodes)
    
    print(f"\nConversione di {num_episodes} episodi...")
    
    for i in range(num_episodes):
        episode_data = dataset[i]
        
        # Accesso corretto ai dati dell'episodio Minari
        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards
        
        # Converti nel formato BC
        demo_episode = {
            'observations': np.array(observations, dtype=np.uint8),
            'actions': np.array(actions, dtype=np.int8),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.zeros(len(actions), dtype=np.bool_)
        }
        # Marca l'ultimo step come done
        demo_episode['dones'][-1] = True
        
        demonstrations.append(demo_episode)
        
        if (i + 1) % 10 == 0:
            print(f"  Convertiti {i + 1}/{num_episodes} episodi...")
    
    print(f"\n✓ Conversione completata!")
    print(f"  Episodi totali: {len(demonstrations)}")
    total_steps = sum(len(ep['actions']) for ep in demonstrations)
    print(f"  Steps totali: {total_steps}")
    
    return demonstrations


def save_demonstrations(demonstrations, dataset_name, source=None, custom_id=None, data_manager=None):
    """Salva le dimostrazioni convertite usando DataManager."""
    data_manager = data_manager or DataManager()
    resolved_source = source or f'minari_{dataset_name}'
    return data_manager.save_demonstrations(
        demonstrations=demonstrations,
        filename=None,
        source=resolved_source,
        custom_id=custom_id
    )


def demonstration_exists(custom_id, data_manager=None):
    """Verifica se esistono già dimostrazioni salvate con l'ID specificato."""
    data_manager = data_manager or DataManager()
    pattern = f"dem_*_{custom_id}.pkl"
    existing_files = sorted(data_manager.demonstrations_dir.glob(pattern))
    if existing_files:
        print(f"\n⚠ Dimostrazioni con ID '{custom_id}' già presenti:")
        for file in existing_files:
            print(f"  - {file}")
    return existing_files


def main():
    """Funzione principale."""
    print("\n" + "="*60)
    print("CARICAMENTO DATASET MINARI PER BEHAVIORAL CLONING")
    print("="*60)
    
    dataset_id = "atari/spaceinvaders/expert-v0"
    dataset_name = "spaceinvaders_expert"
    
    print(f"\nTarget dataset: {dataset_id}")
    local_datasets, _ = list_available_datasets()
    
    if dataset_id not in local_datasets:
        print(f"\n⚠ Dataset non trovato localmente. Avvio download automatico...")
        if not download_dataset(dataset_id):
            print("\n❌ Impossibile completare il flusso senza il dataset richiesto.")
            return
    else:
        print("\n✓ Dataset già disponibile localmente.")
    
    data_manager = DataManager()
    if demonstration_exists('minari', data_manager=data_manager):
        print("\n✓ Dimostrazioni già convertite. Salto la riconversione.")
        return

    dataset = check_dataset_info(dataset_id)
    if dataset is None:
        print("\n❌ Impossibile convertire il dataset.")
        return
    
    demonstrations = convert_minari_to_demonstrations(dataset)
    save_path, _ = save_demonstrations(
        demonstrations,
        dataset_name,
        source='minari',
        custom_id='minari',
        data_manager=data_manager
    )
    
    if save_path:
        print(f"\n✓ Dimostrazioni salvate in: {save_path}")
    print("\nOperazione completata! Puoi ora usare questi dati per il training.")


if __name__ == "__main__":
    main()
