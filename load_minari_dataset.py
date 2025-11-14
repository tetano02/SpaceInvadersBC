"""
Script per caricare e verificare dataset esperti da Minari.
"""
import minari
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from utils import get_next_demonstration_id


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


def save_demonstrations(demonstrations, dataset_name):
    """Salva le dimostrazioni convertite."""
    data_dir = Path('data/demonstrations')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    demo_id = get_next_demonstration_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demonstrations_{demo_id:03d}_{timestamp}_minari_{dataset_name}.pkl"
    filepath = data_dir / filename
    
    data = {
        'demonstrations': demonstrations,
        'num_episodes': len(demonstrations),
        'total_steps': sum(len(ep['actions']) for ep in demonstrations),
        'timestamp': datetime.now().isoformat(),
        'source': f'minari:{dataset_name}'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Dimostrazioni salvate in: {filepath}")
    return filepath


def main():
    """Funzione principale."""
    print("\n" + "="*60)
    print("CARICAMENTO DATASET MINARI PER BEHAVIORAL CLONING")
    print("="*60)
    
    # Lista dataset disponibili
    local_datasets, remote_datasets = list_available_datasets()
    
    # Menu
    print("\n" + "-"*60)
    print("OPZIONI:")
    print("-"*60)
    print("1. Verifica dataset specifico")
    print("2. Scarica 'atari/spaceinvaders/expert-v0'")
    print("3. Carica e converti 'atari/spaceinvaders/expert-v0'")
    print("4. Scarica dataset custom")
    print("0. Esci")
    
    choice = input("\nScelta: ").strip()
    
    if choice == '1':
        dataset_id = input("\nInserisci ID dataset: ").strip()
        check_dataset_info(dataset_id)
    
    elif choice == '2':
        dataset_id = "atari/spaceinvaders/expert-v0"
        download_dataset(dataset_id)
    
    elif choice == '3':
        dataset_id = "atari/spaceinvaders/expert-v0"
        
        # Verifica se è già scaricato
        local_datasets, _ = list_available_datasets()
        if dataset_id not in local_datasets:
            print(f"\n⚠ Dataset non trovato localmente. Download in corso...")
            if not download_dataset(dataset_id):
                print("\n❌ Impossibile procedere senza il dataset.")
                return
        
        # Carica dataset
        dataset = check_dataset_info(dataset_id)
        if dataset is None:
            return
        
        # Chiedi quanti episodi convertire
        max_episodes = input(f"\nQuanti episodi convertire? [default: tutti ({dataset.total_episodes})]: ").strip()
        max_episodes = int(max_episodes) if max_episodes else None
        
        # Converti
        demonstrations = convert_minari_to_demonstrations(dataset, max_episodes)
        
        # Salva
        save_demonstrations(demonstrations, "spaceinvaders_expert")
        
        print("\n✓ Operazione completata! Puoi ora usare questi dati per il training.")
    
    elif choice == '4':
        dataset_id = input("\nInserisci ID dataset da scaricare: ").strip()
        download_dataset(dataset_id)
    
    elif choice == '0':
        print("\nArrivederci!")
        return
    
    else:
        print("\n⚠ Scelta non valida!")


if __name__ == "__main__":
    main()
