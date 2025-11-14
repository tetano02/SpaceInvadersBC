"""
Script per raccogliere dimostrazioni umane giocando manualmente a Space Invaders.
Usa i tasti freccia per muoverti e SPAZIO per sparare.
"""
import gymnasium as gym
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import pygame
from env_make import make_space_invaders_env
from utils import get_next_demonstration_id


class DemonstrationCollector:
    """Raccoglie dimostrazioni umane per Behavioral Cloning."""
    
    def __init__(self, env):
        self.env = env
        self.demonstrations = []
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        # Mappatura tasti -> azioni Space Invaders
        # Azioni: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
        self.key_to_action = {
            pygame.K_SPACE: 1,      # FIRE
            pygame.K_RIGHT: 2,      # RIGHT
            pygame.K_LEFT: 3,       # LEFT
        }
        
    def get_human_action(self):
        """Ottiene l'azione dall'input della tastiera."""
        keys = pygame.key.get_pressed()
        
        # Combinazioni di tasti
        if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
            return 4  # RIGHTFIRE
        elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
            return 5  # LEFTFIRE
        elif keys[pygame.K_RIGHT]:
            return 2  # RIGHT
        elif keys[pygame.K_LEFT]:
            return 3  # LEFT
        elif keys[pygame.K_SPACE]:
            return 1  # FIRE
        else:
            return 0  # NOOP
    
    def collect_episode(self):
        """Raccoglie una singola partita."""
        observation, info = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        print("\n=== NUOVA PARTITA ===")
        print("Controlli:")
        print("  ← →  : Muovi")
        print("  SPAZIO : Spara")
        print("  ESC : Termina raccolta")
        print("  Q : Abbandona partita (non salva)")
        print("=====================\n")
        
        while not (done or truncated):
            # Processa eventi Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, total_reward
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False, total_reward
                    if event.key == pygame.K_q:
                        print("Partita abbandonata (non salvata)")
                        return True, total_reward
            
            # Ottieni azione umana (legge i tasti ad ogni frame)
            action = self.get_human_action()
            
            # Salva osservazione e azione (salva TUTTI i frame)
            self.current_episode['observations'].append(np.copy(observation))
            self.current_episode['actions'].append(action)
            
            # Esegui azione
            observation, reward, done, truncated, info = self.env.step(action)
            
            # Salva reward e done
            self.current_episode['rewards'].append(reward)
            self.current_episode['dones'].append(done or truncated)
            
            total_reward += reward
        
        # Converti liste in array numpy per efficienza
        self.current_episode['observations'] = np.array(self.current_episode['observations'], dtype=np.uint8)
        self.current_episode['actions'] = np.array(self.current_episode['actions'], dtype=np.int8)
        self.current_episode['rewards'] = np.array(self.current_episode['rewards'], dtype=np.float32)
        self.current_episode['dones'] = np.array(self.current_episode['dones'], dtype=np.bool_)
        
        # Salva l'episodio completato
        self.demonstrations.append(self.current_episode.copy())
        print(f"\nPartita completata! Reward totale: {total_reward}")
        print(f"Passi registrati: {len(self.current_episode['actions'])}")
        
        return True, total_reward
    
    def collect_multiple_episodes(self, num_episodes=5):
        """Raccoglie multiple partite."""
        print(f"\n{'='*50}")
        print(f"RACCOLTA DIMOSTRAZIONI - {num_episodes} partite")
        print(f"{'='*50}")
        
        episode_rewards = []
        
        for i in range(num_episodes):
            print(f"\nPartita {i+1}/{num_episodes}")
            continue_collection, reward = self.collect_episode()
            
            if not continue_collection:
                print("\nRaccolta interrotta dall'utente.")
                break
            
            episode_rewards.append(reward)
            
            if i < num_episodes - 1:
                print("\nPremi un tasto per la prossima partita...")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                            waiting = False
                            break
        
        print(f"\n{'='*50}")
        print("RIEPILOGO RACCOLTA")
        print(f"{'='*50}")
        print(f"Partite completate: {len(self.demonstrations)}")
        if episode_rewards:
            print(f"Reward medio: {np.mean(episode_rewards):.2f}")
            print(f"Reward min/max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
            total_steps = sum(len(ep['actions']) for ep in self.demonstrations)
            print(f"Passi totali registrati: {total_steps}")
        
        return self.demonstrations
    
    def save_demonstrations(self, filename=None):
        """Salva le dimostrazioni su file."""
        if not self.demonstrations:
            print("Nessuna dimostrazione da salvare!")
            return
        
        # Crea directory per i dati
        data_dir = Path('data/demonstrations')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome file con ID incrementale e timestamp
        if filename is None:
            demo_id = get_next_demonstration_id()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demonstrations_{demo_id:03d}_{timestamp}.pkl"
        
        filepath = data_dir / filename
        
        # Salva i dati
        data = {
            'demonstrations': self.demonstrations,
            'num_episodes': len(self.demonstrations),
            'total_steps': sum(len(ep['actions']) for ep in self.demonstrations),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Dimostrazioni salvate in: {filepath}")
        return filepath


def main():
    """Funzione principale per raccogliere dimostrazioni."""
    # Richiedi numero di partite PRIMA di inizializzare pygame
    try:
        num_episodes = int(input("Quante partite vuoi giocare? [default: 5]: ") or "5")
    except ValueError:
        num_episodes = 5
    
    print(f"\nInizializzazione ambiente...")
    
    # Inizializza Pygame per catturare input
    pygame.init()
    
    # Crea ambiente
    env = make_space_invaders_env(render_mode='human')
    
    # Crea collector
    collector = DemonstrationCollector(env)
    
    # Raccogli dimostrazioni
    demonstrations = collector.collect_multiple_episodes(num_episodes)
    
    # Salva se ci sono dimostrazioni
    if demonstrations:
        collector.save_demonstrations()
    
    # Chiudi ambiente
    env.close()
    pygame.quit()
    
    print("\nRaccolta completata!")


if __name__ == "__main__":
    main()
