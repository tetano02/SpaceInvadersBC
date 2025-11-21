"""
Script per raccogliere dimostrazioni umane giocando manualmente a Space Invaders.
Usa i tasti freccia per muoverti e SPAZIO per sparare.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
import warnings

# pygame still imports pkg_resources internally; silence its deprecation warning until upstream fixes it
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)

import pygame
from env_make import make_space_invaders_env
from data_manager import DataManager


class DemonstrationCollector:
    """Raccoglie dimostrazioni umane per Behavioral Cloning."""

    def __init__(self, env, fps=30, frame_skip=2):
        self.env = env
        self.demonstrations = []
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Clock per controllare il frame rate
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.frame_skip = frame_skip  # Quanti frame eseguire per ogni azione registrata

        # Mappatura tasti -> azioni Space Invaders
        # Azioni: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
        self.key_to_action = {
            pygame.K_SPACE: 1,  # FIRE
            pygame.K_RIGHT: 2,  # RIGHT
            pygame.K_LEFT: 3,  # LEFT
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
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
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

            # Ottieni azione umana
            action = self.get_human_action()

            # Salva osservazione e azione (copia necessaria perché gym riusa il buffer)
            self.current_episode["observations"].append(observation.copy())
            self.current_episode["actions"].append(action)

            # Esegui l'azione per frame_skip frame e accumula reward
            frame_reward = 0
            for _ in range(self.frame_skip):
                observation, reward, done, truncated, info = self.env.step(action)
                frame_reward += reward
                total_reward += reward
                
                if done or truncated:
                    break
                
                # Limita il frame rate
                self.clock.tick(self.fps)

            # Salva reward accumulato e done
            self.current_episode["rewards"].append(frame_reward)
            self.current_episode["dones"].append(done or truncated)

        # Converti liste in array numpy per efficienza (una sola volta alla fine)
        episode_data = {
            "observations": np.array(
                self.current_episode["observations"], dtype=np.uint8
            ),
            "actions": np.array(
                self.current_episode["actions"], dtype=np.int8
            ),
            "rewards": np.array(
                self.current_episode["rewards"], dtype=np.float32
            ),
            "dones": np.array(
                self.current_episode["dones"], dtype=np.bool_
            ),
        }

        # Salva l'episodio completato
        self.demonstrations.append(episode_data)
        
        # Libera la memoria delle liste temporanee
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        print(f"\nPartita completata! Reward totale: {total_reward}")
        print(f"Passi registrati: {len(episode_data['actions'])}")

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
            print(
                f"Reward min/max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}"
            )
            total_steps = sum(len(ep["actions"]) for ep in self.demonstrations)
            print(f"Passi totali registrati: {total_steps}")

        return self.demonstrations

    def save_demonstrations(self, filename=None):
        """Salva le dimostrazioni su file usando DataManager."""
        data_manager = DataManager()
        return data_manager.save_demonstrations(
            demonstrations=self.demonstrations, filename=filename, source="manual"
        )


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
    env = make_space_invaders_env(render_mode="human")

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
