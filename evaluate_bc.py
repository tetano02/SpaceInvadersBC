"""
Script per valutare e visualizzare la policy appresa con Behavioral Cloning.
"""
import torch
import numpy as np
from pathlib import Path
from env_make import make_space_invaders_env
from behavioral_cloning import BCPolicy


class BCAgent:
    """Agente che usa la policy appresa per giocare."""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.policy = BCPolicy(num_actions=6).to(device)
        
        # Carica modello
        checkpoint = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        
        print(f"Modello caricato da: {model_path}")
        print(f"Device: {device}")
    
    def select_action(self, observation):
        """Seleziona azione dalla policy."""
        # Preprocessa osservazione
        obs = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0) / 255.0
        obs = obs.to(self.device)
        
        # Predici azione
        with torch.no_grad():
            action_logits = self.policy(obs)
            action = torch.argmax(action_logits, dim=1).item()
        
        return action
    
    def play_episode(self, env, render=True, max_steps=10000):
        """Gioca un episodio completo."""
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and steps < max_steps:
            # Seleziona azione
            action = self.select_action(observation)
            
            # Esegui azione
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        return total_reward, steps


def evaluate_agent(agent, num_episodes=10):
    """Valuta l'agente su multiple partite."""
    env = make_space_invaders_env(render_mode='human')
    
    print(f"\n{'='*50}")
    print(f"VALUTAZIONE AGENTE - {num_episodes} episodi")
    print(f"{'='*50}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for i in range(num_episodes):
        reward, steps = agent.play_episode(env)
        episode_rewards.append(reward)
        episode_lengths.append(steps)
        
        print(f"Episodio {i+1}/{num_episodes} - Reward: {reward:.0f}, Steps: {steps}")
    
    env.close()
    
    # Statistiche
    print(f"\n{'='*50}")
    print("RISULTATI VALUTAZIONE")
    print(f"{'='*50}")
    print(f"Reward medio: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Reward min/max: {np.min(episode_rewards):.0f} / {np.max(episode_rewards):.0f}")
    print(f"Durata media episodi: {np.mean(episode_lengths):.0f} steps")
    print(f"{'='*50}\n")
    
    return episode_rewards, episode_lengths


def play_interactively(agent):
    """Gioca un episodio interattivo mostrando le azioni dell'agente."""
    env = make_space_invaders_env(render_mode='human')
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    
    print("\n=== MODALITÀ INTERATTIVA ===")
    print("Premi ENTER per vedere l'agente giocare...")
    input()
    
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        action = agent.select_action(observation)
        print(f"Step {steps}: Azione = {action_names[action]}")
        
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    
    env.close()
    
    print(f"\nEpisodio completato!")
    print(f"Reward totale: {total_reward}")
    print(f"Durata: {steps} steps")


def main():
    """Funzione principale."""
    # Trova modello più recente
    model_dir = Path('data/models')
    if not model_dir.exists():
        print("Errore: Directory 'data/models' non trovata!")
        print("Esegui prima 'behavioral_cloning.py' per addestrare un modello.")
        return
    
    # Usa best_model.pth se esiste, altrimenti il più recente
    best_model = model_dir / 'best_model.pth'
    if best_model.exists():
        model_path = best_model
        print(f"Usando miglior modello: {model_path}")
    else:
        model_files = list(model_dir.glob('bc_model_*.pth'))
        if not model_files:
            print("Errore: Nessun modello trovato!")
            print("Esegui prima 'behavioral_cloning.py' per addestrare un modello.")
            return
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"Usando modello: {model_path}")
    
    # Crea agente
    agent = BCAgent(model_path)
    
    # Menu
    while True:
        print("\n=== VALUTAZIONE BC AGENT ===")
        print("1. Valuta agente (10 episodi)")
        print("2. Gioca episodio singolo (interattivo)")
        print("3. Valutazione estesa (custom numero episodi)")
        print("0. Esci")
        
        choice = input("\nScelta: ")
        
        if choice == '1':
            evaluate_agent(agent, num_episodes=10)
        elif choice == '2':
            play_interactively(agent)
        elif choice == '3':
            num_eps = int(input("Numero di episodi: "))
            evaluate_agent(agent, num_episodes=num_eps)
        elif choice == '0':
            break
        else:
            print("Scelta non valida!")
    
    print("\nArrivederci!")


if __name__ == "__main__":
    main()
