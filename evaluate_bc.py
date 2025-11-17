"""
Script per valutare e visualizzare la policy appresa con Behavioral Cloning.
"""
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from env_make import make_space_invaders_env
from behavioral_cloning import build_policy, DEFAULT_MODEL_TYPE
from data_manager import DataManager


def _format_model_entry(model_path: Path, index: int) -> str:
    """Restituisce una stringa leggibile con info sul modello."""
    try:
        modified = datetime.fromtimestamp(model_path.stat().st_mtime)
        modified_str = modified.strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        modified_str = "N/A"
    tag = "BEST" if model_path.name == 'best_model.pth' else ""
    tag_display = f" [{tag}]" if tag else ""
    return f"{index}. {model_path.name}{tag_display} (ultimo update: {modified_str})"


def select_model_file(data_manager: DataManager, preselected_index: Optional[int] = None) -> Optional[Path]:
    """Permette all'utente di scegliere un modello disponibile in data/models."""
    models = data_manager.list_models()
    if not models:
        print("Errore: Nessun modello trovato in data/models. Esegui prima l'addestramento.")
        return None
    print("\n=== MODELLI DISPONIBILI ===")
    for idx, model_path in enumerate(models, start=1):
        print(_format_model_entry(model_path, idx))
    print("Digita il numero del modello da valutare oppure 'q' per uscire. [ENTER = 1]")
    while True:
        if preselected_index is not None:
            user_input = str(preselected_index)
            preselected_index = None
            print(f"Selezione automatica: {user_input}")
        else:
            user_input = input("Modello: ").strip()
        if user_input.lower() in {'q', 'quit', 'exit'}:
            return None
        if user_input == '':
            selection = 1
        else:
            if not user_input.isdigit():
                print("Inserisci un numero valido o 'q' per uscire.")
                continue
            selection = int(user_input)
        if 1 <= selection <= len(models):
            chosen = models[selection - 1]
            print(f"\n→ Modello selezionato: {chosen.name}")
            return chosen
        print(f"Selezione fuori range (1-{len(models)}). Riprova.")


class BCAgent:
    """Agente che usa la policy appresa per giocare."""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.data_manager = DataManager()
        self.model_path = Path(model_path)
        
        # Carica modello
        checkpoint = torch.load(model_path, map_location=device)
        run_metadata = checkpoint.get('run_metadata') or {}
        self.run_id = checkpoint.get('run_id') or run_metadata.get('run_id')
        self.run_timestamp = checkpoint.get('run_timestamp') or run_metadata.get('run_timestamp')
        self.metrics_csv_path = (
            checkpoint.get('metrics_csv_path')
            or run_metadata.get('metrics_csv_path')
        )
        self.model_type = (
            run_metadata.get('model_type')
            or checkpoint.get('model_type')
            or DEFAULT_MODEL_TYPE
        )
        self.policy = build_policy(self.model_type, num_actions=6).to(device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        if not self.metrics_csv_path and self.run_timestamp and self.run_id:
            self.metrics_csv_path = str(self.data_manager.get_metrics_filepath(self.run_timestamp, self.run_id))
        if not self.metrics_csv_path:
            parsed = self._infer_run_info_from_filename(self.model_path)
            if parsed:
                ts, rid = parsed
                self.run_timestamp = self.run_timestamp or ts
                self.run_id = self.run_id or rid
                self.metrics_csv_path = str(self.data_manager.get_metrics_filepath(ts, rid))
        
        print(f"Modello caricato da: {model_path}")
        print(f"Device: {device} | Tipo modello: {self.model_type}")

    @staticmethod
    def _infer_run_info_from_filename(model_path: Path):
        """Prova a inferire timestamp e id dal nome file del modello (formato mod_TIMESTAMP_ID)."""
        stem = model_path.stem
        parts = stem.split('_')
        if len(parts) >= 3 and parts[0] == 'mod':
            timestamp = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else None
            model_id = parts[3] if len(parts) >= 4 else None
            if timestamp and model_id:
                return timestamp, model_id
        return None
    
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

    def log_evaluation(self, evaluation_summary, episode_rewards, episode_lengths):
        """Salva i risultati della valutazione nel CSV associato al run."""
        if not self.metrics_csv_path:
            print("[AVVISO] Nessun CSV di metriche associato al modello. Salvataggio valutazione saltato.")
            return
        episode_metrics = [{'reward': float(r), 'steps': int(l)} for r, l in zip(episode_rewards, episode_lengths)]
        metadata = {
            'timestamp': evaluation_summary.get('timestamp'),
            'run_id': f"{self.run_timestamp}_{self.run_id}" if self.run_timestamp and self.run_id else self.run_id,
            'num_episodes': evaluation_summary.get('num_episodes'),
            'model_type': self.model_type,
        }
        self.data_manager.append_evaluation_results(
            csv_path=self.metrics_csv_path,
            evaluation_summary=evaluation_summary,
            episode_metrics=episode_metrics,
            metadata=metadata
        )


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
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_steps = np.mean(episode_lengths)
    print(f"\n{'='*50}")
    print("RISULTATI VALUTAZIONE")
    print(f"{'='*50}")
    print(f"Reward medio: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Reward min/max: {min_reward:.0f} / {max_reward:.0f}")
    print(f"Durata media episodi: {mean_steps:.0f} steps")
    print(f"{'='*50}\n")
    
    evaluation_summary = {
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'average_reward': round(float(mean_reward), 3),
        'reward_std': round(float(std_reward), 3),
        'min_reward': round(float(min_reward), 3),
        'max_reward': round(float(max_reward), 3),
        'average_steps': round(float(mean_steps), 3)
    }
    agent.log_evaluation(evaluation_summary, episode_rewards, episode_lengths)
    
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
    data_manager = DataManager()
    model_path = select_model_file(data_manager)
    if model_path is None:
        print("Nessun modello selezionato. Uscita.")
        return
    
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
