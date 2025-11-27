"""
Script to evaluate and visualize the learned policy with Behavioral Cloning.
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
    """Returns a readable string with model info."""
    try:
        modified = datetime.fromtimestamp(model_path.stat().st_mtime)
        modified_str = modified.strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        modified_str = "N/A"
    tag = "BEST" if model_path.name == "best_model.pth" else ""
    tag_display = f" [{tag}]" if tag else ""
    return f"{index}. {model_path.name}{tag_display} (last update: {modified_str})"


def select_model_file(
    data_manager: DataManager, preselected_index: Optional[int] = None
) -> Optional[Path]:
    """Allows the user to choose an available model in data/models."""
    models = data_manager.list_models()
    if not models:
        print("Error: No models found in data/models. Train first.")
        return None
    print("\n=== AVAILABLE MODELS ===")
    for idx, model_path in enumerate(models, start=1):
        print(_format_model_entry(model_path, idx))
    print("Enter the number of the model to evaluate or 'q' to exit. [ENTER = 1]")
    while True:
        if preselected_index is not None:
            user_input = str(preselected_index)
            preselected_index = None
            print(f"Automatic selection: {user_input}")
        else:
            user_input = input("Model: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            return None
        if user_input == "":
            selection = 1
        else:
            if not user_input.isdigit():
                print("Enter a valid number or 'q' to exit.")
                continue
            selection = int(user_input)
        if 1 <= selection <= len(models):
            chosen = models[selection - 1]
            print(f"\n→ Selected model: {chosen.name}")
            return chosen
        print(f"Selection out of range (1-{len(models)}). Try again.")


class BCAgent:
    """Agent that uses the learned policy to play."""

    def __init__(
        self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.data_manager = DataManager()
        self.model_path = Path(model_path)

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        run_metadata = checkpoint.get("run_metadata") or {}
        self.run_id = checkpoint.get("run_id") or run_metadata.get("run_id")
        self.run_timestamp = checkpoint.get("run_timestamp") or run_metadata.get(
            "run_timestamp"
        )
        self.metrics_csv_path = checkpoint.get("metrics_csv_path") or run_metadata.get(
            "metrics_csv_path"
        )
        self.model_type = (
            run_metadata.get("model_type")
            or checkpoint.get("model_type")
            or DEFAULT_MODEL_TYPE
        )
        self.frame_mode = self._determine_frame_mode(run_metadata, checkpoint)
        input_channels = 6 if self.frame_mode == "stacked" else 3
        self.policy = build_policy(
            self.model_type, num_actions=6, in_channels=input_channels
        ).to(device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        self._prev_frame = None
        if not self.metrics_csv_path and self.run_timestamp and self.run_id:
            self.metrics_csv_path = str(
                self.data_manager.get_metrics_filepath(self.run_timestamp, self.run_id)
            )
        if not self.metrics_csv_path:
            parsed = self._infer_run_info_from_filename(self.model_path)
            if parsed:
                ts, rid = parsed
                self.run_timestamp = self.run_timestamp or ts
                self.run_id = self.run_id or rid
                self.metrics_csv_path = str(
                    self.data_manager.get_metrics_filepath(ts, rid)
                )

        print(f"Model loaded from: {model_path}")
        print(
            f"Device: {device} | Model type: {self.model_type} | Input: {self.frame_mode}"
        )

    @staticmethod
    def _infer_run_info_from_filename(model_path: Path):
        """Tries to infer timestamp and id from model filename (format mod_TIMESTAMP_ID)."""
        stem = model_path.stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "mod":
            timestamp = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else None
            model_id = parts[3] if len(parts) >= 4 else None
            if timestamp and model_id:
                return timestamp, model_id
        return None

    @staticmethod
    def _determine_frame_mode(run_metadata, checkpoint):
        if run_metadata and run_metadata.get("frame_mode"):
            return run_metadata["frame_mode"].lower()
        if "frame_mode" in checkpoint:
            return str(checkpoint["frame_mode"]).lower()
        state_dict = checkpoint.get("model_state_dict", {})
        inferred_channels = BCAgent._infer_input_channels(state_dict)
        return "stacked" if inferred_channels and inferred_channels >= 6 else "single"

    @staticmethod
    def _infer_input_channels(state_dict):
        if not state_dict:
            return None
        conv_keys = ["cnn.0.weight", "patch_embed.weight"]
        for key in conv_keys:
            tensor = state_dict.get(key)
            if tensor is not None and tensor.ndim >= 2:
                return tensor.shape[1]
        linear_key = "network.1.weight"
        tensor = state_dict.get(linear_key)
        if tensor is not None and tensor.ndim >= 2:
            flattened = tensor.shape[1]
            pixels = 210 * 160
            if flattened % pixels == 0:
                return flattened // pixels
        # Fallback: inspect first tensor with >=2 dims
        for tensor in state_dict.values():
            if hasattr(tensor, "ndim") and tensor.ndim >= 2:
                return tensor.shape[1]
        return None

    def _preprocess_observation(self, observation):
        """Prepares the input tensor respecting the frame_mode."""
        curr = torch.FloatTensor(observation).permute(2, 0, 1) / 255.0
        if self.frame_mode == "stacked":
            prev = self._prev_frame if self._prev_frame is not None else curr
            stacked = torch.cat([prev, curr], dim=0)
            self._prev_frame = curr
            return stacked.unsqueeze(0).to(self.device)
        self._prev_frame = curr
        return curr.unsqueeze(0).to(self.device)

    def select_action(self, observation):
        """Selects action from policy."""
        obs = self._preprocess_observation(observation)

        # Predict action
        with torch.no_grad():
            action_logits = self.policy(obs)
            action = torch.argmax(action_logits, dim=1).item()

        return action

    def play_episode(self, env, render=True, max_steps=10000):
        """Plays a complete episode."""
        observation, info = env.reset()
        self._prev_frame = None
        total_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated) and steps < max_steps:
            # Select action
            action = self.select_action(observation)

            # Execute action
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        return total_reward, steps

    def log_evaluation(self, evaluation_summary, episode_rewards, episode_lengths):
        """Saves evaluation results to the CSV associated with the run."""
        if not self.metrics_csv_path:
            print(
                "[WARNING] No metrics CSV associated with the model. Evaluation save skipped."
            )
            return
        episode_metrics = [
            {"reward": float(r), "steps": int(l)}
            for r, l in zip(episode_rewards, episode_lengths)
        ]
        metadata = {
            "timestamp": evaluation_summary.get("timestamp"),
            "run_id": (
                f"{self.run_timestamp}_{self.run_id}"
                if self.run_timestamp and self.run_id
                else self.run_id
            ),
            "num_episodes": evaluation_summary.get("num_episodes"),
            "model_type": self.model_type,
        }
        self.data_manager.append_evaluation_results(
            csv_path=self.metrics_csv_path,
            evaluation_summary=evaluation_summary,
            episode_metrics=episode_metrics,
            metadata=metadata,
        )


def evaluate_agent(agent, num_episodes=10, render_mode="human"):
    """Evaluates the agent over multiple games."""
    env = make_space_invaders_env(render_mode=render_mode)

    print(f"\n{'='*50}")
    print(f"AGENT EVALUATION - {num_episodes} episodes")
    print(f"{'='*50}\n")
    if render_mode == "human":
        print("Mode: rendering active (slower, visible)")
    else:
        print("Mode: headless (no rendering, maximum speed)")
    print()

    episode_rewards = []
    episode_lengths = []

    for i in range(num_episodes):
        reward, steps = agent.play_episode(env)
        episode_rewards.append(reward)
        episode_lengths.append(steps)

        print(f"Episode {i+1}/{num_episodes} - Reward: {reward:.0f}, Steps: {steps}")

    env.close()

    # Statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_steps = np.mean(episode_lengths)
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/max reward: {min_reward:.0f} / {max_reward:.0f}")
    print(f"Average episode duration: {mean_steps:.0f} steps")
    print(f"{'='*50}\n")

    evaluation_summary = {
        "timestamp": datetime.now().isoformat(),
        "num_episodes": num_episodes,
        "average_reward": round(float(mean_reward), 3),
        "reward_std": round(float(std_reward), 3),
        "min_reward": round(float(min_reward), 3),
        "max_reward": round(float(max_reward), 3),
        "average_steps": round(float(mean_steps), 3),
    }
    agent.log_evaluation(evaluation_summary, episode_rewards, episode_lengths)

    return episode_rewards, episode_lengths


def prompt_render_choice(default_show=True):
    """Asks the user whether to display gameplay during evaluation."""
    default_text = "Y" if default_show else "N"
    while True:
        choice = (
            input(
                f"Display gameplay during evaluation? [y/n] (default: {default_text}): "
            )
            .strip()
            .lower()
        )
        if not choice:
            return "human" if default_show else None
        if choice in {"s", "si", "y", "yes"}:
            return "human"
        if choice in {"n", "no"}:
            return None
        print("Invalid input. Answer with 'y' or 'n'.")


def play_interactively(agent):
    """Plays an interactive episode showing the agent's actions."""
    env = make_space_invaders_env(render_mode="human")

    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    print("\n=== INTERACTIVE MODE ===")
    print("Press ENTER to watch the agent play...")
    input()

    agent._prev_frame = None
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.select_action(observation)
        print(f"Step {steps}: Action = {action_names[action]}")

        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()

    print(f"\nEpisode completed!")
    print(f"Total reward: {total_reward}")
    print(f"Duration: {steps} steps")


def main():
    """Main function."""
    data_manager = DataManager()
    model_path = select_model_file(data_manager)
    if model_path is None:
        print("No model selected. Exiting.")
        return

    # Create agent
    agent = BCAgent(model_path)

    # Menu
    while True:
        print("\n=== BC AGENT EVALUATION ===")
        print("1. Evaluate agent (30 episodes)")
        print("2. Play single episode (interactive)")
        print("3. Extended evaluation (custom number of episodes)")
        print("0. Exit")

        choice = input("\nChoice: ")

        if choice == "1":
            render_mode = prompt_render_choice()
            evaluate_agent(agent, num_episodes=30, render_mode=render_mode)
        elif choice == "2":
            play_interactively(agent)
        elif choice == "3":
            num_eps = int(input("Number of episodes: "))
            render_mode = prompt_render_choice()
            evaluate_agent(agent, num_episodes=num_eps, render_mode=render_mode)
        elif choice == "0":
            break
        else:
            print("Invalid choice!")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
