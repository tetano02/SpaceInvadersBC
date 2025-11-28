"""
=========================================================
PHASE 0 — PROJECT ANALYSIS
=========================================================
1. Policy networks already available:
   • `BCPolicy` (three-layer CNN), `BCMLPPolicy` (fully-connected), and
     `BCVisionTransformer` (ViT) live in `behavioral_cloning.py` and all emit a
     tensor of shape (batch_size, num_actions). During BC training those logits
     are consumed by `nn.CrossEntropyLoss` for supervised classification.
2. Behavioral Cloning pipeline:
   • `BCDataset` iterates demonstrations stored as dictionaries with numpy arrays
     (`observations`, `actions`, `rewards`, `dones`). Observations are stacked or
     single RGB frames normalized to [0, 1].
   • `BCTrainer` runs supervised updates with Adam and logs metrics via
     `DataManager`.
3. Expert data ingestion:
   • `load_minari_dataset.py` downloads Minari datasets, converts each episode to
     the same demonstration dict used everywhere else, and persists them with
     `DataManager.save_demonstrations`.
   • `DataManager.load_demonstrations` reloads those pickles for subsequent
     training scripts.
4. Transition storage:
   • Demonstrations keep per-episode numpy arrays. Each tuple implicitly stores
     (state_t, action_t, reward_t, done_t) with `observations[i+1]` acting as the
     next-state. This structure is leveraged by BC and will now be reused to
     build an expert replay buffer for GAIL.
5. GAIL integration plan:
   • Provide policy helpers (`infer_policy_output_type` + `select_action`)
     that use categorical sampling from logits with optional epsilon-greedy
     exploration.
   • Implement `Discriminator` (see `discriminator.py`) and a `GAILTrainer` that
     can: load BC checkpoints, convert demonstrations into transition batches,
     collect fresh trajectories from any Gym-compatible env, update the
     discriminator, and update the policy using policy gradient.
   • The trainer exposes modular methods (`collect_agent_trajectories`,
     `update_discriminator`, `compute_gail_rewards`, and policy gradient update)
     so any policy architecture can plug in.
=========================================================
PHASE 1+ IMPLEMENTATION OVERVIEW
=========================================================
The remainder of this module implements the helper utilities described above,
plus a complete `GAILTrainer` with a training loop that follows the requested
sequence (collect → discriminator updates → policy update + logging).
"""

from __future__ import annotations

import gc
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from data_manager import DataManager
from gail.gail_discriminator import Discriminator

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


def default_state_preprocessor(observation: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Converts raw observations to float tensors with shape (C, H, W) or (D,).

    The default logic assumes Atari-like RGB frames (H, W, C) and transposes
    them to channel-first format while normalizing to [0, 1]. Vector inputs are
    simply cast to float32.
    """
    try:
        # Convert to numpy first if it's a tensor to reduce memory
        if isinstance(observation, torch.Tensor):
            if observation.is_cuda:
                observation = observation.cpu()
            # Use detach without clone to avoid copy
            arr = (
                observation.detach().numpy()
                if observation.requires_grad
                else observation.numpy()
            )
        else:
            arr = np.asarray(observation)

        # Check if normalization is needed before converting
        needs_norm = arr.max() > 1.0

        # Convert to float32 and normalize in a single step
        if needs_norm:
            tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
        else:
            tensor = torch.from_numpy(arr.astype(np.float32))

        # Permute if necessary
        if tensor.dim() == 3 and tensor.shape[-1] in {1, 3, 4, 6}:
            tensor = tensor.permute(2, 0, 1)

        return tensor
    except (RuntimeError, MemoryError) as e:
        if "not enough memory" in str(e) or isinstance(e, MemoryError):
            # Force garbage collection and abort with a clear message
            gc.collect()
            raise RuntimeError(
                f"Out of memory in state preprocessing. Try reducing buffer capacity or batch size. Original error: {e}"
            )
        raise


class FrameStackPreprocessor:
    """Wraps a base preprocessor to provide N-frame stacking."""

    def __init__(
        self,
        stack_size: int = 1,
        base_preprocessor: Callable = default_state_preprocessor,
    ) -> None:
        self.stack_size = max(1, int(stack_size))
        self.base_preprocessor = base_preprocessor
        self._buffer: deque[torch.Tensor] = deque(maxlen=self.stack_size)

    def reset(self) -> None:
        self._buffer.clear()

    def __call__(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = self.base_preprocessor(observation)
        if self.stack_size == 1:
            return tensor
        if not self._buffer:
            for _ in range(self.stack_size):
                self._buffer.append(
                    tensor.clone()
                )  # Clone to avoid reference to the same tensor
        else:
            self._buffer.append(tensor)
        # Use stack instead of cat for better efficiency
        result = torch.stack(list(self._buffer), dim=0).flatten(0, 1)
        return result


@dataclass
class Transition:
    """Stores a single transition for replay buffers.

    Note: `state` and `next_state` are stored as uint8 tensors in [0, 255]
    to reduce memory usage and converted back to float32 in
    `_transitions_to_batch`.
    """

    state: torch.Tensor  # (C, H, W) or (D,), stored as uint8
    action: int
    env_reward: float
    next_state: torch.Tensor  # matches state shape, stored as uint8
    done: bool


class ReplayBuffer:
    """Simple FIFO replay buffer for agent trajectories."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)  # deque automatically handles FIFO with maxlen

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = min(batch_size, len(self.buffer))
        # random.sample works directly on a deque; no need to cast to list
        return random.sample(self.buffer, batch_size)


# -----------------------------------------------------------------------------
# Policy helpers
# -----------------------------------------------------------------------------


def infer_policy_output_type(policy: nn.Module, explicit: Optional[str] = None) -> str:
    """Returns 'logits' as all policies emit logits for categorical distributions."""
    return "logits"


def select_action(
    policy: nn.Module,
    state: torch.Tensor,
    epsilon: float = 0.0,
    output_type: Optional[str] = None,
) -> int:
    """Selects an action using categorical sampling from logits.

    Args:
        policy: Any policy network producing (1, num_actions) logits.
        state: Tensor with batch dimension first (1, C, H, W) or (1, D).
        epsilon: Exploration probability (random action selection).
        output_type: Unused, kept for backward compatibility.
    """

    policy_device = next(policy.parameters()).device
    state = state.to(policy_device)
    with torch.no_grad():
        outputs = policy(state)
    num_actions = outputs.shape[-1]

    # Epsilon-greedy exploration
    if epsilon > 0.0 and random.random() < epsilon:
        return random.randrange(num_actions)

    # Categorical sampling from logits
    probs = F.softmax(outputs, dim=-1)
    distribution = Categorical(probs=probs)
    action = distribution.sample()
    return int(action.item())


# -----------------------------------------------------------------------------
# GAIL trainer
# -----------------------------------------------------------------------------


class GAILTrainer:
    """GAIL trainer for policy-gradient policies using logits."""

    def __init__(
        self,
        policy: nn.Module,
        discriminator: Discriminator,
        env,
        expert_demonstrations: Optional[Sequence[dict]] = None,
        expert_files: Optional[Sequence[str | Path]] = None,
        bc_checkpoint_path: Optional[str | Path] = None,
        policy_output_type: Optional[str] = None,
        state_preprocessor: Callable[
            [np.ndarray | torch.Tensor], torch.Tensor
        ] = default_state_preprocessor,
        agent_buffer_capacity: int = 100_000,
        gamma: float = 0.99,
        policy_lr: float = 1e-4,
        discriminator_lr: float = 1e-4,
        batch_size: int = 64,
        device: Optional[str] = None,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = policy.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.env = env
        env_id = getattr(getattr(self.env, "spec", None), "id", "unknown_env")
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.state_preprocessor = state_preprocessor
        self.policy_output_type = infer_policy_output_type(
            self.policy, explicit=policy_output_type
        )

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )
        self.discriminator_loss = nn.BCEWithLogitsLoss()

        self.agent_buffer = ReplayBuffer(agent_buffer_capacity)
        self.data_manager = DataManager()
        self.run_timestamp, self.run_id = self.data_manager.create_run_identifier()
        self.metrics_csv_path = self.data_manager.get_metrics_filepath(
            self.run_timestamp, self.run_id
        )
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None
        self.iteration_history: List[dict] = []
        self.base_model_path = str(bc_checkpoint_path) if bc_checkpoint_path else None
        self.best_mean_return: float = -float("inf")
        self.best_model_path: Optional[str] = None

        # Infer model_type and frame_mode from BC checkpoint if available
        self.model_type: Optional[str] = None
        self.frame_mode: Optional[str] = None
        if bc_checkpoint_path:
            bc_checkpoint = self.data_manager.load_model(
                bc_checkpoint_path, device="cpu"
            )
            bc_metadata = bc_checkpoint.get("run_metadata") or {}
            self.model_type = bc_metadata.get("model_type")
            self.frame_mode = bc_metadata.get("frame_mode")

        self.training_metadata = {
            "training_type": "gail",
            "policy_output_type": self.policy_output_type,
            "model_type": self.model_type,
            "frame_mode": self.frame_mode,
            "base_model_path": self.base_model_path,
            "base_model_name": (
                Path(self.base_model_path).name if self.base_model_path else "N/A"
            ),
            "environment_name": env_id,
            "agent_buffer_capacity": agent_buffer_capacity,
            "batch_size": batch_size,
        }
        self._last_train_config: dict = {}

        if bc_checkpoint_path:
            self._load_bc_checkpoint(bc_checkpoint_path)

        demonstrations = list(expert_demonstrations or [])
        if expert_files:
            for file_path in expert_files:
                demonstrations.extend(self.data_manager.load_demonstrations(file_path))
        self.training_metadata["expert_files"] = [str(p) for p in (expert_files or [])]
        if not demonstrations:
            raise ValueError("GAILTrainer requires at least one expert demonstration.")
        self.expert_transitions = self._prepare_expert_transitions(demonstrations)
        self.observation_shape = tuple(self.expert_transitions[0].state.shape)
        self._reset_state_preprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect_agent_trajectories(self, num_steps: int, epsilon: float = 0.05) -> dict:
        """Rolls out the current policy for `num_steps` and fills the replay buffer."""

        steps_collected = 0
        episode_rewards = []
        episode_return = 0.0
        obs, _ = self.env.reset()
        self._reset_state_preprocessor()
        state = self._prepare_state(obs)

        while steps_collected < num_steps:
            with torch.no_grad():  # Avoid gradient accumulation
                action = select_action(
                    self.policy,
                    state,
                    epsilon=epsilon,
                    output_type=self.policy_output_type,
                )

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            next_state = self._prepare_state(next_obs)

            # Store compressed uint8 version in the buffer (values in [0, 255])
            state_uint8 = (state.squeeze(0).detach().cpu().clamp(0.0, 1.0) * 255.0).to(
                torch.uint8
            )
            next_state_uint8 = (
                next_state.squeeze(0).detach().cpu().clamp(0.0, 1.0) * 255.0
            ).to(torch.uint8)

            transition = Transition(
                state=state_uint8,
                action=int(action),
                env_reward=float(reward),
                next_state=next_state_uint8,
                done=done,
            )
            self.agent_buffer.push(transition)

            episode_return += reward
            steps_collected += 1

            # Free memory of previous state
            del state

            if done:
                episode_rewards.append(episode_return)
                episode_return = 0.0
                obs, _ = self.env.reset()
                self._reset_state_preprocessor()
                state = self._prepare_state(obs)
            else:
                state = next_state

        if not episode_rewards:
            episode_rewards.append(episode_return)

        # Clean up tensor references
        del state, next_state

        return {"steps": steps_collected, "episode_rewards": episode_rewards}

    def sample_expert_batch(self, batch_size: Optional[int] = None) -> dict:
        """Samples a batch of expert transitions."""

        batch_size = batch_size or self.batch_size
        if batch_size >= len(self.expert_transitions):
            transitions = random.choices(self.expert_transitions, k=batch_size)
        else:
            transitions = random.sample(self.expert_transitions, batch_size)
        return self._transitions_to_batch(transitions)

    def sample_agent_batch(self, batch_size: Optional[int] = None) -> dict:
        """Samples a batch from the agent replay buffer."""

        if len(self.agent_buffer) == 0:
            raise RuntimeError("Agent buffer is empty. Collect trajectories first.")
        batch_size = batch_size or self.batch_size
        transitions = self.agent_buffer.sample(batch_size)
        return self._transitions_to_batch(transitions)

    def update_discriminator(self, expert_batch: dict, agent_batch: dict) -> dict:
        """Performs one discriminator update step."""

        expert_states = expert_batch["states"].to(self.device)
        expert_actions = expert_batch["actions"].to(self.device)
        agent_states = agent_batch["states"].to(self.device)
        agent_actions = agent_batch["actions"].to(self.device)

        logits_expert = self.discriminator(expert_states, expert_actions)
        logits_agent = self.discriminator(agent_states, agent_actions)
        labels = torch.cat(
            [
                torch.ones_like(logits_expert),
                torch.zeros_like(logits_agent),
            ]
        )
        logits = torch.cat([logits_expert, logits_agent])

        loss = self.discriminator_loss(logits, labels)
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            expert_acc = (torch.sigmoid(logits_expert) > 0.5).float().mean().item()
            agent_acc = (torch.sigmoid(logits_agent) < 0.5).float().mean().item()

        # Free memory of intermediate tensors
        del expert_states, expert_actions, agent_states, agent_actions
        del logits_expert, logits_agent, labels, logits

        return {
            "loss": float(loss.item()),
            "expert_acc": expert_acc,
            "agent_acc": agent_acc,
        }

    def compute_gail_rewards(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Returns shaped rewards derived from the discriminator."""

        logits = self.discriminator(states.to(self.device), actions.to(self.device))
        probs = torch.sigmoid(logits)
        rewards = -torch.log(1.0 - probs + 1e-8)
        return rewards.detach()

    def update_policy(self, batch: dict, gail_rewards: torch.Tensor) -> dict:
        """Updates the policy using policy gradient."""
        loss = self._update_policy_policy_gradient(batch, gail_rewards)
        return {"policy_loss": loss, "mode": "policy_gradient"}

    def validate_policy(self, num_episodes: int = 10, epsilon: float = 0.0) -> dict:
        """Validates the current policy on a fixed number of complete episodes.

        Args:
            num_episodes: Number of episodes to complete for validation
            epsilon: Epsilon for exploration (default 0.0 = greedy)

        Returns:
            Dict with mean_return, std_return, episode_rewards
        """
        # Memory cleanup before validation
        gc.collect()

        episode_rewards = []

        for episode_idx in range(num_episodes):
            episode_return = 0.0
            obs, _ = self.env.reset()
            self._reset_state_preprocessor()
            state = self._prepare_state(obs)
            done = False

            while not done:
                with torch.no_grad():
                    action = select_action(
                        self.policy,
                        state,
                        epsilon=epsilon,
                        output_type=self.policy_output_type,
                    )

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                episode_return += reward

                if not done:
                    # Free previous state
                    del state
                    state = self._prepare_state(next_obs)

            episode_rewards.append(episode_return)
            print(
                f"  Episode {episode_idx + 1}/{num_episodes} done (reward: {episode_return:.1f})"
            )

            # Periodic cleanup every 10 episodes
            if (episode_idx + 1) % 10 == 0:
                gc.collect()

        mean_return = sum(episode_rewards) / len(episode_rewards)
        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in episode_rewards) / len(
            episode_rewards
        )
        std_return = variance**0.5

        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "episode_rewards": episode_rewards,
        }

    def train(
        self,
        num_iterations: int,
        steps_per_collect: int = 5000,
        discriminator_updates: int = 2,
        policy_updates: int = 1,
        epsilon: float = 0.05,
        log_interval: int = 1,
        use_validation: bool = True,
        validation_episodes: int = 30,
    ) -> None:
        """Runs the full GAIL training loop.

        Args:
            use_validation: If True, validates with more episodes when finding a potential best model
            validation_episodes: Number of episodes for thorough validation
        """

        self.training_start_time = datetime.now()
        self.iteration_history.clear()
        self._last_train_config = {
            "num_iterations": num_iterations,
            "steps_per_collect": steps_per_collect,
            "discriminator_updates": discriminator_updates,
            "policy_updates": policy_updates,
            "epsilon": epsilon,
        }

        try:
            for iteration in range(1, num_iterations + 1):
                rollout_stats = self.collect_agent_trajectories(
                    steps_per_collect, epsilon=epsilon
                )
                disc_losses = []
                for _ in range(discriminator_updates):
                    expert_batch = self.sample_expert_batch()
                    agent_batch = self.sample_agent_batch()
                    disc_metrics = self.update_discriminator(expert_batch, agent_batch)
                    disc_losses.append(disc_metrics)
                    # Free batch after use
                    del expert_batch, agent_batch

                agent_batch = self.sample_agent_batch()
                gail_rewards = self.compute_gail_rewards(
                    agent_batch["states"], agent_batch["actions"]
                )
                policy_metrics = None
                for _ in range(policy_updates):
                    policy_metrics = self.update_policy(agent_batch, gail_rewards)

                avg_disc_loss = sum(m["loss"] for m in disc_losses) / max(
                    len(disc_losses), 1
                )
                avg_expert_acc = sum(m["expert_acc"] for m in disc_losses) / max(
                    len(disc_losses), 1
                )
                avg_agent_acc = sum(m["agent_acc"] for m in disc_losses) / max(
                    len(disc_losses), 1
                )

                mean_reward = sum(rollout_stats["episode_rewards"]) / len(
                    rollout_stats["episode_rewards"]
                )
                self._record_iteration_metrics(
                    iteration=iteration,
                    disc_loss=avg_disc_loss,
                    expert_acc=avg_expert_acc,
                    agent_acc=avg_agent_acc,
                    policy_loss=(
                        policy_metrics["policy_loss"] if policy_metrics else 0.0
                    ),
                    policy_mode=policy_metrics["mode"] if policy_metrics else "n/a",
                    mean_return=mean_reward,
                    steps=rollout_stats["steps"],
                    epsilon=epsilon,
                    disc_updates=discriminator_updates,
                    policy_updates=policy_updates,
                    gail_reward_mean=(
                        float(gail_rewards.mean().item())
                        if gail_rewards.numel() > 0
                        else 0.0
                    ),
                )

                # Clean memory AFTER recording metrics
                del agent_batch, gail_rewards, disc_losses

                # Force garbage collection once per iteration
                gc.collect()

                # If training mean suggests improvement (or within 10% of best), validate thoroughly
                threshold = self.best_mean_return * 0.9  # 10% below best
                if mean_reward > threshold:
                    if use_validation:
                        # Show if it's potential best or just close
                        if mean_reward > self.best_mean_return:
                            print(
                                f"\n[Iter {iteration}] Potential best model (TrainMean={mean_reward:.1f})"
                            )
                        else:
                            print(
                                f"\n[Iter {iteration}] Model close to best (TrainMean={mean_reward:.1f}, threshold={threshold:.1f})"
                            )
                        print(
                            f"[Iter {iteration}] Validating with {validation_episodes} episodes..."
                        )
                        val_stats = self.validate_policy(
                            num_episodes=validation_episodes, epsilon=0.0
                        )
                        val_mean = val_stats["mean_return"]
                        val_std = val_stats["std_return"]
                        print(
                            f"[Iter {iteration}] Validation: MeanReturn={val_mean:.1f} (±{val_std:.1f})"
                        )

                        # Save ONLY if validation confirms improvement
                        if val_mean > self.best_mean_return:
                            self.best_mean_return = val_mean
                            saved_path, _, _ = self.save_checkpoint(
                                filename="best_model_temp.pth"
                            )
                            self.best_model_path = str(saved_path)
                            print(
                                f"[Iter {iteration}] ✓ Confirmed! New best model: ValMean={val_mean:.1f}"
                            )
                        else:
                            print(
                                f"[Iter {iteration}] ✗ Not confirmed (ValMean={val_mean:.1f} < Best={self.best_mean_return:.1f})"
                            )
                    else:
                        # Without validation, use training mean directly
                        if mean_reward > self.best_mean_return:
                            self.best_mean_return = mean_reward
                            saved_path, _, _ = self.save_checkpoint(
                                filename="best_model_temp.pth"
                            )
                            self.best_model_path = str(saved_path)
                            if iteration % log_interval == 0:
                                print(
                                    f"[Iter {iteration}] New best model! MeanEpReturn={mean_reward:.1f}"
                                )

                if iteration % log_interval == 0:
                    print(
                        f"[Iter {iteration}] DiscLoss={avg_disc_loss:.3f} "
                        f"E_acc={avg_expert_acc:.2f} A_acc={avg_agent_acc:.2f} "
                        f"Policy({policy_metrics['mode']}): {policy_metrics['policy_loss']:.3f} "
                        f"MeanEpReturn={mean_reward:.1f}"
                    )
        finally:
            self.training_end_time = datetime.now()
            self._export_iteration_metrics()

            # Final comparison: last model vs best saved model
            print(f"\n{'='*60}")
            print("Final comparison: last model vs best model")
            print(f"{'='*60}")

            # Always validate last model for accurate comparison
            print(f"\nEvaluating final model (iteration {num_iterations})...")
            final_val_stats = self.validate_policy(
                num_episodes=validation_episodes, epsilon=0.0
            )
            final_mean = final_val_stats["mean_return"]
            final_std = final_val_stats["std_return"]
            print(f"Final model: MeanReturn={final_mean:.1f} (±{final_std:.1f})")

            # Compare with saved best
            if self.best_model_path:
                print(f"Best model saved: MeanReturn={self.best_mean_return:.1f}")

                if final_mean > self.best_mean_return:
                    print(f"\n✓ Final model is better! Saving as best model...")
                    self.best_mean_return = final_mean
                    final_best_mean = final_mean
                    # Save current model (which is the best)
                    saved_path, _, _ = self.save_checkpoint(
                        filename="best_model_temp.pth"
                    )
                    self.best_model_path = str(saved_path)
                else:
                    print(f"\n✓ Previous best model is better, keeping it.")
                    final_best_mean = self.best_mean_return
            else:
                # No best saved, use the last one
                print("\nNo previous best model, saving final model.")
                self.best_mean_return = final_mean
                final_best_mean = final_mean
                saved_path, _, _ = self.save_checkpoint(filename="best_model_temp.pth")
                self.best_model_path = str(saved_path)

            # Copy best model with run timestamp and ID (also keeps temp)
            if self.best_model_path:
                temp_path = Path(self.best_model_path)
                if temp_path.exists():
                    # Generate final name with timestamp and ID
                    final_filename = f"mod_{self.run_timestamp}_{self.run_id}.pth"
                    final_path = temp_path.parent / final_filename
                    # Copy instead of rename to keep best_model_temp.pth
                    import shutil

                    shutil.copy2(temp_path, final_path)
                    self.best_model_path = str(final_path)

            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(
                f"Best model (ValMeanReturn={final_best_mean:.1f}): {self.best_model_path}"
            )
            print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_state(self, observation) -> torch.Tensor:
        processed = self.state_preprocessor(observation)
        processed = processed.to(self.device)
        base_dim = len(getattr(self, "observation_shape", processed.shape))
        if processed.dim() == base_dim:
            processed = processed.unsqueeze(0)
        return processed

    def _reset_state_preprocessor(self) -> None:
        if hasattr(self.state_preprocessor, "reset"):
            self.state_preprocessor.reset()

    def _record_iteration_metrics(
        self,
        iteration: int,
        disc_loss: float,
        expert_acc: float,
        agent_acc: float,
        policy_loss: float,
        policy_mode: str,
        mean_return: float,
        steps: int,
        epsilon: float,
        disc_updates: int,
        policy_updates: int,
        gail_reward_mean: float,
    ) -> None:
        self.iteration_history.append(
            {
                "iteration": iteration,
                "disc_loss": disc_loss,
                "expert_acc": expert_acc,
                "agent_acc": agent_acc,
                "policy_loss": policy_loss,
                "policy_mode": policy_mode,
                "mean_return": mean_return,
                "steps": steps,
                "epsilon": epsilon,
                "disc_updates": disc_updates,
                "policy_updates": policy_updates,
                "gail_reward_mean": gail_reward_mean,
            }
        )

    def _export_iteration_metrics(self) -> None:
        if not self.iteration_history:
            return
        metadata = self.training_metadata.copy()
        metadata.update(
            {
                "run_timestamp": self.run_timestamp,
                "run_id": self.run_id,
                "num_iterations": len(self.iteration_history),
                "steps_per_collect": self._last_train_config.get("steps_per_collect"),
                "discriminator_updates": self._last_train_config.get(
                    "discriminator_updates"
                ),
                "policy_updates": self._last_train_config.get("policy_updates"),
                "epsilon": self._last_train_config.get("epsilon"),
            }
        )
        if self.training_start_time and self.training_end_time:
            duration = (
                self.training_end_time - self.training_start_time
            ).total_seconds()
            metadata["training_start"] = self.training_start_time.isoformat()
            metadata["training_end"] = self.training_end_time.isoformat()
            metadata["training_duration_seconds"] = round(duration, 2)
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                serializable_metadata[key] = " | ".join(map(str, value))
            else:
                serializable_metadata[key] = value
        self.metrics_csv_path = self.data_manager.save_gail_metrics_csv(
            run_timestamp=self.run_timestamp,
            run_id=self.run_id,
            metadata=serializable_metadata,
            iteration_metrics=self.iteration_history,
            target_path=self.metrics_csv_path,
        )

    def save_checkpoint(self, filename: Optional[str | Path] = None):
        metadata = self.training_metadata.copy()
        metadata.update(
            {
                "run_timestamp": self.run_timestamp,
                "run_id": self.run_id,
                "metrics_csv_path": str(self.metrics_csv_path),
                "training_type": "gail",
            }
        )
        custom_timestamp = self.run_timestamp if filename is None else None
        custom_id = self.run_id if filename is None else None
        return self.data_manager.save_model(
            model_state_dict=self.policy.state_dict(),
            optimizer_state_dict=self.policy_optimizer.state_dict(),
            train_losses=None,
            val_losses=None,
            train_accuracies=None,
            val_accuracies=None,
            filename=filename,
            custom_timestamp=custom_timestamp,
            custom_id=custom_id,
            metadata=metadata,
        )

    def _load_bc_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = self.data_manager.load_model(checkpoint_path, device=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded BC weights from {checkpoint_path}")

    def _prepare_expert_transitions(
        self, demonstrations: Sequence[dict]
    ) -> List[Transition]:
        transitions: List[Transition] = []
        for episode in demonstrations:
            self._reset_state_preprocessor()
            observations = episode["observations"]
            actions = episode["actions"]
            rewards = episode.get("rewards")
            dones = episode.get("dones")
            encoded_states: List[torch.Tensor] = []
            for obs in observations:
                float_tensor = self.state_preprocessor(obs)  # float32 in [0, 1]
                uint8_tensor = (float_tensor.detach().cpu().clamp(0.0, 1.0) * 255.0).to(
                    torch.uint8
                )
                encoded_states.append(uint8_tensor)

            if not encoded_states:
                continue

            episode_len = len(actions)
            last_index = len(encoded_states) - 1
            for idx in range(episode_len):
                state = encoded_states[min(idx, last_index)]
                next_idx = min(idx + 1, last_index)
                next_state = encoded_states[next_idx]
                reward = float(rewards[idx]) if rewards is not None else 0.0
                done = (
                    bool(dones[idx]) if dones is not None else (idx == episode_len - 1)
                )
                transitions.append(
                    Transition(
                        state=state,
                        action=int(actions[idx]),
                        env_reward=reward,
                        next_state=next_state,
                        done=done,
                    )
                )
        return transitions

    def _transitions_to_batch(self, transitions: Sequence[Transition]) -> dict:
        # Build lists instead of immediate stack to reduce memory spikes
        states_list = [t.state for t in transitions]
        actions_list = [t.action for t in transitions]
        next_states_list = [t.next_state for t in transitions]
        dones_list = [t.done for t in transitions]
        rewards_list = [t.env_reward for t in transitions]

        # Decode uint8 states to float32 in [0, 1] directly on target device
        states = torch.stack(states_list).to(self.device).to(torch.float32) / 255.0
        actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        next_states = (
            torch.stack(next_states_list).to(self.device).to(torch.float32) / 255.0
        )
        dones = torch.tensor(dones_list, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # Clean up lists
        del states_list, actions_list, next_states_list, dones_list, rewards_list

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "dones": dones,
            "env_rewards": rewards,
        }

    def _update_policy_policy_gradient(
        self, batch: dict, gail_rewards: torch.Tensor
    ) -> float:
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        gail_rewards = gail_rewards.to(self.device).squeeze(-1)
        advantages = (gail_rewards - gail_rewards.mean()) / (gail_rewards.std() + 1e-8)

        logits = self.policy(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = -(selected_log_probs * advantages.detach()).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        return float(loss.item())


# -----------------------------------------------------------------------------
# Interactive GAIL training script
# -----------------------------------------------------------------------------


def choose_model_file(model_files):
    """Allows the user to choose a saved model."""
    if not model_files:
        print("\nNo models available.")
        return None

    # Filter only files that actually exist
    existing_models = [p for p in model_files if p.exists()]
    if not existing_models:
        print("\nNo valid models found.")
        return None

    sorted_models = sorted(
        existing_models, key=lambda p: p.stat().st_mtime, reverse=True
    )
    print("\n=== AVAILABLE MODELS FOR GAIL ===")
    for idx, model_path in enumerate(sorted_models, start=1):
        timestamp = datetime.fromtimestamp(model_path.stat().st_mtime)
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {idx}. {model_path.name} (last update: {ts_str})")
    print("Enter the model number or 'q' to cancel. [ENTER = 1]")

    while True:
        choice = input("Model: ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        if not choice:
            return sorted_models[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sorted_models):
                return sorted_models[idx - 1]
        print("Invalid input, try again.")


def prompt_int_input(prompt: str, default: int) -> int:
    """Helper for integer input with default value."""
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default
        if user_input.isdigit():
            return int(user_input)
        print(f"Enter a valid number or press ENTER to use {default}")


def prompt_float_input(prompt: str, default: float) -> float:
    """Helper for float input with default value."""
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default
        try:
            return float(user_input)
        except ValueError:
            print(f"Enter a valid number or press ENTER to use {default}")


def main(demo_files=None, model_files=None):
    """Interactive workflow to improve a policy via GAIL.

    Args:
        demo_files: List of Path with demonstration files
        model_files: List of Path with trained models
    """
    from data_manager import DataManager
    from behavioral_cloning import (
        build_policy,
        DEFAULT_MODEL_TYPE,
        select_demonstration_files,
        select_device,
    )
    from gail.gail_discriminator import Discriminator
    from env_make import make_space_invaders_env
    from test_model import BCAgent

    if not demo_files:
        print("\n⚠ Demonstrations are required to run GAIL.")
        return

    if not model_files:
        print("\n⚠ Trained models are required to run GAIL.")
        return

    data_manager = DataManager()
    model_path = choose_model_file(model_files)
    if model_path is None:
        print("\nOperation cancelled: no model selected.")
        return

    checkpoint = data_manager.load_model(model_path, device="cpu")
    run_metadata = checkpoint.get("run_metadata") or {}
    model_type = run_metadata.get("model_type", DEFAULT_MODEL_TYPE)
    frame_mode = BCAgent._determine_frame_mode(run_metadata, checkpoint) or "single"
    frame_mode = frame_mode.lower()
    stack_size = 2 if frame_mode == "stacked" else 1

    print(f"\nSelected model: {model_path.name}")
    print(f"Architecture type: {model_type} | Input: {frame_mode}")

    selected_demo_files = select_demonstration_files(demo_files)
    if not selected_demo_files:
        print("\nOperation cancelled: no demonstrations selected.")
        return

    preview_data = data_manager.load_demonstrations(selected_demo_files[0])
    first_episode = preview_data[0]
    first_obs = first_episode["observations"][0]
    if first_obs.ndim == 3:
        base_channels = first_obs.shape[2]
        height, width = first_obs.shape[0], first_obs.shape[1]
        input_channels = base_channels * stack_size
        observation_shape = (input_channels, height, width)
    else:
        base_dim = first_obs.shape[0]
        input_channels = base_dim * stack_size
        observation_shape = (input_channels,)

    device = select_device()
    env = make_space_invaders_env(render_mode=None)
    num_actions = env.action_space.n

    policy = build_policy(
        model_type=model_type, num_actions=num_actions, in_channels=input_channels
    )
    preprocessor = FrameStackPreprocessor(stack_size=stack_size)
    discriminator = Discriminator(
        observation_shape=observation_shape, num_actions=num_actions
    )

    print("\nConfigure GAIL training:")
    num_iterations = prompt_int_input("GAIL iterations [default: 30]: ", 30)
    steps_per_collect = prompt_int_input(
        "Steps collected per iteration [default: 5000]: ", 5000
    )
    disc_updates = prompt_int_input(
        "Discriminator updates per iteration [default: 2]: ", 2
    )
    policy_updates = prompt_int_input("Policy updates per iteration [default: 1]: ", 1)
    epsilon = prompt_float_input("Exploration epsilon [default: 0.05]: ", 0.05)
    buffer_capacity = prompt_int_input(
        "Agent buffer capacity [default: 50000]: ", 50000
    )

    print("\nBest model validation (recommended to identify true best):")
    use_validation_input = (
        input("Validate with more episodes when finding potential best? [Y/n]: ")
        .strip()
        .lower()
    )
    use_validation = use_validation_input not in {"n", "no"}

    validation_episodes = (
        prompt_int_input("Episodes for validation [default: 30]: ", 30)
        if use_validation
        else 30
    )

    # Ask user for BC model baseline
    print("\n" + "=" * 60)
    print("Starting model baseline (BC)")
    print("=" * 60)
    print("Enter the average performance of the BC model to use as baseline.")
    print("(Leave empty to start from -inf, any model will be considered better)")

    baseline_input = input("BC model MeanReturn [default: no baseline]: ").strip()

    baseline_mean = None
    if baseline_input:
        try:
            baseline_mean = float(baseline_input)
            print(f"\nBaseline set: {baseline_mean:.1f}")
            print("GAIL will try to exceed this performance.\n")
        except ValueError:
            print("Invalid value. No baseline set.\n")
    else:
        print("No baseline set. Any improvement will be considered best.\n")

    trainer = GAILTrainer(
        policy=policy,
        discriminator=discriminator,
        env=env,
        expert_files=selected_demo_files,
        bc_checkpoint_path=model_path,
        state_preprocessor=preprocessor,
        agent_buffer_capacity=buffer_capacity,
        device=device,
    )

    # Set baseline as initial best_mean_return if available
    if baseline_mean is not None:
        trainer.best_mean_return = baseline_mean
        print(f"Initial baseline set: {baseline_mean:.1f}")
        print("GAIL will try to exceed this performance.\n")

    try:
        trainer.train(
            num_iterations=num_iterations,
            steps_per_collect=steps_per_collect,
            discriminator_updates=disc_updates,
            policy_updates=policy_updates,
            epsilon=epsilon,
            use_validation=use_validation,
            validation_episodes=validation_episodes,
        )
    finally:
        env.close()

    print(f"\nGAIL metrics saved in: {trainer.metrics_csv_path}")
    print(f"Best model automatically saved in: {trainer.best_model_path}")


if __name__ == "__main__":
    # Standalone usage example
    from pathlib import Path

    demo_dir = Path("data/demonstrations")
    model_dir = Path("data/models")

    demo_files = sorted(demo_dir.glob("*.pkl")) if demo_dir.exists() else []
    model_files = sorted(model_dir.glob("*.pth")) if model_dir.exists() else []

    main(demo_files=demo_files, model_files=model_files)
