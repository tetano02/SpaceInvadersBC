"""
=========================================================
PHASE 0 — PROJECT ANALYSIS
=========================================================
1. Policy networks already available:
   • `BCPolicy` (three-layer CNN), `BCMLPPolicy` (fully-connected), and
     `BCVisionTransformer` (ViT) live in `behavioral_cloning.py` and all emit a
     tensor of shape (batch_size, num_actions). During BC training those logits
     are consumed by `nn.CrossEntropyLoss`, meaning the existing forward pass can
     serve either as class logits (policy-gradient style) or as Q-values by
     interpreting the tensor as action-value estimates.
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
   • Provide a policy-agnostic helper (`infer_policy_output_type` +
     `select_action`) that routes between Q-learning style (epsilon-greedy over
     Q-values) and policy-gradient style (categorical sampling from logits).
   • Implement `Discriminator` (see `discriminator.py`) and a `GAILTrainer` that
     can: load BC checkpoints, convert demonstrations into transition batches,
     collect fresh trajectories from any Gym-compatible env, update the
     discriminator, and update the policy with the appropriate loss function.
   • The trainer exposes modular methods (`collect_agent_trajectories`,
     `update_discriminator`, `compute_gail_rewards`, and two distinct policy
     update paths) so any current or future policy architecture can plug in.
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
from discriminator import Discriminator

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
        # Converti a numpy prima se è un tensor per ridurre memoria
        if isinstance(observation, torch.Tensor):
            if observation.is_cuda:
                observation = observation.cpu()
            # Usa detach senza clone per evitare copia
            arr = observation.detach().numpy() if observation.requires_grad else observation.numpy()
        else:
            arr = np.asarray(observation)
        
        # Controlla se serve normalizzazione prima di convertire
        needs_norm = arr.max() > 1.0
        
        # Converti a float32 e normalizza in un solo passaggio
        if needs_norm:
            tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
        else:
            tensor = torch.from_numpy(arr.astype(np.float32))
        
        # Permuta se necessario
        if tensor.dim() == 3 and tensor.shape[-1] in {1, 3, 4, 6}:
            tensor = tensor.permute(2, 0, 1)
        
        return tensor
    except (RuntimeError, MemoryError) as e:
        if "not enough memory" in str(e) or isinstance(e, MemoryError):
            # Forza garbage collection e riprova
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Out of memory in state preprocessing. Try reducing buffer capacity or batch size. Original error: {e}")
        raise


class FrameStackPreprocessor:
    """Wraps a base preprocessor to provide N-frame stacking."""

    def __init__(self, stack_size: int = 1, base_preprocessor: Callable = default_state_preprocessor) -> None:
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
                self._buffer.append(tensor.clone())  # Clone per evitare reference allo stesso tensor
        else:
            self._buffer.append(tensor)
        # Usa stack invece di cat per essere più efficiente
        result = torch.stack(list(self._buffer), dim=0).flatten(0, 1)
        return result


@dataclass
class Transition:
    """Stores a single transition for replay buffers."""

    state: torch.Tensor  # (C, H, W) or (D,)
    action: int
    env_reward: float
    next_state: torch.Tensor  # matches state shape
    done: bool


class ReplayBuffer:
    """Simple FIFO replay buffer for agent trajectories."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: List[Transition] = []

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        if len(self.buffer) >= self.capacity:
            # Rimuovi e libera esplicitamente l'elemento più vecchio
            old = self.buffer.pop(0)
            del old
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)


# -----------------------------------------------------------------------------
# Policy helpers
# -----------------------------------------------------------------------------


def infer_policy_output_type(policy: nn.Module, explicit: Optional[str] = None) -> str:
    """Infers whether a policy emits Q-values or logits.

    Checks, in order: an explicit override, `policy_output_type`, `output_type`,
    boolean flags like `returns_q_values`, and defaults to "logits".
    """

    if explicit:
        return explicit.lower()
    for attr in ("policy_output_type", "output_type"):
        value = getattr(policy, attr, None)
        if value:
            return str(value).lower()
    if getattr(policy, "returns_q_values", False):
        return "q_values"
    return "logits"


def select_action(
    policy: nn.Module,
    state: torch.Tensor,
    epsilon: float = 0.0,
    output_type: Optional[str] = None,
) -> int:
    """Selects an action using epsilon-greedy (Q) or categorical sampling (logits).

    Args:
        policy: Any policy network producing (1, num_actions) outputs.
        state: Tensor with batch dimension first (1, C, H, W) or (1, D).
        epsilon: Exploration probability for Q-value policies.
        output_type: Optional override ("q_values" or "logits").
    """

    policy_mode = infer_policy_output_type(policy, explicit=output_type)
    policy_device = next(policy.parameters()).device
    state = state.to(policy_device)
    with torch.no_grad():
        outputs = policy(state)
    num_actions = outputs.shape[-1]

    if policy_mode == "q_values":
        if random.random() < epsilon:
            return random.randrange(num_actions)
        action = torch.argmax(outputs, dim=-1)
        return int(action.item())

    # Treat outputs as logits for a categorical distribution.
    if epsilon > 0.0 and random.random() < epsilon:
        return random.randrange(num_actions)
    probs = F.softmax(outputs, dim=-1)
    distribution = Categorical(probs=probs)
    action = distribution.sample()
    return int(action.item())


# -----------------------------------------------------------------------------
# GAIL trainer
# -----------------------------------------------------------------------------


class GAILTrainer:
    """Generic GAIL trainer supporting both Q-learning and policy-gradient policies."""

    def __init__(
        self,
        policy: nn.Module,
        discriminator: Discriminator,
        env,
        expert_demonstrations: Optional[Sequence[dict]] = None,
        expert_files: Optional[Sequence[str | Path]] = None,
        bc_checkpoint_path: Optional[str | Path] = None,
        policy_output_type: Optional[str] = None,
        state_preprocessor: Callable[[np.ndarray | torch.Tensor], torch.Tensor] = default_state_preprocessor,
        agent_buffer_capacity: int = 100_000,
        gamma: float = 0.99,
        policy_lr: float = 1e-4,
        discriminator_lr: float = 1e-4,
        batch_size: int = 64,
        device: Optional[str] = None,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = policy.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.env = env
        env_id = getattr(getattr(self.env, "spec", None), "id", "unknown_env")
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.state_preprocessor = state_preprocessor
        self.policy_output_type = infer_policy_output_type(self.policy, explicit=policy_output_type)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.discriminator_loss = nn.BCEWithLogitsLoss()

        self.agent_buffer = ReplayBuffer(agent_buffer_capacity)
        self.data_manager = DataManager()
        self.run_timestamp, self.run_id = self.data_manager.create_run_identifier()
        self.metrics_csv_path = self.data_manager.get_metrics_filepath(self.run_timestamp, self.run_id)
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None
        self.iteration_history: List[dict] = []
        self.base_model_path = str(bc_checkpoint_path) if bc_checkpoint_path else None
        self.training_metadata = {
            "training_type": "gail",
            "policy_output_type": self.policy_output_type,
            "base_model_path": self.base_model_path,
            "base_model_name": Path(self.base_model_path).name if self.base_model_path else "N/A",
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
            with torch.no_grad():  # Evita accumulo di gradienti
                action = select_action(self.policy, state, epsilon=epsilon, output_type=self.policy_output_type)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            next_state = self._prepare_state(next_obs)

            transition = Transition(
                state=state.squeeze(0).detach().cpu(),
                action=int(action),
                env_reward=float(reward),
                next_state=next_state.squeeze(0).detach().cpu(),
                done=done,
            )
            self.agent_buffer.push(transition)

            episode_return += reward
            steps_collected += 1
            
            # Libera memoria dello stato precedente
            del state
            
            # Pulizia periodica della memoria ogni 500 step
            if steps_collected % 500 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        
        # Pulisci riferimenti ai tensori
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
        labels = torch.cat([
            torch.ones_like(logits_expert),
            torch.zeros_like(logits_agent),
        ])
        logits = torch.cat([logits_expert, logits_agent])

        loss = self.discriminator_loss(logits, labels)
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            expert_acc = (torch.sigmoid(logits_expert) > 0.5).float().mean().item()
            agent_acc = (torch.sigmoid(logits_agent) < 0.5).float().mean().item()

        # Libera memoria dei tensori intermedi
        del expert_states, expert_actions, agent_states, agent_actions
        del logits_expert, logits_agent, labels, logits
        
        return {
            "loss": float(loss.item()),
            "expert_acc": expert_acc,
            "agent_acc": agent_acc,
        }

    def compute_gail_rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns shaped rewards derived from the discriminator."""

        logits = self.discriminator(states.to(self.device), actions.to(self.device))
        probs = torch.sigmoid(logits)
        rewards = -torch.log(1.0 - probs + 1e-8)
        return rewards.detach()

    def update_policy(self, batch: dict, gail_rewards: torch.Tensor) -> dict:
        """Delegates to the appropriate policy update rule."""

        if self.policy_output_type == "q_values":
            loss = self._update_policy_q_learning(batch, gail_rewards)
            return {"policy_loss": loss, "mode": "q_learning"}
        loss = self._update_policy_policy_gradient(batch, gail_rewards)
        return {"policy_loss": loss, "mode": "policy_gradient"}

    def train(
        self,
        num_iterations: int,
        steps_per_collect: int = 2048,
        discriminator_updates: int = 5,
        policy_updates: int = 1,
        epsilon: float = 0.05,
        log_interval: int = 1,
    ) -> None:
        """Runs the full GAIL training loop."""

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
                rollout_stats = self.collect_agent_trajectories(steps_per_collect, epsilon=epsilon)
                disc_losses = []
                for _ in range(discriminator_updates):
                    expert_batch = self.sample_expert_batch()
                    agent_batch = self.sample_agent_batch()
                    disc_metrics = self.update_discriminator(expert_batch, agent_batch)
                    disc_losses.append(disc_metrics)
                    # Libera batch dopo l'uso
                    del expert_batch, agent_batch

                agent_batch = self.sample_agent_batch()
                gail_rewards = self.compute_gail_rewards(agent_batch["states"], agent_batch["actions"])
                policy_metrics = None
                for _ in range(policy_updates):
                    policy_metrics = self.update_policy(agent_batch, gail_rewards)

                avg_disc_loss = sum(m["loss"] for m in disc_losses) / max(len(disc_losses), 1)
                avg_expert_acc = sum(m["expert_acc"] for m in disc_losses) / max(len(disc_losses), 1)
                avg_agent_acc = sum(m["agent_acc"] for m in disc_losses) / max(len(disc_losses), 1)
                
                mean_reward = sum(rollout_stats["episode_rewards"]) / len(rollout_stats["episode_rewards"])
                self._record_iteration_metrics(
                    iteration=iteration,
                    disc_loss=avg_disc_loss,
                    expert_acc=avg_expert_acc,
                    agent_acc=avg_agent_acc,
                    policy_loss=policy_metrics["policy_loss"] if policy_metrics else 0.0,
                    policy_mode=policy_metrics["mode"] if policy_metrics else "n/a",
                    mean_return=mean_reward,
                    steps=rollout_stats["steps"],
                    epsilon=epsilon,
                    disc_updates=discriminator_updates,
                    policy_updates=policy_updates,
                    gail_reward_mean=float(gail_rewards.mean().item()) if gail_rewards.numel() > 0 else 0.0,
                )
                
                # Pulisci memoria DOPO aver registrato le metriche
                del agent_batch, gail_rewards, disc_losses
                
                # Pulisci cache CUDA ogni iterazione
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                gc.collect()  # Forza garbage collection

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
                "discriminator_updates": self._last_train_config.get("discriminator_updates"),
                "policy_updates": self._last_train_config.get("policy_updates"),
                "epsilon": self._last_train_config.get("epsilon"),
            }
        )
        if self.training_start_time and self.training_end_time:
            duration = (self.training_end_time - self.training_start_time).total_seconds()
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

    def _prepare_expert_transitions(self, demonstrations: Sequence[dict]) -> List[Transition]:
        transitions: List[Transition] = []
        for episode in demonstrations:
            self._reset_state_preprocessor()
            observations = episode["observations"]
            actions = episode["actions"]
            rewards = episode.get("rewards")
            dones = episode.get("dones")
            encoded_states: List[torch.Tensor] = []
            for obs in observations:
                tensor = self.state_preprocessor(obs)
                encoded_states.append(tensor.detach().cpu().clone())

            if not encoded_states:
                continue

            episode_len = len(actions)
            last_index = len(encoded_states) - 1
            for idx in range(episode_len):
                state = encoded_states[min(idx, last_index)]
                next_idx = min(idx + 1, last_index)
                next_state = encoded_states[next_idx]
                reward = float(rewards[idx]) if rewards is not None else 0.0
                done = bool(dones[idx]) if dones is not None else (idx == episode_len - 1)
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
        # Costruisci liste invece di stack immediato per ridurre picchi di memoria
        states_list = [t.state for t in transitions]
        actions_list = [t.action for t in transitions]
        next_states_list = [t.next_state for t in transitions]
        dones_list = [t.done for t in transitions]
        rewards_list = [t.env_reward for t in transitions]
        
        # Crea tensori direttamente sul device target
        states = torch.stack(states_list).to(self.device)
        actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        next_states = torch.stack(next_states_list).to(self.device)
        dones = torch.tensor(dones_list, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        
        # Pulisci liste
        del states_list, actions_list, next_states_list, dones_list, rewards_list
        
        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "dones": dones,
            "env_rewards": rewards,
        }

    def _update_policy_q_learning(self, batch: dict, gail_rewards: torch.Tensor) -> float:
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        gail_rewards = gail_rewards.to(self.device).squeeze(-1)

        q_values = self.policy(states)
        current_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.policy(next_states)
            max_next_q = next_q.max(dim=1).values
            targets = gail_rewards + self.gamma * (1.0 - dones) * max_next_q
        loss = F.mse_loss(current_q, targets)
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        return float(loss.item())

    def _update_policy_policy_gradient(self, batch: dict, gail_rewards: torch.Tensor) -> float:
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


USAGE_EXAMPLE = """
Example: loading a BC policy and running GAIL
------------------------------------------------
import torch
from behavioral_cloning import build_policy
from discriminator import Discriminator
from env_make import make_space_invaders_env
from gail_trainer import GAILTrainer

policy = build_policy(model_type="dqn", num_actions=6)
policy.policy_output_type = "q_values"  # mark as Q-network

disc = Discriminator(observation_shape=(6, 210, 160), num_actions=6)

env = make_space_invaders_env(render_mode=None)
trainer = GAILTrainer(
    policy=policy,
    discriminator=disc,
    env=env,
    expert_files=["data/demonstrations/dem_YYMMDD_ID.pkl"],
    bc_checkpoint_path="data/models/best_model.pth",
    policy_output_type="q_values",
)
trainer.train(num_iterations=100)
trainer.save_checkpoint()
"""
