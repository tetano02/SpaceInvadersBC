"""Generic (state, action) discriminator for GAIL."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """CNN + MLP discriminator that consumes arbitrary state shapes.

    The module accepts a state tensor and a discrete action, embeds the state with
    either a lightweight CNN (for image inputs) or an MLP (for vector inputs),
    concatenates a one-hot representation of the action, and outputs a single
    logit indicating how "expert-like" the (state, action) pair looks.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        conv_channels: Sequence[int] | None = None,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_actions <= 1:
            raise ValueError("Discriminator requires at least 2 actions.")
        self.observation_shape = tuple(int(dim) for dim in observation_shape)
        self.num_actions = int(num_actions)
        self._is_image = len(self.observation_shape) == 3

        conv_channels = conv_channels or (32, 64, 64)
        if self._is_image:
            self.state_encoder = self._build_cnn_encoder(conv_channels)
            self.state_feature_dim = self._infer_conv_output_dim()
        else:
            flat_dim = int(torch.tensor(self.observation_shape).prod().item())
            self.state_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.GELU(),
            )
            self.state_feature_dim = mlp_hidden_dim

        self.head = nn.Sequential(
            nn.Linear(self.state_feature_dim + self.num_actions, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns a logit per (state, action) pair.

        Args:
            states: Tensor shaped either (B, C, H, W) for image inputs or
                (B, obs_dim) for vector inputs. Values are expected in [0, 1].
            actions: Tensor shaped (B,) with integer actions or (B, num_actions)
                if already one-hot encoded.
        Returns:
            Tensor of shape (B,) containing logits (no sigmoid applied).
        """

        if states.dim() == len(self.observation_shape):
            states = states.unsqueeze(0)
        features = self._encode_state(states)
        action_one_hot = self._encode_actions(actions, states.device, features.dtype)
        logits = self.head(torch.cat([features, action_one_hot], dim=1))
        return logits.squeeze(-1)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _build_cnn_encoder(self, conv_channels: Sequence[int]) -> nn.Sequential:
        layers = []
        in_channels = self.observation_shape[0]
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.BatchNorm2d(out_channels),
                ]
            )
            in_channels = out_channels
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def _infer_conv_output_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *self.observation_shape)
            flat = self.state_encoder(dummy)
        return flat.shape[1]

    def _encode_state(self, states: torch.Tensor) -> torch.Tensor:
        if self._is_image:
            return self.state_encoder(states)
        if states.dim() > 2:
            states = states.view(states.size(0), -1)
        return self.state_encoder(states)

    def _encode_actions(self, actions: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if actions.dim() == 1:
            action_ids = actions.to(torch.long)
            one_hot = F.one_hot(action_ids, num_classes=self.num_actions).to(dtype=dtype)
        else:
            if actions.size(-1) != self.num_actions:
                raise ValueError(
                    f"Action tensor last dimension {actions.size(-1)} does not match num_actions={self.num_actions}."
                )
            one_hot = actions.to(dtype)
        return one_hot.to(device)
