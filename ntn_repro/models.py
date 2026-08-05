from __future__ import annotations

import math
from typing import Sequence

from .deps import require_torch


def mlp(input_dim: int, output_dim: int, hidden_sizes: Sequence[int]):
    torch = require_torch()
    nn = torch.nn
    layers: list[nn.Module] = []
    last = input_dim
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(last, int(hidden)), nn.ReLU()])
        last = int(hidden)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 512):
        torch = require_torch()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.pe = pe.unsqueeze(0)


class TransformerTrajectoryModel:
    def __init__(
        self,
        history_length: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        torch = require_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.history_length = history_length
                self.horizon = horizon
                self.input_proj = nn.Linear(3, d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
                # The paper predicts one position at t + horizon (Eq. 8), not
                # every intermediate position from t + 1 through t + horizon.
                self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
                pe = PositionalEncoding(d_model, max_len=max(history_length + 1, 512)).pe
                self.register_buffer("positional_encoding", pe)

            def forward(self, x):
                x = self.input_proj(x)
                x = x + self.positional_encoding[:, : x.shape[1], :]
                encoded = self.encoder(x)
                return self.head(encoded[:, -1, :])

        self.module = _Model()


class ActorCriticNet:
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        torch = require_torch()
        nn = torch.nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = mlp(state_dim, action_dim, hidden_sizes)
                self.critic = mlp(state_dim, 1, hidden_sizes)

            def forward(self, state):
                return self.actor(state), self.critic(state).squeeze(-1)

            def act(self, state, action_mask=None):
                logits, value = self.forward(state)
                if action_mask is not None:
                    logits = logits.masked_fill(~action_mask, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                return action, dist.log_prob(action), dist.entropy(), value

        self.module = _Net()


class DQNNet:
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        self.module = mlp(state_dim, action_dim, hidden_sizes)

