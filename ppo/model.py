import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    """The Actor network for the PPO agent."""
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        """Forward pass with action masking."""
        probs = self.network(state)
        # Apply a mask to invalid actions, then re-normalize
        masked_probs = probs * action_mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        return Categorical(masked_probs)

class Critic(nn.Module):
    """The Critic network for the PPO agent."""
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get the state value."""
        return self.network(state)
