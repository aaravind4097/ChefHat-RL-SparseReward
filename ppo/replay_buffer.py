import numpy as np
import torch

class ReplayBuffer:
    """A buffer for storing trajectories for PPO."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def get_tensors(self, device):
        """Converts stored lists to tensors."""
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions).to(device)
        logprobs = torch.tensor(self.logprobs).to(device)
        return states, actions, logprobs
