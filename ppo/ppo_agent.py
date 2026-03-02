import torch
import torch.optim as optim
from .model import Actor, Critic
from .replay_buffer import ReplayBuffer
from ..ChefsHatGYM.src.agents.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr, gamma, gae_lambda, clip_epsilon, ppo_epochs, batch_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer = ReplayBuffer()

    def request_action(self, observation):
        state = torch.FloatTensor(observation["observation"]).to(self.device)
        action_mask = torch.FloatTensor(observation["action_mask"]).to(self.device)
        dist = self.actor(state, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach()

    def learn(self):
        # ... (PPO learning logic as in previous variants) ...
        pass

    def save_model(self, path):
        torch.save(self.actor.state_dict(), f"{path}/actor.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pt"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pt"))
