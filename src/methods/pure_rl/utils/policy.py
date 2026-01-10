from abc import abstractmethod
import torch
from torch import nn
import gymnasium as gym

class Policy(nn.Module):

    def __init__(self, env : gym.Env, gamma : float = 0.99, epsilon : float = 0.50):
        super().__init__()
        self.env = env

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self._lambda = 0.5

    def save(self):
        torch.save(self.state_dict(), "./policy.pkl")

    def load(self):
        self.load_state_dict(torch.load("./policy.pkl", weights_only=True))

    @abstractmethod
    def get_act(self, state):
        pass


