from torch import nn
import gymnasium as gym

class Policy(nn.Module):

    def __init__(self, env : gym.Env, gamma : float = 0.95, epsilon : float = 0.50):
        super().__init__()
        self.env = env

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
