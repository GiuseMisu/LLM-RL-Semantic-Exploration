from abc import abstractmethod
import torch
from torch import nn
import gymnasium as gym

class Policy(nn.Module):

    def __init__(self, env : gym.Env, 
                 gamma : float = 0.99, 
                 epsilon : float = 0.50,
                 model_name : str = "policy"
                 ):
        super().__init__()
        self.env = env

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self._lambda = 0.5

        self.model_path = f"./{model_name}.pkl"

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        print(f"[LOAD] Loading model from {self.model_path}")
        self.load_state_dict(torch.load(self.model_path, weights_only=True))

    @abstractmethod
    def get_act(self, state):
        pass


