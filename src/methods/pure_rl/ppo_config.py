import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym

from pure_rl.utils.network import BaseNet
from pure_rl.utils.policy import Policy
from pure_rl.utils.rollout import Rollout

"""
Policy Gradient for PPO
cite: https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7
"""
class PPO(Policy):
    
    def __init__(self, env : gym.Env, input_dim : int):

        super().__init__(env)

        self.actor = BaseNet(input_dim)
        self.critic = BaseNet(input_dim)

        # hyperparameters
        self.lr = 1e-3
        # ...

        self.rollout = Rollout(self.env, self)

    def get_surrogate_loss(self, 
                           actions_log_probability_old : torch.Tensor, 
                           actions_log_probability_new : torch.Tensor,
                           advantages : torch.Tensor                           
                           ) -> torch.Tensor :

        advantages = advantages.detach()

        policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()

        # TODO: write a meaningful comment
        surrogate_loss_full = policy_ratio * advantages
        surrogate_loss_clamped = torch.clamp(policy_ratio, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantages
        surrogate_loss = torch.min(surrogate_loss_full, surrogate_loss_clamped)
        
        return surrogate_loss 

    def get_loss(self, surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
        # We calulate entropy and total policy by equation 2 and 4
        entropy_bonus = entropy_coefficient * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        return policy_loss, value_loss

        
