import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

class Rollout():

    def __init__(self, env : gym.Env, agent : nn.Module, iterations : int = 1024):
        self.env = env
        self.agent = agent
        self.iterations = iterations
        self.returns = None
        self.values = None
        self.advantages = None


    def calculate_returns(self, rewards : torch.Tensor) -> torch.Tensor :

        return 


    def forward_pass(self):
        states, actions, log_probs, values, rewards, done, episode_reward = [], [], [], [], [], False, 0.
        state = self.env.reset()
        # agent.train() # TODO: Y train before Rollout?

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            states.append(state_tensor)

            # Get action probabilities and value prediction from the agent.
            action_pred, value_pred = self.agent(state_tensor)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()

            # Store log probability of the selected action.
            log_probs.append(dist.log_prob(action))
            values.append(value_pred)
            reward = self.env.step(action.item())[1]
            rewards.append(reward)
            episode_reward += float(reward)

        # Convert to tensors and calculate advantages (returns - values).
        states = torch.cat(states)
        log_probs, values, rewards = torch.cat(log_probs), torch.cat(values).squeeze(-1), torch.cat(rewards)
        returns = self.calculate_returns(rewards)
        advantages = returns - values

        return episode_reward, states, torch.cat(actions), log_probs, advantages, returns

    def compute_advantages(self) -> torch.Tensor :
        #Returns are just discounted rewards as was visible in equation 1
        advantages = self.returns - self.values
        # Normalize the advantage
        self.advantages = (advantages - advantages.mean()) / advantages.std()
        return self.advantages