import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from utils.policy import Policy

class Rollout():

    def __init__(self, env : gym.Env, agent : Policy, iterations : int = 1024):
        self.env = env
        self.agent = agent
        self.iterations = iterations

        # if self.agent.rollout != self:
        #     raise Exception("Rollout's agent's rollout must be same as rollout, very simple...")


    def calculate_returns(self, rewards : torch.Tensor) -> torch.Tensor :
        G = torch.zeros_like(rewards)
        discounts = torch.from_numpy(np.power(self.agent.gamma, np.arange(len(rewards))))

        for t in range(len(rewards)):
            G[t] = (rewards[t:]*discounts[:len(rewards)-t]).sum()

        return G
    
    def calculate_advantages(self, returns : torch.Tensor, values : torch.Tensor) -> torch.Tensor :
        #
        advantages = returns - values
        # Normalize the advantage
        self.advantages = (advantages - advantages.mean()) / advantages.std()
        return self.advantages


    def forward_pass(self):
        states, actions, log_probs, values, rewards, done, episode_reward = [], [], [], [], [], False, 0.
        state, _ = self.env.reset()
        # agent.train() # TODO: Y train before Rollout?

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            states.append(state_tensor)

            # Get action probabilities and value prediction from the agent.
            action_pred, value_pred = self.agent(state_tensor)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            actions.append(action)

            # Store log probability of the selected action.
            log_probs.append(dist.log_prob(action))
            values.append(value_pred)
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            rewards.append(torch.FloatTensor([reward]))
            episode_reward += float(reward)

        # Convert to tensors and calculate advantages (returns - values).
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs, values, rewards = torch.cat(log_probs), torch.cat(values).squeeze(-1), torch.cat(rewards)
        returns = self.calculate_returns(rewards)
        advantages = self.calculate_advantages(returns, values)

        return episode_reward, states, actions, log_probs, advantages, returns