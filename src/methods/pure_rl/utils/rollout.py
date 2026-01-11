import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from .policy import Policy

class Rollout():

    def __init__(self, env : gym.Env, agent : Policy, iterations : int = 1024):
        self.env = env
        self.agent = agent
        self.iterations = iterations
        self.max_episode_len = 9999 # PLACEHOLDER

        # if self.agent.rollout != self:
        #     raise Exception("Rollout's agent's rollout must be same as rollout, very simple...")


    def calculate_returns(self, rewards : torch.Tensor, indexes : list) -> torch.Tensor :
        with torch.no_grad():
            G = torch.zeros_like(rewards)
            # discounts = torch.from_numpy(np.power(self.agent.gamma, np.arange(len(rewards))))

            # for t in range(len(rewards)):
            #     G[t] = (rewards[t:]*discounts[:len(rewards)-t]).sum()

            start = 0
            for i in indexes:
                l = i-start+1 # episode length
                discounts = torch.from_numpy(np.power(self.agent.gamma, np.arange(l)))
                for t in range(l):
                    G[start+t] = (rewards[start+t:i+1]*discounts[:l-t]).sum()

                start = i+1

            i = len(rewards)-1
            l = len(rewards)-start
            discounts = torch.from_numpy(np.power(self.agent.gamma, np.arange(l)))            
            for t in range(l):
                G[start+t] = (rewards[start+t:i+1]*discounts[:l-t]).sum()

        return G
    
    def calculate_advantages(self, returns : torch.Tensor, values : torch.Tensor) -> torch.Tensor :
        with torch.no_grad():
            advantages = returns - values
            # Normalize the advantage
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / std
            else:
                #prevent division by zero
                advantages = advantages - advantages.mean()  # Just center, don't normalize
        
        
        return advantages
    
    # FIXME: Is this even different from the two above? 
    def calculate_advantages_GAE(self, rewards : torch.Tensor, values : torch.Tensor) -> torch.Tensor :
        advantages = torch.zeros_like(values)
        last_advantage = 0
        last_value = values[-1].item()
        with torch.no_grad():
            for t in reversed(range(values.shape[0])):
                delta = rewards[t] + self.agent.gamma * last_value - values[t]
                last_advantage = delta + self.agent.gamma * self.agent._lambda * last_advantage
                advantages[t] = last_advantage
                last_value = values[t]                

        return advantages


    def forward_pass(self):
        states, actions, log_probs, values, rewards, done = [], [], [], [], [], False
        total_reward = episode_reward = avg_reward = 0.
        ep_len = 0
        state, _ = self.env.reset()
        # agent.train() # TODO: Y train before Rollout?

        i = 0
        indexes, eps_sizes = [], []
        while i < self.iterations:

            # for MiniGrid 7x7x3 input MiniGrid: (7, 7, 3) -> (1, 7, 7, 3)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            states.append(state_tensor)

            with torch.no_grad():
                # Get action probabilities and value prediction from the agent.
                action_pred, value_pred = self.agent.get_act(state_tensor)
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                action = dist.sample()
                actions.append(action)

            # Store log probability of the selected action.
            log_probs.append(dist.log_prob(action))

            
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
           
            if done:
                #debug mid epoch print
                #if terminated:
                    #print(f"Episode: {i+1}/{self.iterations} env solved, Reward {reward}")
                # if truncated:
                #     print(f"Episode: {i+1}/{self.iterations}: env truncated, Reward {reward}")
                    
                values.append(torch.zeros_like(values[0]))
                state, _ = self.env.reset()
                indexes.append(i) # saves where an episode ends
                eps_sizes.append(ep_len)
                episode_reward = 0.
                ep_len = 0
            elif ep_len >= self.max_episode_len:
                values.append(value_pred)
                state, _ = self.env.reset()
                indexes.append(i) # saves where an episode ends
                eps_sizes.append(ep_len)
                episode_reward = 0.
                ep_len = 0
            else:
                values.append(value_pred)
            
            rewards.append(torch.FloatTensor([reward]))
            episode_reward += float(reward)
            total_reward += float(reward)
                
            i+=1
            ep_len+=1

        eps_sizes.append(ep_len)
        avg_reward = total_reward/(len(indexes)+1)
        #print(f"average reward {avg_reward}")

        # Convert to tensors and calculate advantages (returns - values).
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs, values, rewards = torch.cat(log_probs), torch.cat(values).squeeze(-1), torch.cat(rewards)
        returns = self.calculate_returns(rewards, indexes)
        advantages = self.calculate_advantages(returns, values)

        return avg_reward, states, actions, log_probs, advantages, returns, eps_sizes