import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym

from utils.network import BaseNet
from utils.policy import Policy
from utils.rollout import Rollout

"""
Policy Gradient for PPO
cite: https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7
"""
class PPO(Policy):
    
    def __init__(self, env : gym.Env, gamma : float = 0.99, epsilon : float = 0.99, input_dim : int = 8, output_dim : int = 4, epochs : int = 100):

        super().__init__(env=env, gamma=gamma, epsilon=epsilon)

        self.name = 'PPO'

        self.actor = BaseNet(input_dim, output_dim) # CNN
        self.critic = BaseNet(input_dim) # CNN

        # hyperparameters
        self.lr = 1e-3
        self.epochs = epochs
        self.batch_size = 128
        self.entropy_coeff = 0.02
        self.steps = 10
        # ...

        self.optimizer = Adam(self.parameters(), lr = self.lr)

        self.rollout = Rollout(self.env, self)

    def forward(self, state : torch.Tensor):
        return self.actor(state), self.critic(state)

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

    def get_loss(self, surrogate_loss : torch.Tensor, entropy : torch.Tensor, returns : torch.Tensor, value_pred : torch.Tensor):
        # We calulate entropy and total policy by equation 2 and 4
        entropy_bonus = self.entropy_coeff * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        return policy_loss, value_loss

    def step(self, states : torch.Tensor, actions : torch.Tensor, old_log_probs : torch.Tensor, advantages : torch.Tensor, returns : torch.Tensor):
        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, returns),
            batch_size=self.batch_size, shuffle=False # shuffle=True seems to work better
        )

        for _ in range(self.steps):
            j = 0
            for batch in dataset:
                j+=1
                batch_states, batch_actions, old_probs, adv, ret = batch
                action_pred, value_pred = self.forward(batch_states)
                value_pred = value_pred.squeeze(-1)

                # Calculate new action probabilities and entropy.
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Calculate policy loss (surrogate loss) and value loss.
                surrogate_loss = self.get_surrogate_loss(old_probs, new_log_probs, adv)
                policy_loss, value_loss = self.get_loss(surrogate_loss, entropy, ret, value_pred)

                # Backpropagate and update weights.
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                self.optimizer.step()

    def trainer(self):
        max_rew = -float("inf")
        for e in range(self.epochs):
            
            episode_reward, states, actions, log_probs, advantages, returns, _ = self.rollout.forward_pass()
            if episode_reward > 0 and episode_reward > max_rew:
                print(f"Good reward {episode_reward}, at epoch {e}, saving...")
                max_rew = episode_reward
                self.save() 

            if e%100 == 0:
                print(f"running epoch {e}")
                print(f"avg reward {episode_reward}")

            self.step(states, actions, log_probs, advantages, returns)

"""
PPO but recurrent
cite: 
@inproceedings{
  pleines2023memory,
  title={Memory Gym: Partially Observable Challenges to Memory-Based Agents},
  author={Marco Pleines and Matthias Pallasch and Frank Zimmer and Mike Preuss},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=jHc8dCx6DDr}
}
"""
# Still a work in progress
class RecurrentPPO(PPO):
    def __init__(
            self, 
            env : gym.Env, 
            gamma : float = 0.99, 
            epsilon : float = 0.99, 
            input_dim : int = 8, 
            output_dim : int = 4, 
            encode_dim : int = 8, 
            hidden_dim : int = 64, 
            epochs : int = 100, 
            recurrence : str = "lstm"
            ):
        
        super().__init__(env=env, gamma=gamma, epsilon=epsilon, input_dim=input_dim, output_dim=output_dim, epochs=epochs)

        #self.encoder = BaseNet(input_dim, encode_dim) # CNN

        self.hidden_dim = hidden_dim
        self.recurrence = recurrence

        if self.recurrence == "lstm":
            self.recurrent = nn.LSTM(encode_dim, hidden_size = self.hidden_dim, batch_first = True)
        elif self.recurrence == "gru":
            self.recurrent = nn.GRU(encode_dim, hidden_size = self.hidden_dim, batch_first = True)
        

    def forward(self, state : torch.Tensor, cell : torch.Tensor, seq_len : int):
        
        #x = self.encoder(state) # TODO: forse non serve per input 1-D, per N-D cambiare con CNN e usare flatten
        x = state

        if seq_len == 1:
            x, cell = self.recurrent(x.unsqueeze(1), cell)
            x = x.squeeze(1)
        else:
            x = x.reshape((x.shape[0]//seq_len), seq_len, x.shape[1])

            x, cell = self.recurrent(x, cell)
            x = x.reshape(x.shae[0]*x.shape[1], x.shape[2])

        return self.actor(x), self.critic(x), cell
    
    def step(self, states : torch.Tensor, actions : torch.Tensor, old_log_probs : torch.Tensor, advantages : torch.Tensor, returns : torch.Tensor):
        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, returns),
            batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.steps):
            j = 0
            for batch in dataset:
                j+=1
                batch_states, batch_actions, old_probs, adv, ret = batch
                action_pred, value_pred = self.forward(batch_states)
                value_pred = value_pred.squeeze(-1)

                # Calculate new action probabilities and entropy.
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Calculate policy loss (surrogate loss) and value loss.
                surrogate_loss = self.get_surrogate_loss(old_probs, new_log_probs, adv)
                policy_loss, value_loss = self.get_loss(surrogate_loss, entropy, ret, value_pred)

                # Backpropagate and update weights.
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                self.optimizer.step()

    def trainer(self):
        for e in range(self.epochs):
            
            episode_reward, states, actions, log_probs, advantages, returns, indexes = self.rollout.forward_pass()

            if e%100 == 0:
                print(f"running epoch {e}")
                print(f"episode reward {episode_reward}")

            self.step(states, actions, log_probs, advantages, returns)

    def init_cells(self, num_sequences : int):
        hxs = torch.zeros((num_sequences), self.hidden_dim, dtype=torch.float32).unsqueeze(0)
        cxs = None
        if self.recurrence == "lstm":
            cxs = torch.zeros((num_sequences), self.hidden_dim, dtype=torch.float32).unsqueeze(0)
        return hxs, cxs
    
    #def batch_episodes(self, )