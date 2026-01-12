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

import math

"""
Policy Gradient for PPO
cite: https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7
"""
class PPO(Policy):
    
    def __init__(
            self, env : gym.Env, 
            gamma : float = 0.99, 
            epsilon : float = 0.2,
            input_dim : int = 8, 
            output_dim : int = 4, 
            epochs : int = 100
            ):

        super().__init__(env=env, gamma=gamma, epsilon=epsilon)

        self.name = 'PPO'

        self.actor = BaseNet(input_dim, output_dim)
        self.critic = BaseNet(input_dim)

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
    
    def get_act(self, state : torch.Tensor):
        return self.forward(state)

    def get_surrogate_loss(self, 
                           actions_log_probability_old : torch.Tensor, 
                           actions_log_probability_new : torch.Tensor,
                           advantages : torch.Tensor                           
                           ) -> torch.Tensor :

        advantages = advantages.detach()

        policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()

        surrogate_loss_full = policy_ratio * advantages
        surrogate_loss_clamped = torch.clamp(policy_ratio, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantages
        surrogate_loss = torch.min(surrogate_loss_full, surrogate_loss_clamped)
        
        return surrogate_loss 

    def get_loss(self, surrogate_loss : torch.Tensor, entropy : torch.Tensor, returns : torch.Tensor, value_pred : torch.Tensor):
        # We calulate entropy and total policy by equation 2 and 4
        entropy_bonus = self.entropy_coeff * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).mean()
        value_loss = F.smooth_l1_loss(returns, value_pred).mean()
        return policy_loss, value_loss

    def step(self, states : torch.Tensor, actions : torch.Tensor, old_log_probs : torch.Tensor, advantages : torch.Tensor, returns : torch.Tensor):
        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, returns),
            batch_size=self.batch_size, shuffle=True # shuffle=True seems to work better
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

            print(f"\nEpoch {e+1}/{self.epochs} | Episode Average Reward: {episode_reward:.2f}\n")

            if episode_reward > 0 and episode_reward > max_rew:
                print(f"Good reward {episode_reward}, at epoch {e}, saving...")
                max_rew = episode_reward
                self.save() 

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
            output_dim : int = 4, 
            encode_dim : int = 8, 
            hidden_dim : int = 64, 
            epochs : int = 100, 
            recurrence : str = "lstm"
            ):
        
        super().__init__(env=env, gamma=gamma, epsilon=epsilon, input_dim=hidden_dim, output_dim=output_dim, epochs=epochs)

        # TODO: for 3D state it will need CNN Encoder
        #self.encoder = CNN(input_dim, encode_dim) # CNN

        self.name = "RPPO"

        self.hidden_dim = hidden_dim
        self.recurrence = recurrence
        self.cell = None

        if self.recurrence == "lstm":
            self.recurrent = nn.LSTM(encode_dim, hidden_size = self.hidden_dim, batch_first = True)
        elif self.recurrence == "gru":
            self.recurrent = nn.GRU(encode_dim, hidden_size = self.hidden_dim, batch_first = True)
        

    def forward(self, state : torch.Tensor, cell : torch.Tensor | tuple[torch.Tensor, torch.Tensor | None], seq_len : int = 1):
        
        #x = self.encoder(state) # TODO: forse non serve per input 1-D, per N-D cambiare con CNN e usare flatten
        x = state

        if seq_len == 1:
            x, cell = self.recurrent(x.unsqueeze(1), cell)
            x = x.squeeze(1)
        else:
            x = x.reshape((x.shape[0]//seq_len), seq_len, x.shape[1])

            x, cell = self.recurrent(x, cell)
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])

        return self.actor(x), self.critic(x), cell
    
    def get_act(self, state : torch.Tensor):
        if self.cell == None:
            h,c = self.init_cells(1)
            if c is not None:
                self.cell = (h,c)
            else:
                self.cell = h
        a, v, cell = self.forward(state, self.cell)
        self.cell = cell
        return a, v
    
    def step(self, states : torch.Tensor, actions : torch.Tensor, old_log_probs : torch.Tensor, advantages : torch.Tensor, returns : torch.Tensor, eps_sizes : list):
        
        dataset = self.batch_episodes(states, actions, old_log_probs, advantages, returns, eps_sizes)

        h,c = self.init_cells(len(eps_sizes))
        if c is not None:
            cell = (h, c)
        else:
            cell = h

        for _ in range(self.steps):
            j = 0
            for batch in dataset:

                j+=1
                batch_states, batch_actions, old_probs, adv, ret = batch
                batch_states, batch_actions, old_probs, adv, ret = batch_states.transpose(0,1), batch_actions.transpose(0,1), old_probs.transpose(0,1), adv.transpose(0,1), ret.transpose(0,1)

                batch_states = batch_states.reshape(batch_states.shape[0]*batch_states.shape[1],-1)

                action_pred, value_pred, cell = self.forward(batch_states, cell=cell, seq_len=self.batch_size)
                
                value_pred = value_pred.squeeze(-1).view(ret.shape)
                
                action_pred = action_pred.view(-1, self.batch_size, action_pred.shape[1])

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
                (policy_loss + value_loss).backward(retain_graph=True)
                self.optimizer.step()

    def trainer(self):
        max_rew = -float("inf")
        for e in range(self.epochs):
            
            episode_reward, states, actions, log_probs, advantages, returns, eps_sizes = self.rollout.forward_pass()
            self.cell = None

            print(f"\nEpoch {e+1}/{self.epochs} | Episode Average Reward: {episode_reward:.2f}\n")
            if episode_reward > 0 and episode_reward > max_rew:
                print(f"Good reward {episode_reward}, at epoch {e}, saving...")
                max_rew = episode_reward
                self.save() 

            self.step(states, actions, log_probs, advantages, returns, eps_sizes)

    def init_cells(self, num_sequences : int):
        hxs = torch.zeros((num_sequences), self.hidden_dim, dtype=torch.float32).unsqueeze(0)
        cxs = None
        if self.recurrence == "lstm":
            cxs = torch.zeros((num_sequences), self.hidden_dim, dtype=torch.float32).unsqueeze(0)
        return hxs, cxs
    
    def batch_episodes(self, states : torch.Tensor, actions : torch.Tensor, old_log_probs : torch.Tensor, advantages : torch.Tensor, returns : torch.Tensor, eps_sizes : list):
        # Prepares data for recurrent network training:
        #   -episodes are stacked on top of each other
        #   -length is padded to be multiple of batch size
        # TODO: (possibly) make more efficient
        # Create DataLoader for mini-batches    
        states_per_seq = list(states.split(eps_sizes, dim = 0))
        actions_per_seq = list(actions.split(eps_sizes, dim = 0))
        old_log_probs_per_seq = list(old_log_probs.split(eps_sizes, dim = 0))
        advantages_per_seq = list(advantages.split(eps_sizes, dim = 0))
        returns_per_seq = list(returns.split(eps_sizes, dim = 0))

        # maximum number of rows among the tensors
        max_rows = max(tensor.size(0) for tensor in states_per_seq)
        max_rows = math.ceil(max_rows/self.batch_size) * self.batch_size
  
        for n, _ in enumerate(states_per_seq):
            sz = states_per_seq[n].size(0)
            states_per_seq[n] = torch.nn.functional.pad(states_per_seq[n], (0, 0, 0, max_rows - sz))
            actions_per_seq[n] = torch.nn.functional.pad(actions_per_seq[n], (0, max_rows - sz))
            old_log_probs_per_seq[n] = torch.nn.functional.pad(old_log_probs_per_seq[n], (0, max_rows - sz))
            advantages_per_seq[n] = torch.nn.functional.pad(advantages_per_seq[n], (0, max_rows - sz))
            returns_per_seq[n] = torch.nn.functional.pad(returns_per_seq[n], (0, max_rows - sz))

        # Stack padded tensors
        states = torch.stack(states_per_seq, dim=0)
        actions = torch.stack(actions_per_seq, dim=0)
        old_log_probs = torch.stack(old_log_probs_per_seq, dim=0)
        advantages = torch.stack(advantages_per_seq, dim=0)
        returns = torch.stack(returns_per_seq, dim=0)

        dataset = DataLoader(
             TensorDataset(states.transpose(0,1), actions.transpose(0,1), old_log_probs.detach().transpose(0,1), advantages.transpose(0,1), returns.transpose(0,1)),
             batch_size=self.batch_size, shuffle=False
        )
        
        return dataset
    
"""
PPO but Random Network Distillation.
A version of PPO better suited for sparse reward environments
cite: https://www.emergentmind.com/topics/random-network-distillation-rnd
"""
class RNDPPO(PPO):
    def __init__(
            self, env : gym.Env, 
            gamma : float = 0.99, 
            epsilon : float = 0.2,
            input_dim : int = 8, 
            output_dim : int = 4, 
            epochs : int = 100
            ):

        super().__init__(env=env, gamma=gamma, epsilon=epsilon, input_dim=input_dim, output_dim=output_dim)

        self.name = 'RNDPPO'

        # TODO: Like in RPPO for 3D state it will need CNN Encoder
        #self.encoder = ...

        self.random_state_network = BaseNet(input_dim, input_dim)
        self.intrinsic_value_head = BaseNet(input_dim)

        # hyperparameters
        self.lr = 1e-3
        self.epochs = epochs
        self.batch_size = 128
        self.entropy_coeff = 0.02
        self.steps = 10
        # ...

        self.intrinsic_reward = None

        self.optimizer = Adam(self.parameters(), lr = self.lr)

        self.rollout = Rollout(self.env, self)

        
    def forward(self, state : torch.Tensor):
        # encoded_state = self.encoder(state)
        action = self.actor(state)  
        # action = self.actor(encoded_state)
        with torch.no_grad():
            self.random_state_network.eval()
            random_state = self.random_state_network(state)

        extrinsic_value = self.critic(state) # self.critic(encoded_state)
        intrinsic_value = self.intrinsic_value_head(random_state)

        self.intrinsic_reward = F.mse_loss(state, random_state).item()

        return action, extrinsic_value+intrinsic_value
    
    def augment_reward(self, reward: float):
        if self.intrinsic_reward is None:
            raise TypeError(f"Forward Pass required before computing full reward")
        
        augmented_reward = self.intrinsic_reward+reward
        self.intrinsic_reward = None

        return augmented_reward