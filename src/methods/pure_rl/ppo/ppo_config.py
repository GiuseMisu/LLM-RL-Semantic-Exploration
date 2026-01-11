import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym

import numpy as np

from src.methods.pure_rl.utils.network import BaseNet, MiniGridCNN
from src.methods.pure_rl.utils.policy import Policy
from src.methods.pure_rl.utils.rollout import Rollout

import math

"""
Policy Gradient for PPO
cite: https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7
"""
class PPO(Policy):
    
    def __init__(
            self, env : gym.Env, 
            gamma : float = 0.99, 
            epsilon : float = 0.2,  # old 0.99
            epochs : int = 100,
            output_dim : int = 7, #==> SI PUO TOGLIERE entrambi environment hanno 7 azioni c'é autodetect          
            model_name : str = "PPO_model"            
            ):

        super().__init__(env=env, gamma=gamma, epsilon=epsilon, model_name=model_name)

        # detect action space if missing => minigird has 7 actions
        if output_dim is None:
            output_dim = env.action_space.n 
        
        self.name = 'PPO'

        # CNN encoder for MiniGrid observations
        self.encoder = MiniGridCNN(output_dim=128)
        
        # actor-critic use encoded features (dim = output_dim)
        self.actor = BaseNet(input_dim=128, output_dim=output_dim)
        self.critic = BaseNet(input_dim=128, output_dim=1)
        

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
        features = self.encoder(state)  # (batch, 7, 7, 3) -> (batch, 128)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_act(self, state : torch.Tensor):
        return self.forward(state)

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

    def trainer(self, early_stopping_threshold: float = 0.95, window_size: int = 10):
        max_rew = -float("inf")
        consecutive_epochs_mean_reward = []

        for e in range(self.epochs):
            
            episode_reward, states, actions, log_probs, advantages, returns, _ = self.rollout.forward_pass()
            if episode_reward > max_rew:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward per Episode: {episode_reward:.5f} ==> New best reward, saving")
                max_rew = episode_reward
                self.save() 
            else:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward per Episode: {episode_reward:.5f}")

            consecutive_epochs_mean_reward.append(episode_reward)
            if len(consecutive_epochs_mean_reward) > window_size:
                consecutive_epochs_mean_reward.pop(0)
            
            if len(consecutive_epochs_mean_reward) == window_size: # check if enough data
                avg_recent = np.mean(consecutive_epochs_mean_reward)
                if avg_recent >= early_stopping_threshold:
                    print(f"\nEARLY STOPPING TRIGGERED at epoch {e+1}")
                    print(f"Average reward over last {window_size} epochs: {avg_recent:.5f}")
                    print(f"Threshold: {early_stopping_threshold}\n")
                    ## Don't save again - best model already saved self.save()  
                    break

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
            epsilon : float = 0.2,  
            epochs : int = 100,

            output_dim : int = 7, # => si puo togliere come sopra
            encode_dim : int = 128, # =>  CNN output size
            hidden_dim : int = 64, 
             
            recurrence : str = "lstm",
            model_name : str = "RecurrentPPO_model" 
            ):
        
        super().__init__(env=env, 
                         gamma=gamma, 
                         epsilon=epsilon, 
                         output_dim=output_dim, 
                         epochs=epochs,
                         model_name=model_name+"_"+recurrence #with ppo recurrent save also the type of recurrence
                         )

        # Auto-detect action space if not provided
        if output_dim is None:
            output_dim = env.action_space.n

        # old code self.encoder = BaseNet(input_dim, encode_dim) # CNN
        # IMPORTANT: Override the encoder from PPO
        self.encoder = MiniGridCNN(output_dim=encode_dim)  # 7×7×3 -> 128 D
        
        self.hidden_dim = hidden_dim
        self.recurrence = recurrence
        self.cell = None

        if self.recurrence == "lstm":
            self.recurrent = nn.LSTM(encode_dim, hidden_size = self.hidden_dim, batch_first = True)
        elif self.recurrence == "gru":
            self.recurrent = nn.GRU(encode_dim, hidden_size = self.hidden_dim, batch_first = True)
        
        # Re-initialize actor/critic to use hidden_dim (64D) instead of encode_dim
        self.actor = BaseNet(input_dim=self.hidden_dim, output_dim=output_dim or env.action_space.n)
        self.critic = BaseNet(input_dim=self.hidden_dim, output_dim=1)
        
        # Re-create optimizer to include all new parameters
        self.optimizer = Adam(self.parameters(), lr=self.lr)

    def forward(self, state : torch.Tensor, cell : torch.Tensor | tuple[torch.Tensor, torch.Tensor | None], seq_len : int = 1):
  
        x = self.encoder(state) # MiniGridCNN per input 7x7x3        

        if seq_len == 1:
            # single step
            x, cell = self.recurrent(x.unsqueeze(1), cell)
            x = x.squeeze(1)
        else:
            # batch of sequences, to reshape in (sequences, seq_len, features)
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
            sz = states_per_seq[n].size(0)  # Current episode length    
            # Pad states (4D): [episode_len, 7, 7, 3]
            # Padding tuple is read from last dim to first: (dim3, dim2, dim1, dim0)
            states_per_seq[n] = torch.nn.functional.pad(
                states_per_seq[n], 
                (0, 0, 0, 0, 0, 0, 0, max_rows - sz))  # Only pad dim0 (episode_len) at the end
            
            
            # Pad 1D tensors: [episode_len]
            actions_per_seq[n] = torch.nn.functional.pad(
                actions_per_seq[n], 
                (0, max_rows - sz))  # Pad dim0 at the end
            
            old_log_probs_per_seq[n] = torch.nn.functional.pad(
                old_log_probs_per_seq[n], 
                (0, max_rows - sz))
            
            advantages_per_seq[n] = torch.nn.functional.pad(
                advantages_per_seq[n], 
                (0, max_rows - sz))
            
            returns_per_seq[n] = torch.nn.functional.pad(
                returns_per_seq[n], 
                (0, max_rows - sz))
            

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

        h,c = self.init_cells(states.shape[0])
        cell = (h, c)

        for _ in range(self.steps):
            j = 0
            for batch in dataset:

                j+=1
                batch_states, batch_actions, old_probs, adv, ret = batch
                batch_states, batch_actions, old_probs, adv, ret = (
                    batch_states.transpose(0,1), 
                    batch_actions.transpose(0,1), 
                    old_probs.transpose(0,1), 
                    adv.transpose(0,1), 
                    ret.transpose(0,1)
                )

                # for CNN Flatten batch and sequence dims
                # From (num_seqs, seq_len, 7, 7, 3) -> (num_seqs*seq_len, 7, 7, 3)
                batch_size_flat = batch_states.shape[0] * batch_states.shape[1]
                spatial_dims = batch_states.shape[2:]  # (7, 7, 3)
                batch_states = batch_states.reshape(batch_size_flat, *spatial_dims)
                
                action_pred, value_pred, cell = self.forward(
                    batch_states, cell=cell, 
                    seq_len=min(self.batch_size, max_rows)
                    )
                
                value_pred = value_pred.squeeze(-1).view(ret.shape)                
                action_pred = action_pred.view(-1, min(self.batch_size, max_rows), action_pred.shape[1])

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

                #old code (policy_loss + value_loss).backward(retain_graph=True)

                (policy_loss + value_loss).backward() # No retain_graph
                self.optimizer.step()

                # Detach cell state to break gradient history
                # prevents "modified by in-place operation" error that occured
                # Each forward pass through LSTM updates the hidden state: h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})
                # If keep the same cell state across multiple backward passes
                # But in PPO, each mini-batch should be independent
                cell = (cell[0].detach(), cell[1].detach())


    def trainer(self, early_stopping_threshold: float = 0.95, window_size: int = 10):
        max_rew = -float("inf")
        consecutive_epochs_mean_reward = []

        for e in range(self.epochs):
            
            episode_reward, states, actions, log_probs, advantages, returns, eps_sizes = self.rollout.forward_pass()
            self.cell = None

            if episode_reward > max_rew:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward per Episode: {episode_reward:.5f} ==> New best reward, saving")
                max_rew = episode_reward
                self.save() 
            else:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward per Episode: {episode_reward:.5f}")

            consecutive_epochs_mean_reward.append(episode_reward)
            if len(consecutive_epochs_mean_reward) > window_size:
                consecutive_epochs_mean_reward.pop(0)
            
            if len(consecutive_epochs_mean_reward) == window_size: # check if enough data
                avg_recent = np.mean(consecutive_epochs_mean_reward)
                if avg_recent >= early_stopping_threshold:
                    print(f"\nEARLY STOPPING TRIGGERED at epoch {e+1}")
                    print(f"Average reward over last {window_size} epochs: {avg_recent:.5f}")
                    print(f"Threshold: {early_stopping_threshold}\n")
                    ## Don't save again - best model already saved self.save()  
                    break
            
            self.step(states, actions, log_probs, advantages, returns, eps_sizes)

    def init_cells(self, num_sequences : int):
        hxs = torch.zeros((num_sequences), self.hidden_dim, dtype=torch.float32).unsqueeze(0)
        cxs = None
        if self.recurrence == "lstm":
            cxs = torch.zeros((num_sequences), self.hidden_dim, dtype=torch.float32).unsqueeze(0)
        return hxs, cxs
    