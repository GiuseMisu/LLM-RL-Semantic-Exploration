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
            epsilon : float = 0.2,
            epochs : int = 100,
            output_dim : int | None = None, 
            encode_dim : int = 128,          
            model_name : str = "PPO"            
            ):

        super().__init__(env=env, gamma=gamma, epsilon=epsilon, model_name=model_name)

        # detect action space if missing
        if output_dim is None:
            output_dim = int(env.action_space.n) 

        # CNN encoder for MiniGrid observations
        self.encoder = MiniGridCNN(output_dim=encode_dim, device=self.device)
        
        # actor-critic use encoded features (dim = output_dim)
        self.actor = BaseNet(input_dim=encode_dim, output_dim=output_dim, device=self.device)
        self.critic = BaseNet(input_dim=encode_dim, output_dim=1, device=self.device)
        
        # hyperparameters
        self.lr = 1e-3
        self.epochs = epochs
        self.batch_size = 128
        self.entropy_coeff = 0.02
        self.steps = 10
        # ...

        self.optimizer = Adam(self.parameters(), lr = self.lr)

        self.rollout = Rollout(self.env, self)
        print(f"[{self.name}]")

    def forward(self, state : torch.Tensor) -> tuple:
        """
        Forward pass for action selection and value estimation.
        """
        state = state.to(self.device)
        encoded_state = self.encoder(state)
        action_logits = self.actor(encoded_state)
        value = self.critic(encoded_state)

        return action_logits, value
    
    def get_act(self, state : torch.Tensor) -> tuple:
        """
        Returns only action and value computed via forward
        """
        return self.forward(state)

    def get_surrogate_loss(self, 
                           actions_log_probability_old : torch.Tensor, 
                           actions_log_probability_new : torch.Tensor,
                           advantages : torch.Tensor                           
                           ) -> torch.Tensor :
        """
        Computes surrogate loss
        """
        advantages = advantages.detach()

        policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()

        surrogate_loss_full = policy_ratio * advantages
        surrogate_loss_clamped = torch.clamp(policy_ratio, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantages
        surrogate_loss = torch.min(surrogate_loss_full, surrogate_loss_clamped) 

        return surrogate_loss 

    def get_loss(self, surrogate_loss : torch.Tensor, entropy : torch.Tensor,
                  returns : torch.Tensor, value_pred : torch.Tensor) -> tuple:
        """
        Computes PPO loss
        """        
        # We calulate entropy and total policy by equation 2 and 4
        entropy_bonus = self.entropy_coeff * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).mean()
        value_loss = F.smooth_l1_loss(returns, value_pred).mean()

        return policy_loss, value_loss

    def step(self, states : torch.Tensor, actions : torch.Tensor, old_log_probs : 
             torch.Tensor, advantages : torch.Tensor, returns : torch.Tensor) -> None:
        """
        Single training step for training
        """        
        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, returns),
            batch_size=self.batch_size, shuffle=True # shuffle=True seems to work better
        )

        for _ in range(self.steps):
            for batch in dataset:
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
        """
        Training loop for PPO
        """
        max_rew = -float("inf")
        consecutive_epochs_mean_reward = []

        for e in range(self.epochs):
            
            episode_reward, states, actions, log_probs, advantages, returns, _ = self.rollout.forward_pass()
            if episode_reward[0] > max_rew:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward per Episode: {episode_reward[0]:.5f} ==> New best reward, saving")
                max_rew = episode_reward[0]
                self.save() 
            else:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward per Episode: {episode_reward[0]:.5f}")
                
            #print(f"Epoch {e+1}/{self.epochs} | Average Augmented Reward per Episode: {episode_reward[1]:.5f}")

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

            self.step(states.to(self.device), actions.to(self.device), log_probs.to(self.device), advantages.to(self.device), returns.to(self.device))

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
   
#==============================
# NEW RecurrentPPO
#=============================

class RecurrentPPO(PPO):
    '''
    Standard PPO treats each observation independently (Markov assumption), 
    but this fails when the agent needs memory to make good decisions, so we add recurrence.
    This class extends PPO with recurrent layers (LSTM/GRU) to handle partial observability.
    maintaining a hidden state that summarizes past observations help the agent make better decisions.
    
    Implementation creates Separate Sequences and Save Hidden States
    es Episode 1: [s0, s1, s2, ..., s47]  Episode 2: [s48, s49, s50, ..., s100] (new episode, agent exploring)
    If you naively fed [s0, s1, ..., s100] as one long sequence to the LSTM:
    Problem 1: The LSTM would think s48 follows from s47, carrying over memory from episode 1 into episode 2, but they're completely unrelated
    Problem 2: Backpropagating through 100+ steps causes vanishing gradients and GPU memory explosion.
   
    To solve this TBPTT -> splits episodes in smaller chunks of fixed length
    Idea: Don't backpropagate through the entire episode. Instead, split into chunks and backpropagate only within each chunk!!!!
    es episode length = 48, chunk size = 16 
    => Sequences: chunck1=[s0-s15], chunck2=[s16-s31], chunck3=[s32-s47] => only backprop through 16 steps at a time for each chunk
    
    [WARNING] When you process Chunk 2 [s16-s31], the LSTM needs to "know" what happened in [s0-s15] => YOU MUST SAVE THE HIDDEN STATE FOR EACH CHUNK START
    when TRAINING Chunk 2 [s16-s31] it Start with saved h_15 = the boundary hidden state after processing s15

    [WARNING] Mask Padding => Last chunk may be shorter than chunk size (e.g. episode ends in the middle of a chunk), pad with zeros to maintain consistent input size
    the paddings are not real usefull data so when computing loss we must ignore them => create a mask that indicates which parts are real data vs padding
    '''
    def __init__(
            self, 
            env: gym.Env, 
            gamma: float = 0.99, 
            epsilon: float = 0.2,  
            epochs: int = 100,
            output_dim: int | None = None, # => si puo togliere come sopra
            encode_dim: int = 128,  # =>  CNN output size
            hidden_dim: int = 128,  # Match encode_dim for no bottleneck
            sequence_length: int = 16,  # TBPTT sequence length
            recurrence: str = "lstm",
            model_name: str = "RecurrentPPO"  #with ppo recurrent save also the type of recurrence
            ):
        
        # Call PPO init but we'll override some components
        super().__init__(
            env=env, 
            gamma=gamma, 
            epsilon=epsilon, 
            output_dim=output_dim,
            encode_dim=encode_dim, 
            epochs=epochs,
            model_name=model_name + "_" + recurrence
        )

        if output_dim is None: # detect action space if missing
            output_dim = int(env.action_space.n)
        
        self.hidden_dim = hidden_dim
        self.recurrence = recurrence
        self.sequence_length = sequence_length
        
        # Lower learning rate for recurrent networks
        self.lr = 3e-4
        
        # Recurrent layer
        if self.recurrence == "lstm":
            self.recurrent = nn.LSTM(encode_dim, hidden_dim, batch_first=True, device=self.device)
        elif self.recurrence == "gru":
            self.recurrent = nn.GRU(encode_dim, hidden_dim, batch_first=True, device=self.device)

        #Orthogonal Initialization for LSTM
        for name, param in self.recurrent.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.layer_norm = nn.LayerNorm(hidden_dim, device=self.device)
        
        # Actor/Critic use hidden_dim
        self.actor = BaseNet(input_dim=hidden_dim, output_dim=output_dim, device=self.device)
        self.critic = BaseNet(input_dim=hidden_dim, output_dim=1, device=self.device)
        
        # Recreate optimizer with all parameters and new lr
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        
        print(f"[{self.name}] hidden_dim={hidden_dim}, seq_len={sequence_length}, lr={self.lr}")

    def init_hidden(self, batch_size: int = 1):
        """Initialize hidden states to zeros"""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        c = None
        if self.recurrence == "lstm":
            c = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        return h, c

    def forward(self, state: torch.Tensor, hidden=None, seq_len: int = 1):
        """
        Forward pass with recurrent processing.
        
        Args:
            state: Input state tensor
            hidden: Tuple (h, c) for LSTM or h for GRU
            seq_len: Sequence length for reshaping batched sequences
        """
        state = state.to(self.device)
        
        x = self.encoder(state)  # CNN encoding (batch, encode_dim)
        
        # Handle hidden state format
        if hidden is None:
            batch_size = x.shape[0] // seq_len
            hidden = self.init_hidden(batch_size)
        
        if isinstance(hidden, tuple) and hidden[1] is None:
            hidden = hidden[0]
        
        if seq_len == 1:
            # Single step: x is (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
            x, new_hidden = self.recurrent(x, hidden)
            x = self.layer_norm(x) # to stabilize training
            x = x.squeeze(1)  # (batch, features)
        else: 
            # Batch of sequences: reshape for recurrent processing
            batch_size = x.shape[0] // seq_len
            x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, features)
            x, new_hidden = self.recurrent(x, hidden)
            x = self.layer_norm(x) # to stabilize training
            x = x.reshape(-1, self.hidden_dim)  # (batch*seq_len, hidden_dim)
        
        # Ensure hidden is always tuple format
        if not isinstance(new_hidden, tuple):
            new_hidden = (new_hidden, None)
            
        action_logits = self.actor(x)
        value = self.critic(x)
        
        return action_logits, value, new_hidden
    
    def get_act(self, state: torch.Tensor):
        """Get action for inference (single step)"""
        if not hasattr(self, '_hidden') or self._hidden is None:
            self._hidden = self.init_hidden(1)
        
        action, value, self._hidden = self.forward(state, self._hidden, seq_len=1)
        return action, value
    
    def reset_hidden(self):
        """Reset hidden state (call at episode start during evaluation)"""
        self._hidden = None

    def prepare_sequences(self, states, actions, log_probs, advantages, returns, 
                          eps_sizes, hidden_states, episode_ends):
        """
        Training RNN on long sequences is memory-intensive-> solution is TBPTT
        Instead of backpropagating through entire episodes, the data is split into chunks
        Each processed indip with its corresponding initial hidden state
        To do so must guarantees that Sequences NEVER cross episode boundaries

        1. Splits data into chunks of sequence_length
        2. Respects episode boundaries - Never crosses from one episode to another
        3. Pads short sequences - If a chunk is shorter than sequence_length, pad with zeros
        4. Creates masks - Track which positions are real vs padded
        5. Hidden State Alignment: Each sequence stores the hidden state that was active at its start (captured during rollout)
        Returns sequences, masks and corresponding hidden states.

        es: seq_size = 16
        Episode 1: [s0, s1, ..., s47]           |   Episode 2: [s48, s49, ..., s100]
        Sequences: [s0-s15], [s16-s31],         |   [s32-s47+padding], [s48-s63], ...
        Masks:     [1,1,...,1], [1,1,...,1],    |   [1,...,1,0,0,0], [1,1,...,1], ...
        """
        total_steps = states.shape[0]
        seq_len = self.sequence_length
        
        # We'll create sequences that respect episode boundaries
        sequences_states = []
        sequences_actions = []
        sequences_log_probs = []
        sequences_advantages = []
        sequences_returns = []
        sequences_masks = [] #=> CRUCIAL TO UNDERSTAND WHICH PART IS PADDING AND WHICH IS REAL DATA
        sequences_hidden = []
                
        # Track which hidden state corresponds to which position
        hidden_idx = 0        
        i = 0
        # following is the while loop where we create sequences
        while i < total_steps:
            # Determine sequence end (either seq_len steps or episode boundary)
            seq_end = min(i + seq_len, total_steps)
            
            # Check for episode boundaries within this sequence
            for ep_end in episode_ends:
                if i <= ep_end < seq_end:
                    seq_end = ep_end + 1
                    break
            
            actual_len = seq_end - i
            
            # Extract sequence
            seq_states = states[i:seq_end]
            seq_actions = actions[i:seq_end]
            seq_log_probs = log_probs[i:seq_end]
            seq_advantages = advantages[i:seq_end]
            seq_returns = returns[i:seq_end]
            
            # Create mask (1 for valid, 0 for padding)
            mask = torch.ones(seq_len, device=self.device)
            
            # Pad if necessary
            if actual_len < seq_len:
                pad_len = seq_len - actual_len
                seq_states = torch.cat([seq_states, torch.zeros(pad_len, *states.shape[1:], device=self.device)])
                seq_actions = torch.cat([seq_actions, torch.zeros(pad_len, dtype=actions.dtype, device=self.device)])
                seq_log_probs = torch.cat([seq_log_probs, torch.zeros(pad_len, device=self.device)])
                seq_advantages = torch.cat([seq_advantages, torch.zeros(pad_len, device=self.device)])
                seq_returns = torch.cat([seq_returns, torch.zeros(pad_len, device=self.device)])
                mask[actual_len:] = 0
            
            sequences_states.append(seq_states)
            sequences_actions.append(seq_actions)
            sequences_log_probs.append(seq_log_probs)
            sequences_advantages.append(seq_advantages)
            sequences_returns.append(seq_returns)
            sequences_masks.append(mask)
            
            # Get corresponding hidden state-> were saved at chunk boundaries during rollout
            if hidden_idx < len(hidden_states):
                sequences_hidden.append(hidden_states[hidden_idx])
                hidden_idx += 1
            else:
                # Fallback: use zero hidden state
                h, c = self.init_hidden(1)
                sequences_hidden.append((h, c) if c is not None else h)
            
            i = seq_end
        
        return (
            torch.stack(sequences_states),      # (num_seqs, seq_len, 7, 7, 3)
            torch.stack(sequences_actions),     # (num_seqs, seq_len)
            torch.stack(sequences_log_probs),   # (num_seqs, seq_len)
            torch.stack(sequences_advantages),  # (num_seqs, seq_len)
            torch.stack(sequences_returns),     # (num_seqs, seq_len)
            torch.stack(sequences_masks),       # (num_seqs, seq_len)
            sequences_hidden                    # list of hidden states
        )

    def step(self, states, actions, old_log_probs, advantages, returns, 
             eps_sizes, hidden_states, episode_ends):
        """Training step with proper sequence handling"""
        
        # Prepare sequences that respect episode boundaries and the corresponding hidden states
        (seq_states, seq_actions, seq_log_probs, seq_advantages, 
         seq_returns, seq_masks, seq_hidden) = self.prepare_sequences(
            states, actions, old_log_probs, advantages, returns,
            eps_sizes, hidden_states, episode_ends
        )
        
        num_sequences = seq_states.shape[0]
        seq_len = self.sequence_length
        
        # Create indices for shuffling sequences (not timesteps within sequences!)
        indices = np.arange(num_sequences)
        
        for _ in range(self.steps):
            # Shuffle sequence order each iteration
            np.random.shuffle(indices)
            
            # Process in mini-batches of sequences
            batch_size = min(self.batch_size // seq_len, num_sequences)  # Number of sequences per batch
            batch_size = max(1, batch_size)
            
            #=========================
            # for loop to group multiple sequences together, concatenating their hidden states
            #=========================
            for start in range(0, num_sequences, batch_size):
                end = min(start + batch_size, num_sequences)
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = seq_states[batch_indices]      # (batch, seq_len, 7, 7, 3)
                batch_actions = seq_actions[batch_indices]    # (batch, seq_len)
                batch_old_probs = seq_log_probs[batch_indices]
                batch_adv = seq_advantages[batch_indices]
                batch_ret = seq_returns[batch_indices]
                batch_mask = seq_masks[batch_indices]
       
                # Get hidden states for this batch
                batch_hidden_list_h = []
                batch_hidden_list_c = []
                
                for idx in batch_indices:
                    h_state = seq_hidden[idx]
                    if isinstance(h_state, tuple):
                        batch_hidden_list_h.append(h_state[0])
                        if h_state[1] is not None:
                            batch_hidden_list_c.append(h_state[1])
                    else:
                        batch_hidden_list_h.append(h_state)
                
                batch_hidden_h = torch.cat(batch_hidden_list_h, dim=1)
                
                if self.recurrence == "lstm" and len(batch_hidden_list_c) > 0:
                    batch_hidden_c = torch.cat(batch_hidden_list_c, dim=1)
                    batch_hidden = (batch_hidden_h, batch_hidden_c)
                else:
                    batch_hidden = batch_hidden_h
                
                # Flatten for CNN: (batch * seq_len, 7, 7, 3)
                flat_states = batch_states.view(-1, *batch_states.shape[2:])
                
                # Forward pass
                action_pred, value_pred, _ = self.forward(
                    flat_states, 
                    hidden=batch_hidden,
                    seq_len=seq_len
                )
                
                # Reshape outputs: (batch * seq_len, ...) -> (batch, seq_len, ...)
                curr_batch = end - start
                action_pred = action_pred.view(curr_batch, seq_len, -1)
                value_pred = value_pred.view(curr_batch, seq_len)
                
                # Flatten for loss computation
                batch_actions_flat = batch_actions.view(-1)
                
                # Compute new log probs and entropy
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob.view(-1, action_prob.shape[-1]))
                new_log_probs = dist.log_prob(batch_actions_flat).view(curr_batch, seq_len)
                entropy = dist.entropy().view(curr_batch, seq_len)
                

                #================ [IMP] Masked Loss Computation ====================
                #Only valid (non-padded) timesteps contribute to the loss
                # Apply mask to losses-> important for ignore padded parts (not real data)
                #===================================================================
                # Surrogate loss
                ratio = (new_log_probs - batch_old_probs).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_adv
                surrogate_loss = torch.min(surr1, surr2)
                
                # Masked policy loss
                entropy_bonus = self.entropy_coeff * entropy
                policy_loss = -((surrogate_loss + entropy_bonus) * batch_mask).sum() / batch_mask.sum()
                
                # Masked value loss
                value_loss = ((batch_ret - value_pred) ** 2 * batch_mask).sum() / batch_mask.sum()
                
                # Backprop 
                self.optimizer.zero_grad()
                # Actor (policy): Learns which actions to take / Critic (value): Learns to estimate expected returns
                # Their losses have different magnitudes and scales: Policy loss (surrogate loss) is typically small, Value loss can be larger
                # 0.5 Prevents value function from dominating / The original PPO paper use 0.5
                (policy_loss + 0.5 * value_loss).backward()

                # Gradient clipping (important for RNNs)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                
                self.optimizer.step()

    def trainer(self, early_stopping_threshold: float = 0.95, window_size: int = 10):
        """Training loop for Recurrent PPO"""
        max_rew = -float("inf")
        consecutive_epochs_mean_reward = []

        for e in range(self.epochs):
            # specialized rollout that captures hidden states it collects: 
            # Standard RL data (states, actions, rewards) / Hidden states at chunk boundaries / Episode sizes and episode end indices
            (episode_reward, states, actions, log_probs, advantages, returns, 
             eps_sizes, hidden_states, episode_ends) = self.rollout.forward_pass_recurrent(
                init_hidden_fn=self.init_hidden,
                sequence_length=self.sequence_length
            )
            
            # Reset hidden for next rollout
            self._hidden = None

            if episode_reward > max_rew:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward: {episode_reward:.5f} ==> New best reward, saving")
                max_rew = episode_reward
                self.save() 
            else:
                print(f"Epoch {e+1}/{self.epochs} | Average Reward: {episode_reward:.5f}")

            consecutive_epochs_mean_reward.append(episode_reward)
            if len(consecutive_epochs_mean_reward) > window_size:
                consecutive_epochs_mean_reward.pop(0)
            
            if len(consecutive_epochs_mean_reward) == window_size:
                avg_recent = np.mean(consecutive_epochs_mean_reward)
                if avg_recent >= early_stopping_threshold:
                    print(f"\nEARLY STOPPING at epoch {e+1}")
                    print(f"Avg reward over {window_size} epochs: {avg_recent:.5f}")
                    break

            # Training step
            self.step(
                states.to(self.device), 
                actions.to(self.device), 
                log_probs.to(self.device), 
                advantages.to(self.device), 
                returns.to(self.device),
                eps_sizes,
                hidden_states,
                episode_ends
            )