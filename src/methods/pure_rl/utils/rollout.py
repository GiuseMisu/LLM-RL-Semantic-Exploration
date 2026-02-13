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
        self.iterations = iterations #Number of steps to collect per rollout
        self.max_episode_len = 9999 # PLACEHOLDER


    def calculate_returns(self, rewards : torch.Tensor, indexes : list) -> torch.Tensor :
        with torch.no_grad():
            G = torch.zeros_like(rewards)
            
            start = 0
            for i in indexes:
                l = i-start+1 # episode length
                discounts = torch.from_numpy(np.power(self.agent.gamma, np.arange(l))).to(self.agent.device)
                for t in range(l):
                    G[start+t] = (rewards[start+t:i+1]*discounts[:l-t]).sum()

                start = i+1

            i = len(rewards)-1
            l = len(rewards)-start
            discounts = torch.from_numpy(np.power(self.agent.gamma, np.arange(l))).to(self.agent.device)            
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
    
    def forward_pass(self):
        states, actions, log_probs, values, rewards, done = [], [], [], [], [], False

        # metrics to track
        total_reward = 0.
        total_env_reward = 0.
        total_llm_reward = 0. 
        episode_reward = 0.
        ep_len = 0

        state, _ = self.env.reset()

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
                                    
            # [IMPPP] the reward here is the SHAPED reward from the environment
            # (i.e., may include intrinsic rewards if the env is wrapped accordingly)
            state, reward, terminated, truncated, info = self.env.step(action.item())
            
            # Extract TRUE env reward: use info['env_reward'] if wrapper provides it, else use reward
            # This handles both wrapped (LLM/Eureka) and unwrapped (pure RL) cases
            true_env_reward = info.get('env_reward', reward) if isinstance(info, dict) else reward
            total_env_reward += float(true_env_reward)

            # Extract LLM reward if available (0.0 for pure RL)
            llm_reward = info.get('llm_reward', 0.0) if isinstance(info, dict) else 0.0
            total_llm_reward += float(llm_reward)

            # Track shaped reward (what agent optimizes)
            episode_reward += float(reward)
            total_reward += float(reward)
                                    
            done = terminated or truncated
           
            if done:                   
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
            
            rewards.append(torch.FloatTensor([reward]).to(self.agent.device))
                
            i+=1
            ep_len+=1

        eps_sizes.append(ep_len)
        
        # Convert to tensors and calculate advantages (returns - values).
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs, values, rewards = torch.cat(log_probs), torch.cat(values).squeeze(-1), torch.cat(rewards)
        returns = self.calculate_returns(rewards, indexes)
        advantages = self.calculate_advantages(returns, values)

        # Calculate average rewards per episode
        num_episodes = max(len(indexes) + 1, 1)
        avg_reward = total_reward / num_episodes
        avg_env_reward = total_env_reward / num_episodes
        avg_llm_reward = total_llm_reward / num_episodes

        return (avg_reward, avg_env_reward, avg_llm_reward), states, actions, log_probs, advantages, returns, eps_sizes
    

    def forward_pass_recurrent(self, init_hidden_fn, sequence_length: int = 16):
        """
        Forward pass for recurrent policies (Stores hidden states at the START of each sequence chunk)
        original forward_pass() was for PPO where each state is independent, so no hidden states needed to be stored and passed around
        Now:
        action_pred, value_pred, hidden = self.agent.forward(state_tensor, hidden, ...)
                          ^^^^^^                                   ^^^^^^
                      Updated hidden                          Previous hidden passed in  
        The hidden state flows from step to step, accumulating memory.

        Args:
            init_hidden_fn: Function (defined in ppo_config method of RecurrentPPO class) that initialize hidden states 
            sequence_length: How many steps processed together during training = Length of chunks for TBPTT (Truncated Backprop Through Time) 
        Returns:
            avg_reward: Average reward per episode
            states: All states tensor
            actions: All actions tensor  
            log_probs: Log probabilities tensor
            advantages: Advantages tensor
            returns: Returns tensor
            eps_sizes: List of episode sizes
            hidden_states: List of hidden states at chunk boundaries
            indexes: List of episode end indices
        """
        states, actions, log_probs, values, rewards = [], [], [], [], []
        hidden_states = []  # Store hidden state at start of each chunk
        
        # metrics to track
        total_reward = 0.
        total_env_reward = 0. 
        total_llm_reward = 0.
        episode_reward = 0.
        num_episodes = 0

        state, _ = self.env.reset()
        
        # Initialize hidden state (h: hidden state, c: cell state)
        h, c = init_hidden_fn(1)
        hidden = (h, c) if c is not None else h # handles both LSTM and GRU cases
        
        i = 0
        indexes = []  # Episode end indices
        eps_sizes = []  # Episode lengths
        ep_len = 0                    # How many steps in current episode, needed for reward calculation
        steps_since_chunk_start = 0   # How many steps since last saved hidden state, needed to understand when to save hidden states for training
        
        while i < self.iterations: #iterations = total steps to collect
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            states.append(state_tensor)
            
            # Save hidden state at the start of each sequence chunk => every TBPTT step
            if steps_since_chunk_start % sequence_length == 0:
                if isinstance(hidden, tuple):
                    hidden_states.append((
                        hidden[0].detach().clone(), 
                        hidden[1].detach().clone() if hidden[1] is not None else None
                    ))
                else:
                    hidden_states.append(hidden.detach().clone()) # Removes the tensor from the computation graph and create a copy

            # Forward Pass Call
            with torch.no_grad():
                # Forward returns (action_logits, value, new_hidden)
                action_pred, value_pred, hidden = self.agent.forward(state_tensor, hidden, seq_len=1)
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                action = dist.sample()
                actions.append(action)

            log_probs.append(dist.log_prob(action))
            
            # Normalize value_pred shape to be consistent: always (1,)
            if value_pred.dim() > 1:
                value_pred = value_pred.squeeze(-1)  # (1, 1) -> (1,)
            if value_pred.dim() == 0:
                value_pred = value_pred.unsqueeze(0)  # scalar -> (1,)
                                    
            # [IMPPP] the reward here is the SHAPED reward from the environment
            # (i.e., may include intrinsic rewards if the env is wrapped accordingly)
            state, reward, terminated, truncated, info = self.env.step(action.item())
            
            # Extract TRUE env reward: use info['env_reward'] if wrapper provides it, else use reward
            # This handles both wrapped (LLM/Eureka) and unwrapped (pure RL) cases
            true_env_reward = info.get('env_reward', reward) if isinstance(info, dict) else reward
            total_env_reward += float(true_env_reward)

            # Extract LLM reward if available (0.0 for pure RL)
            llm_reward = info.get('llm_reward', 0.0) if isinstance(info, dict) else 0.0
            total_llm_reward += float(llm_reward)

            episode_reward += float(reward)
            
            done = terminated or truncated
           
            if done:
                # Terminal state: value = 0
                values.append(torch.zeros(1, device=self.agent.device))
                state, _ = self.env.reset()
                indexes.append(i)
                eps_sizes.append(ep_len + 1)  # +1 for current step
                
                total_reward += episode_reward
                num_episodes += 1
                episode_reward = 0.
                ep_len = 0
                
                # Reset hidden state for new episode, otherwise The LSTM remembers things from previous episodes that are irrelevant
                h, c = init_hidden_fn(1)
                hidden = (h, c) if c is not None else h
                steps_since_chunk_start = 0  # Reset chunk counter for new episode
            else:
                values.append(value_pred)
                ep_len += 1
                steps_since_chunk_start += 1
                
            rewards.append(torch.FloatTensor([reward]).to(self.agent.device))
            i += 1

        # Handle incomplete episode at the end
        if ep_len > 0:
            eps_sizes.append(ep_len)
        
        # Calculate average rewards per episode
        num_episodes = max(num_episodes, 1)
        avg_reward = total_reward / num_episodes
        avg_env_reward = total_env_reward / num_episodes
        avg_llm_reward = total_llm_reward / num_episodes 
        
        # Convert to tensors
        states = torch.cat(states)  # (total_steps, 7, 7, 3)
        actions = torch.cat(actions)  # (total_steps,)
        log_probs = torch.cat(log_probs)  # (total_steps,)
        values = torch.cat(values)  # (total_steps,) - should all be 1D now
        rewards = torch.cat(rewards)  # (total_steps,)
        
        # Ensure values is 1D
        if values.dim() > 1:
            values = values.squeeze(-1)
        
        returns = self.calculate_returns(rewards, indexes)
        advantages = self.calculate_advantages(returns, values)

        # Return tuple with (avg_reward, avg_env_reward, avg_llm_reward)
        return ((avg_reward, avg_env_reward, avg_llm_reward), states, actions, log_probs, advantages, returns, 
                eps_sizes, hidden_states, indexes)