"""
RND-specific Rollout with Separate Advantage Streams.

Canonical RND uses TWO separate return/advantage computations:
1. Extrinsic returns: R^E_t = sum_{k=0}^{inf} gamma^k * r^E_{t+k}
2. Intrinsic returns: R^I_t = sum_{k=0}^{inf} gamma_intrinsic^k * r^I_{t+k}

The combined advantage A_t = A^E_t + A^I_t is used for policy optimization.

This separation is important because:
- Extrinsic rewards are sparse and long-term 
- Intrinsic rewards are dense but non-stationary

RNDRollout class extends Rollout to handle dual reward streams WITHOUT modifying
the base Rollout class (which PPO and RecurrentPPO depend on)
"""

import numpy as np
import torch
from torch import distributions
from torch.nn import functional as F

from src.methods.pure_rl.utils.rollout import Rollout


class RNDRollout(Rollout):
    """
    Rollout class specialized for RND with separate extrinsic/intrinsic advantage streams.
    """
    
    def __init__(self, env, agent, iterations: int = 1024):
        super().__init__(env, agent, iterations)
    
    def calculate_returns_with_gamma(self, rewards: torch.Tensor, indexes: list, gamma: float) -> torch.Tensor:
        """
        Calculate returns with a specific gamma value.
        This allows different discount factors for extrinsic vs intrinsic rewards.
        
        Args:
            rewards: Reward tensor
            indexes: Episode boundary indices
            gamma: Discount factor to use
        """
        with torch.no_grad():
            G = torch.zeros_like(rewards)
            
            start = 0
            for i in indexes:
                l = i - start + 1  # episode length
                discounts = torch.from_numpy(np.power(gamma, np.arange(l))).to(self.agent.device)
                for t in range(l):
                    G[start + t] = (rewards[start + t:i + 1] * discounts[:l - t]).sum()
                start = i + 1

            # Handle remaining steps after last episode boundary
            i = len(rewards) - 1
            l = len(rewards) - start
            if l > 0:
                discounts = torch.from_numpy(np.power(gamma, np.arange(l))).to(self.agent.device)
                for t in range(l):
                    G[start + t] = (rewards[start + t:i + 1] * discounts[:l - t]).sum()

        return G

    def forward_pass_dual_stream(self):
        """
        Forward pass with SEPARATE extrinsic and intrinsic reward streams.
        
        - Track extrinsic and intrinsic rewards separately
        - Compute returns with different gamma values
        - Compute separate advantages for each stream
        - Combine advantages for policy optimization
        
        Returns:
            avg_rewards: Tuple of (avg_extrinsic_reward, avg_total_reward)
            states: State tensor
            actions: Action tensor
            log_probs: Log probability tensor
            combined_advantages: A^E + A^I (for policy loss)
            extrinsic_returns: Returns computed with gamma (for extrinsic critic)
            intrinsic_returns: Returns computed with gamma_intrinsic (for intrinsic critic)
            eps_sizes: Episode sizes
        """
        states, actions, log_probs = [], [], []
        extrinsic_values, intrinsic_values = [], []
        extrinsic_rewards, intrinsic_rewards = [], []
        
        done = False
        total_extrinsic_reward = 0.  # Sum of all extrinsic (env) rewards
        total_intrinsic_reward = 0.  # Sum of all intrinsic (curiosity) rewards
        episode_reward = 0.
        ep_len = 0
        state, _ = self.env.reset()

        i = 0
        indexes, eps_sizes = [], []
        
        while i < self.iterations:
            # Prepare state tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            states.append(state_tensor)

            with torch.no_grad():
                # Get action and SEPARATE value predictions
                action_pred, ext_value, int_value = self.agent.forward_dual_value(state_tensor)
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                action = dist.sample()
                actions.append(action)

            log_probs.append(dist.log_prob(action))
            
            # Take action in environment
            next_state, reward_ext, terminated, truncated, _ = self.env.step(action.item())
            
            episode_reward += float(reward_ext)
            total_extrinsic_reward += float(reward_ext)
            
            # Compute intrinsic reward on NEXT state (canonical RND)
            reward_int = self.agent.compute_intrinsic_reward_normalized(next_state)
            total_intrinsic_reward += float(reward_int)
            
            done = terminated or truncated
           
            if done:
                # Terminal state: values = 0
                extrinsic_values.append(torch.zeros_like(extrinsic_values[0]) if extrinsic_values else torch.zeros(1, device=self.agent.device))
                intrinsic_values.append(torch.zeros_like(intrinsic_values[0]) if intrinsic_values else torch.zeros(1, device=self.agent.device))
                
                state, _ = self.env.reset()
                indexes.append(i)
                eps_sizes.append(ep_len)
                episode_reward = 0.
                ep_len = 0
            else:
                extrinsic_values.append(ext_value)
                intrinsic_values.append(int_value)
            
            # Store separate rewards
            extrinsic_rewards.append(torch.FloatTensor([reward_ext]).to(self.agent.device))
            intrinsic_rewards.append(torch.FloatTensor([reward_int]).to(self.agent.device))
            
            state = next_state
            i += 1
            ep_len += 1

        eps_sizes.append(ep_len)
        num_episodes = max(len(indexes), 1)
        
        # Calculate average rewards per episode for logging
        avg_extrinsic = total_extrinsic_reward / num_episodes  # Avg ENV reward
        avg_intrinsic = total_intrinsic_reward / num_episodes  # Avg CURIOSITY reward  
        avg_total = (total_extrinsic_reward + total_intrinsic_reward) / num_episodes  # Avg TOTAL reward
        
        # Convert to tensors
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        
        extrinsic_values = torch.cat(extrinsic_values).squeeze(-1)
        intrinsic_values = torch.cat(intrinsic_values).squeeze(-1)
        extrinsic_rewards = torch.cat(extrinsic_rewards)
        intrinsic_rewards = torch.cat(intrinsic_rewards)
        
        # Calculate returns with DIFFERENT gamma values
        extrinsic_returns = self.calculate_returns_with_gamma(
            extrinsic_rewards, indexes, self.agent.gamma
        )
        intrinsic_returns = self.calculate_returns_with_gamma(
            intrinsic_rewards, indexes, self.agent.gamma_intrinsic
        )
        
        # Calculate RAW advantages (no normalization yet)
        # We compute: A = R - V without normalizing, to preserve relative scales
        with torch.no_grad():
            extrinsic_advantages = extrinsic_returns - extrinsic_values
            intrinsic_advantages = intrinsic_returns - intrinsic_values
        
        # Combine advantages for policy optimization
        # The intrinsic rewards are already scaled by intrinsic_reward_coeff,
        # so the relative importance is already encoded in the magnitudes.
        
        # to emphasize extrinsic rewards: decrease intrinsic_reward_coeff
        combined_advantages = extrinsic_advantages + intrinsic_advantages
        
        # Normalize ONLY the final combined advantages (standard PPO practice)
        std = combined_advantages.std()
        if std > 1e-8:
            combined_advantages = (combined_advantages - combined_advantages.mean()) / std
        else:
            combined_advantages = combined_advantages - combined_advantages.mean()

        return (
            (avg_extrinsic, avg_intrinsic, avg_total),
            states, 
            actions, 
            log_probs, 
            combined_advantages,
            extrinsic_returns,
            intrinsic_returns,
            eps_sizes
        )
