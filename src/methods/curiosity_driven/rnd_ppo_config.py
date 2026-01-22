import os
import warnings
# ---  SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")


import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
import numpy as np

from src.methods.pure_rl.ppo.ppo_config import PPO
from src.methods.pure_rl.utils.network import BaseNet, MiniGridCNN
from src.methods.pure_rl.utils.rollout import Rollout

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
            gamma_intrinsic : float = 0.99,  # Separate discount for intrinsic rewards
            output_dim : int | None = None,  # Auto-detect from env
            encode_dim : int = 128,
            rnd_dim : int = 128,
            epochs : int = 100,
            model_name : str = "RNDPPO"
            ):

        super().__init__(
                        env=env, 
                        gamma=gamma,
                        epsilon=epsilon,
                        epochs=epochs,
                        output_dim=output_dim,
                        encode_dim=encode_dim,
                        model_name=model_name
                        )

        # RND feature dimension
        self.rnd_feature_dim = rnd_dim
        
        # Target Network: Random, FROZEN (never trained)
        # Produces deterministic random features for states
        self.rnd_target = MiniGridCNN(output_dim=self.rnd_feature_dim, device=self.device)

        #Initialize with larger scale for better RND signal
        for module in self.rnd_target.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        for param in self.rnd_target.parameters():
            param.requires_grad = False  # Freeze target network        

        # Predictor Network: Trained to predict target network's output
        self.rnd_predictor = MiniGridCNN(output_dim=self.rnd_feature_dim, device=self.device)
        # Initialize predictor
        for module in self.rnd_predictor.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        #===========the reward from novelty=============
        # Intrinsic value head: Estimates expected intrinsic returns
        self.intrinsic_critic = BaseNet(input_dim=encode_dim, output_dim=1, device=self.device)
        
        # hyperparameters
        self.lr = 1e-3
        self.epochs = epochs
        self.batch_size = 128
        self.entropy_coeff = 0.02
        self.steps = 10        

        # RND-specific hyperparameters
        self.intrinsic_reward_coeff = 0.005 # Scale intrinsic rewards
        self.rnd_loss_coeff = 0.5   # Weight for RND predictor loss
        self.gamma_intrinsic = gamma_intrinsic # separate discount for intrinsic rewards
        
        # statistics for intrinsic reward normalization
        # different states would have wildly different magnitudes:
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0
        self.intrinsic_reward_count = 0
        self._m2 = 0.0  # For sum of squared deviations

        self.optimizer = Adam([p for n, p in self.named_parameters() if "rnd_target" not in n], lr=self.lr)

        self.rollout = Rollout(self.env, self)

        #CHECK
        #print(f"[RNDPPO] intrinsic_reward_coeff={self.intrinsic_reward_coeff}, rnd_loss_coeff={self.rnd_loss_coeff}")

    def compute_intrinsic_reward(self, state: torch.Tensor) -> tuple:
        """
        Computes intrinsic reward
        Intrinsic reward = MSE between predictor (ENV) and target outputs (NOVELTY)
        """
        with torch.no_grad():
            target_features = self.rnd_target(state)        
        predictor_features = self.rnd_predictor(state)                
        intrinsic_rewards = ((predictor_features - target_features) ** 2).mean(dim=1) # MSE per sample
        
        return intrinsic_rewards, predictor_features, target_features
            
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass for action selection and value estimation.
        Also computes and stores intrinsic reward for augment_reward().
        """
        state = state.to(self.device)
        # Encode state for policy
        encoded_state = self.encoder(state)  # (batch, encode_dim)

        # Compute action logits and values
        action_logits = self.actor(encoded_state)
        extrinsic_value = self.critic(encoded_state) # REWARD FROM ENV
        intrinsic_value = self.intrinsic_critic(encoded_state) # REWARD FROM NOVELTY
        
        # Combined value (for advantage computation)
        value = extrinsic_value + intrinsic_value
        
        # Compute intrinsic reward for this state (used by augment_reward)
        with torch.no_grad():
            intrinsic_rewards, _, _ = self.compute_intrinsic_reward(state)
            # Store for augment_reward (single value for single state)
            self._current_intrinsic_reward = intrinsic_rewards.mean().item()       

        return action_logits, value
        
    def augment_reward(self, reward: float) -> float:
        """
        Augment extrinsic reward (THE ENV REW) with normalized intrinsic reward (THE NOVELTY REW)
        """
        if self._current_intrinsic_reward is None:
            raise RuntimeError("Forward pass required before augment_reward()")
        
        raw_intrinsic = self._current_intrinsic_reward
        
        # Update running statistics
        self._update_intrinsic_stats(raw_intrinsic)
        
        # Normalize to prevent early explosion
        # 0.1 to prevent division by tiny numbers
        effective_std = max(self.intrinsic_reward_std, 0.1)
        normalized_intrinsic = raw_intrinsic / effective_std
        
        # Clip to prevent extreme outliers (keep rewards reasonable)
        normalized_intrinsic = np.clip(normalized_intrinsic, 0.0, 10.0)
        
        # Apply coefficient
        scaled_intrinsic = self.intrinsic_reward_coeff * normalized_intrinsic
        
        augmented_reward = reward + scaled_intrinsic        
        self._current_intrinsic_reward = None

        return augmented_reward

    def _update_intrinsic_stats(self, intrinsic_reward: float) -> None:
        """
        algorithm for running mean and std
        needed to have a intrinsic reward (NOVELTY REWARD) with STABLE distribution
        """
        self.intrinsic_reward_count += 1

        # Update RUNNING mean
        delta = intrinsic_reward - self.intrinsic_reward_mean
        self.intrinsic_reward_mean += delta / self.intrinsic_reward_count

        # Update RUNNING variance (M2 = sum of squared deviations)
        delta2 = intrinsic_reward - self.intrinsic_reward_mean
        self._m2 += delta * delta2
        
        if self.intrinsic_reward_count >= 2:
            variance = self._m2 / (self.intrinsic_reward_count - 1)
            self.intrinsic_reward_std = np.sqrt(max(variance, 1e-8))

    
    def compute_rnd_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute RND predictor loss: MSE between predictor and target
        This trains the predictor to recognize visited states -> SO TO REDUCE THE NOVELTY REWARD
        """
        _, predictor_features, target_features = self.compute_intrinsic_reward(states)
        
        # MSE loss for predictor training
        rnd_loss = F.mse_loss(predictor_features, target_features.detach()) 

        return rnd_loss
    
    def step(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, 
             advantages: torch.Tensor, returns: torch.Tensor) -> None:
        """
        PPO update step with RND predictor training
        """
        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, returns),
            batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.steps):
            for batch in dataset:
                batch_states, batch_actions, old_probs, adv, ret = batch
                
                # Forward pass
                action_pred, value_pred = self.forward(batch_states)
                value_pred = value_pred.squeeze(-1)

                # Calculate new action probabilities and entropy
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # PPO losses
                surrogate_loss = self.get_surrogate_loss(old_probs, new_log_probs, adv)
                policy_loss, value_loss = self.get_loss(surrogate_loss, entropy, ret, value_pred)

                # RND predictor loss (trains predictor to match target)
                rnd_loss = self.compute_rnd_loss(batch_states)

                # Total loss
                total_loss = policy_loss + value_loss + self.rnd_loss_coeff * rnd_loss

                # Backpropagate and update
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()