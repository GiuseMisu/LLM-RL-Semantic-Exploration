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
import re

from src.methods.pure_rl.ppo.ppo_config import PPO
from src.methods.pure_rl.utils.network import BaseNet, MiniGridCNN
from src.methods.curiosity_driven.rnd_rollout import RNDRollout
from src.common.metrics import MetricsTracker

"""
PPO but Random Network Distillation.
This implementation uses SEPARATE ADVANTAGE STREAMS (canonical RND):
- Extrinsic returns computed with gamma (long-term environment rewards)
- Intrinsic returns computed with gamma_intrinsic (novelty-based exploration bonus)
- Combined advantage = A^E + A^I for policy optimization

V^E Learns ONLY about task rewards
V^I Learns ONLY about intrinsic  rewards (curiosisty) 
IMP => intrinsic does rapidly change so need a separate value that does not dampen the extrinsic one

cite: https://www.emergentmind.com/topics/random-network-distillation-rnd
"""
class RNDPPO(PPO):
    def __init__(
            self, env : gym.Env, 
            epsilon : float = 0.2,
            gamma : float = 0.99,            # Discount for extrinsic rewards            
            gamma_intrinsic : float = 0.99,  # Separate discount for intrinsic rewards
            output_dim : int | None = None,  # Auto-detect from env
            encode_dim : int = 128,
            rnd_dim : int = 128,
            epochs : int = 100,
            model_name : str = "RNDPPO",

            intrinsic_reward_coeff : float = 0.005,  #Scale intrinsic rewards

            track_stats: bool = True
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

        #================Extract Env Type and Model Name For File Name================
        self.env_type = "unknown"
        if hasattr(env.unwrapped, "spec") and env.unwrapped.spec is not None:
            env_id = env.unwrapped.spec.id
            size_match = re.search(r'(\d+)x(\d+)', env_id)
            if size_match:
                env_dimension = size_match.group(1) + 'x' + size_match.group(2)
                if "empty" in env_id.lower() or "minigrid-empty" in env_id.lower() :
                    self.env_type = "EMPTY_" + env_dimension
                elif "door" in env_id.lower()  and "key" in env_id.lower()  or "doorkey" in env_id.lower() :
                    self.env_type = "DOORKEY_" + env_dimension
                else:
                    print(f"[WARNING] Unrecognized MiniGrid env type in env_id: {env_id}, defaulting to OTHER")
                    self.env_type = "OTHER_" + env_dimension
            else:
                print(f"[WARNING] Could not parse env dimensions from env_id: {env_id}")
        
        self.model_name = model_name
        #==============================================================================

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
        self.intrinsic_reward_coeff = intrinsic_reward_coeff  # Scale intrinsic rewards (default 0.005)
        self.rnd_loss_coeff = 0.5   # Weight for RND predictor loss
        
        # Separate discount factor for intrinsic rewards
        self.gamma_intrinsic = gamma_intrinsic
        
        # Running statistics for intrinsic reward normalization (Welford's algorithm)
        # Different states have wildly different intrinsic magnitudes - normalization stabilizes training
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0
        self.intrinsic_reward_count = 0
        self._m2 = 0.0  # For sum of squared deviations

        self.optimizer = Adam([p for n, p in self.named_parameters() if "rnd_target" not in n], lr=self.lr)

        # Use RND-specific rollout with dual advantage streams
        self.rollout = RNDRollout(self.env, self)

        self.track_stats = track_stats

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
    
    def compute_intrinsic_reward_normalized(self, next_state) -> float:
        """
        Compute normalized intrinsic reward for a single state
        Used by RNDRollout for separate advantage stream computation
        Returns-> Normalized and scaled intrinsic reward
        """
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            intrinsic_rewards, _, _ = self.compute_intrinsic_reward(next_state_tensor)
            raw_intrinsic = intrinsic_rewards.mean().item()
        
        # Update running statistics for normalization
        self._update_intrinsic_stats(raw_intrinsic)
        
        # Normalize to prevent early explosion
        effective_std = max(self.intrinsic_reward_std, 0.1)
        normalized_intrinsic = raw_intrinsic / effective_std
        
        # Clip to prevent extreme outliers
        normalized_intrinsic = np.clip(normalized_intrinsic, 0.0, 10.0)
        
        # Apply coefficient
        scaled_intrinsic = self.intrinsic_reward_coeff * normalized_intrinsic
        
        return scaled_intrinsic
            
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass for action selection and value estimation.
        Returns combined value (extrinsic + intrinsic) for backward compatibility.
        
        NOTE: Intrinsic reward is computed in augment_reward() on the NEXT state, not here. 
        This follows canonical RND where novelty = how novel is the state you REACHED, not the state you LEFT.
        """
        state = state.to(self.device)
        encoded_state = self.encoder(state)

        action_logits = self.actor(encoded_state)
        extrinsic_value = self.critic(encoded_state)
        intrinsic_value = self.intrinsic_critic(encoded_state)
        
        # Combined value for backward compatibility
        value = extrinsic_value + intrinsic_value

        return action_logits, value
    
    def forward_dual_value(self, state: torch.Tensor) -> tuple:
        """
        Forward pass returning SEPARATE extrinsic and intrinsic values.
        Used by RNDRollout for dual advantage stream computation.
        
        Returns:
            action_logits: Action logits for policy
            extrinsic_value: Value prediction for extrinsic rewards
            intrinsic_value: Value prediction for intrinsic rewards
        """
        state = state.to(self.device)
        encoded_state = self.encoder(state)

        action_logits = self.actor(encoded_state)
        extrinsic_value = self.critic(encoded_state)
        intrinsic_value = self.intrinsic_critic(encoded_state)

        return action_logits, extrinsic_value, intrinsic_value
        
    def augment_reward(self, reward: float, next_state=None) -> float:
        """
        Augment extrinsic reward with normalized intrinsic reward.
        
        Canonical RND: intrinsic reward is computed on the NEXT state (s_{t+1}),
        rewarding the agent for reaching novel states, not for leaving them.
        
        Args:
            reward: Extrinsic reward from environment
            next_state: The state reached after taking the action (numpy array)
        """
        if next_state is None:
            # Fallback: no intrinsic reward if next_state not provided
            return reward
        
        # Convert next_state to tensor and compute intrinsic reward
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            intrinsic_rewards, _, _ = self.compute_intrinsic_reward(next_state_tensor)
            raw_intrinsic = intrinsic_rewards.mean().item()
        
        # Update running statistics for normalization
        self._update_intrinsic_stats(raw_intrinsic)
        
        # Normalize to prevent early explosion
        # 0.1 minimum to prevent division by tiny numbers
        effective_std = max(self.intrinsic_reward_std, 0.1)
        normalized_intrinsic = raw_intrinsic / effective_std
        
        # Clip to prevent extreme outliers (keep rewards reasonable)
        normalized_intrinsic = np.clip(normalized_intrinsic, 0.0, 10.0)
        
        # Apply coefficient
        scaled_intrinsic = self.intrinsic_reward_coeff * normalized_intrinsic
        
        augmented_reward = reward + scaled_intrinsic

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
    
    def step_dual_stream(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, 
                          advantages: torch.Tensor, extrinsic_returns: torch.Tensor, 
                          intrinsic_returns: torch.Tensor) -> None:
        """
        PPO update step with SEPARATE value losses for extrinsic and intrinsic critics.
        
        This is the canonical RND approach:
        - Policy loss uses combined advantages (A^E + A^I)
        - Extrinsic critic is trained on extrinsic returns (gamma)
        - Intrinsic critic is trained on intrinsic returns (gamma_intrinsic)
        - RND predictor is trained to reduce novelty for visited states
        """
        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, 
                          extrinsic_returns, intrinsic_returns),
            batch_size=self.batch_size, shuffle=True
        )

        # [LOGGING]
        total_p_loss, total_v_loss, total_rnd_loss, total_ent, count = 0, 0, 0, 0, 0

        for _ in range(self.steps):
            for batch in dataset:
                batch_states, batch_actions, old_probs, adv, ext_ret, int_ret = batch
                
                # Forward pass with separate values
                action_pred, ext_value, int_value = self.forward_dual_value(batch_states)
                ext_value = ext_value.squeeze(-1)
                int_value = int_value.squeeze(-1)

                # Calculate new action probabilities and entropy
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Policy loss (uses combined advantages)
                surrogate_loss = self.get_surrogate_loss(old_probs, new_log_probs, adv)
                entropy_bonus = self.entropy_coeff * entropy
                policy_loss = -(surrogate_loss + entropy_bonus).mean()
                
                # SEPARATE value losses for each critic
                extrinsic_value_loss = F.smooth_l1_loss(ext_ret, ext_value).mean()
                intrinsic_value_loss = F.smooth_l1_loss(int_ret, int_value).mean()
                value_loss = extrinsic_value_loss + intrinsic_value_loss

                # RND predictor loss (trains predictor to match target)
                rnd_loss = self.compute_rnd_loss(batch_states)

                # Total loss
                total_loss = policy_loss + value_loss + self.rnd_loss_coeff * rnd_loss

                # Backpropagate and update
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # [LOGGING]
                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_rnd_loss += rnd_loss.item()
                total_ent += entropy.mean().item()
                count += 1
                
        return total_p_loss/count, total_v_loss/count, total_rnd_loss/count, total_ent/count #added for logging stats


    def trainer(self, early_stopping_threshold: float = 0.95, window_size: int = 10):
        """
        Training loop for RND-PPO with SEPARATE advantage streams.
        
        Uses dual stream rollout to compute:
        - Extrinsic returns with gamma
        - Intrinsic returns with gamma_intrinsic
        - Combined advantages for policy optimization
        """

        if self.track_stats:    
            log_name = self.model_name + "_" + self.env_type
            tracker = MetricsTracker(run_name=log_name, log_dir="logs")

        max_rew = -float("inf")
        consecutive_epochs_mean_reward = []

        for e in range(self.epochs):
            # Use dual stream forward pass
            # Returns: (avg_extrinsic, avg_intrinsic, avg_total)
            (avg_ext, avg_int, avg_total), states, actions, log_probs, advantages, \
                extrinsic_returns, intrinsic_returns, eps_sizes = self.rollout.forward_pass_dual_stream()
            
            #[LOGGING]
            avg_ep_len = np.mean(eps_sizes) if eps_sizes else 0

            #three reward components:
            # - Env Reward: The actual reward from the environment (what matters for task success)
            # - Curiosity Reward: The intrinsic reward from RND (exploration bonus)
            # - Total Reward: Env + Curiosity (what the agent optimizes)
            if avg_ext > max_rew:
                print(f"Epoch {e+1}/{self.epochs} | Env: {avg_ext:.4f} | Curiosity: {avg_int:.4f} | Total: {avg_total:.4f} ==> New best ENV REWARD, saving")
                max_rew = avg_ext
                self.save(
                    filename=f"{self.model_name}_{self.env_type}_best_env_reward"
                ) 
            else:
                print(f"Epoch {e+1}/{self.epochs} | Env: {avg_ext:.4f} | Curiosity: {avg_int:.4f} | Total: {avg_total:.4f}")

            #activate the early stopping mechanism if early_stopping_threshold is set
            if early_stopping_threshold is not None:
                consecutive_epochs_mean_reward.append(avg_ext)
                if len(consecutive_epochs_mean_reward) > window_size:
                    consecutive_epochs_mean_reward.pop(0)
                
                if len(consecutive_epochs_mean_reward) == window_size:
                    avg_recent = np.mean(consecutive_epochs_mean_reward)
                    if avg_recent >= early_stopping_threshold:
                        print(f"\nEARLY STOPPING TRIGGERED at epoch {e+1}")
                        print(f"Average reward over last {window_size} epochs: {avg_recent:.5f}")
                        print(f"Threshold: {early_stopping_threshold}\n")
                        break

            # Use dual stream step with separate returns
            p_loss, v_loss, rnd_loss, ent = self.step_dual_stream(
                states.to(self.device), 
                actions.to(self.device), 
                log_probs.to(self.device), 
                advantages.to(self.device), 
                extrinsic_returns.to(self.device),
                intrinsic_returns.to(self.device)
            )

            if self.track_stats:
                tracker.log(e, {
                    "Extrinsic_Reward": avg_ext,
                    "Intrinsic_Reward": avg_int,
                    "Total_Reward": avg_total,
                    "Episode_Length": avg_ep_len,
                    "Policy_Loss": p_loss,
                    "Value_Loss": v_loss,
                    "RND_Loss": rnd_loss,
                    "Entropy": ent
                })
        if self.track_stats:
            tracker.save()
            tracker.plot()