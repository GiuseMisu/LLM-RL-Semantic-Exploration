import gymnasium as gym
import numpy as np
from typing import Optional, Callable

class LLMRewardWrapper(gym.Wrapper):
    """
    Wrapper between the environment and PPO -> Intercepts observations and rewards
    it augments environment rewards with LLM-generated rew
    
    1 Receives the original env observation (dict with 'image' and 'mission')
    2 Converts it to text with textualizer func
    3 Queries LLM for a reward
    4 Combines LLM reward with environment reward
    5 Returns the standard image observation for the RL agent
    
    Args:
        env: The base MiniGrid environment (WITHOUT ImgObsWrapper)
        llm_client: RobustCachedLLMClient instance
        textualizer_fn: Function that converts env to text (get_DOORKEY_description)
        llm_weight: How much to weight LLM rewards (0.0 = pure env, 1.0 = pure LLM)
    """
    
    def __init__(
        self,
        env: gym.Env,
        llm_client: Optional[object] = None,
        textualizer_fn: Optional[Callable] = None,
        llm_weight: float = 1.0,
        verbose: bool = False
    ):
        super().__init__(env)
        
        # CRITICAL: Define observation_space to match the image we return (7x7x3 for MiniGrid)
        # The base MiniGrid env has a Dict space {'image': Box(7,7,3), 'mission': ...}
        # but we extract and return only 'image'
        #LLMRewardWrapper returns only the image array to the agent (it extracts obs['image'] in step() / reset())
        # so check it and sets the wrapper’s observation space to the inner Box(7,7,3)
        if hasattr(env.observation_space, 'spaces') and 'image' in env.observation_space.spaces:
            self.observation_space = env.observation_space.spaces['image']
        else:
            # Fallback if the space is already flat or different structure
            # for cases where the wrapped env already returns a flat Box 
            self.observation_space = env.observation_space
        

        self.llm_client = llm_client  # pure RL mode (llm_client=None) passes only env rewards
        self.textualizer_fn = textualizer_fn
        self.llm_weight = llm_weight
        self.verbose = verbose
               
        # Statistics tracking => might be usefull for different LLM strategies
        self.episode_env_reward = 0.0
        self.episode_llm_reward = 0.0
        self.episode_final_reward = 0.0
        self.episode_step_count = 0

        # Global statistics across all episodes/epochs
        self.episode_history = []  # List of dicts with per-episode stats
        self.total_episodes = 0
        
        # Flag to check if LLM is enabled
        self.llm_enabled = (llm_client is not None and textualizer_fn is not None)
        
        if not self.llm_enabled:
            print("[LLM Wrapper] Running in PURE RL mode (no LLM)")
        else:
            print(f"[LLM Wrapper] LLM-guided mode: LLMreward_weight={llm_weight}")
    
    def reset(self, **kwargs):
        #Reset the environment and stats

        # Save current episode statistics if it's not the first reset
        if self.episode_step_count > 0:
            self._save_episode_stats()
        
        # Reset current episode counters
        self.episode_env_reward = 0.0
        self.episode_llm_reward = 0.0
        self.episode_final_reward = 0.0
        self.episode_step_count = 0
        
        obs, info = self.env.reset(**kwargs)
        
        # Return just the image part (for compatibility with ImgObsWrapper)
        return obs['image'], info
    
    def step(self, action):
        # Take a step and augment the reward with LLM feedback
        
        # Execute action in environment
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Extract image for RL agent
        image_obs = obs['image']
        
        # Get LLM reward if enabled
        llm_reward = 0.0
        if self.llm_enabled:
            # Convert current state to text
            text_obs = self.textualizer_fn(self.env)
            
            # Query LLM for reward
            llm_reward = self.llm_client.robust_get_reward(
                text_obs, 
                verbose=self.verbose,
                generate_explanation=False # reasoning activation
            )
        
        #print to show when env returned reward != 0
        if env_reward != 0:
            print(f"===> [SOLVED] Step: {self.episode_step_count} / Env reward: {env_reward} / Final Reward: {100.0 + (env_reward*100)}")
            # IF ENV RETURNED A REWARD => The agent solved the environment goal
            # REWIRE THE REWARD HERE: Make it huge so the agent knows this is the ultimate goal
            final_reward = 100.0 + (env_reward*100)  #aggiungi original reward che premia la velocita oltre a il completamento
        else:
            # LLM gives zero use the llm reward only
            final_reward = self.llm_weight * llm_reward
            
        # Track statistics
        self.episode_env_reward += env_reward
        self.episode_llm_reward += llm_reward
        self.episode_final_reward += final_reward
        self.episode_step_count += 1
        
        # Add reward info to info dict
        info['env_reward'] = env_reward
        info['llm_reward'] = llm_reward
        info['final_reward'] = final_reward

        #print(f"[LLMRwrdWrp] step:{self.episode_step_count:3d} |Env: {int(env_reward):1d} / LLM: {llm_reward:5.2f} / Final: {final_reward:5.2f}")

        return image_obs, final_reward, terminated, truncated, info
    
    def _save_episode_stats(self):
        # method to save current episode stats
        episode_stats = {
            'episode_num': self.total_episodes,
            'env_reward': self.episode_env_reward,
            'llm_reward': self.episode_llm_reward,
            'final_reward': self.episode_final_reward,
            'steps': self.episode_step_count
        }
        self.episode_history.append(episode_stats)
        self.total_episodes += 1
    def finalize_episode(self):
        # at the end of training to capture the final episode
        if self.episode_step_count > 0:
            self._save_episode_stats()
            # Reset counters to prevent double-counting
            self.episode_step_count = 0
            
    def get_statistics(self):
        """Return comprehensive reward statistics across all episodes."""
        if len(self.episode_history) == 0:
            return {
                'total_episodes': 0,
                'avg_env_reward': 0.0,
                'avg_llm_reward': 0.0,
                'avg_final_reward': 0.0,
                'avg_steps': 0.0,
                'episode_history': []
            }
        
        total_env = sum(ep['env_reward'] for ep in self.episode_history)
        total_llm = sum(ep['llm_reward'] for ep in self.episode_history)
        total_final = sum(ep['final_reward'] for ep in self.episode_history)
        total_steps = sum(ep['steps'] for ep in self.episode_history)
        n_episodes = len(self.episode_history)
        
        return {
            'total_episodes': n_episodes,
            'total_env_reward': total_env,
            'total_llm_reward': total_llm,
            'total_final_reward': total_final,
            'total_steps': total_steps,
            'avg_env_reward_per_episode': total_env / n_episodes,
            'avg_llm_reward_per_episode': total_llm / n_episodes,
            'avg_final_reward_per_episode': total_final / n_episodes,
            'avg_steps_per_episode': total_steps / n_episodes,
            'episode_history': self.episode_history  # Full per-episode breakdown
        }
    
    def print_statistics_summary(self):
        """Print a nicely formatted statistics summary."""
        stats = self.get_statistics()
        
        if stats['total_episodes'] == 0:
            print("No episodes completed yet.")
            return
        
        print(f"\n{'='*60}")
        print("TRAINING STATISTICS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Episodes:           {stats['total_episodes']}")
        print(f"Total Steps:              {stats['total_steps']}")
        print(f"\nCumulative Rewards:")
        print(f"  Environment Total:      {stats['total_env_reward']:.2f}")
        print(f"  LLM Total:              {stats['total_llm_reward']:.2f}")
        print(f"  Final Total:            {stats['total_final_reward']:.2f}")
        print(f"\nAverage Per Episode:")
        print(f"  Avg Env Reward:         {stats['avg_env_reward_per_episode']:.4f}")
        print(f"  Avg LLM Reward:         {stats['avg_llm_reward_per_episode']:.4f}")
        print(f"  Avg Final Reward:       {stats['avg_final_reward_per_episode']:.4f}")
        print(f"  Avg Steps:              {stats['avg_steps_per_episode']:.2f}")
        print(f"{'='*60}")