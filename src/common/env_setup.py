from typing import Optional
import gymnasium as gym

#PURE RL APPROACH NEEDS ImgObsWrapper
from minigrid.wrappers import ImgObsWrapper

# LLM SCALAR-REWARD WRAPPER 
from src.methods.llm_guided.ScalarApproach.llm_reward_wrapper import LLMRewardWrapper

# LLM EUREKA APPROACH WRAPPER
from src.methods.llm_guided.EurekaApproach.eureka_wrapper import EurekaRewardWrapper     

def make_minigrid_env(
        env_id="MiniGrid-DoorKey-5x5-v0", 
        seed: Optional[int] = None,
        render_mode="rgb_array", 
        max_steps: Optional[int] = None,

        #[OPTIONAL PARAMETERS] needed for LLM-SCALAR REWARD 
        use_llm_rewards=False,
        llm_client=None,
        textualizer_fn=None,
        llm_weight=1.0,
        verbose=False,

        # PARAM FOR LLM-EUREKA APPROACH
        eureka_reward_code: Optional[str] = None
        ):
    
    """
    Create a single MiniGrid environment instance.
    Args:
        env_id (str): The environment ID: e.g., "MiniGrid-DoorKey-5x5-v0" or "MiniGrid-Empty-5x5-v0"
        render_mode (str): 'rgb_array' for training, 'human' for visualization
    LLM integration Args:
        use_llm_rewards (bool): Whether to use LLM-augmented rewards
        llm_client: Instance of RobustCachedLLMClient (or None for pure RL)
        textualizer_fn: Function to convert env to text
        llm_weight (float): Weight for LLM rewards (0.0-1.0)
        verbose (bool): Print LLM reasoning
    """
    def _init():
        if max_steps is not None:
            env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
        else:
            # If max_steps is None, MiniGrid default limit
            env = gym.make(env_id, render_mode=render_mode)

        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)

        # CRITICAL: If using LLM, wrap BEFORE ImgObsWrapper
        if use_llm_rewards and llm_client is not None:
            print(f"[Env Setup] Wrapping environment -> LLMRewardWrapper for: {env_id}")
            env = LLMRewardWrapper(
                env,
                llm_client=llm_client,
                textualizer_fn=textualizer_fn,
                llm_weight=llm_weight,
                verbose=verbose
            )

        #EUREKA APPROACH
        # wrap with Eureka first (to calculate reward), then ImgObs (for PPO compatibility)
        elif eureka_reward_code is not None:
            print(f"[Env Setup] Wrapping environment -> EurekaRewardWrapper for: {env_id}")
            env = EurekaRewardWrapper(env, eureka_reward_code)
            env = ImgObsWrapper(env)

        else:  
            print(f"[Env Setup] Using PURE RL (no LLM) for: {env_id}")
            # WRAPPER
            # MiniGrid returns a dict: {'image': ..., 'mission': ...}
            # RL alg cannot handle dictionaries, they need pixel for CNN 
            # ImgObsWrapper extracts ONLY the 'image' key (7x7x3 grid).
            #==============================
            # IMP:
            # WRAPPER NEEDED FOR PURE RL ALG
            # WITH ENV.UNWRAPPED YOU GET THE DICT OBS => NEED FOR LLM-GUIDED METHODS
            #==============================
            env = ImgObsWrapper(env)            
        
        return env

    return _init


if __name__ == "__main__":
    
    env_id_door_key = "MiniGrid-DoorKey-5x5-v0"
    env_id_empty = "MiniGrid-Empty-5x5-v0" 
    print(f"--- Inspecting: {env_id_empty} ---")
    
    # Create one env instance
    test_env = make_minigrid_env(env_id=env_id_empty, render_mode="human")()
    obs, info = test_env.reset()
    
    print(f"Observation Shape: {obs.shape}")
    print(f"Action Space:      {test_env.action_space}")
    
    for step in range(5):
        action = test_env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")
        if terminated or truncated:
            obs, info = test_env.reset()
                
    test_env.close()