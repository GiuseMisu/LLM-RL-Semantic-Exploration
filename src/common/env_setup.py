from typing import Optional, Callable
import os
import gymnasium as gym
from gymnasium.envs.registration import register

import warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")

#PURE RL APPROACH NEEDS ImgObsWrapper
from minigrid.wrappers import ImgObsWrapper
# LLM REWARD WRAPPER FOR LLM-GUIDED METHODS
from src.methods.llm_guided.llm_reward_wrapper import LLMRewardWrapper
            
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv



def make_minigrid_env(
        env_id="MiniGrid-DoorKey-5x5-v0", 
        render_mode="rgb_array", 
        max_steps: Optional[int] = None,
        log_dir=None, 
        idx=0,
        # NEW ==OPTIONAL== PARAMETERS for LLM integration
        use_llm_rewards=False,
        llm_client=None,
        textualizer_fn=None,
        llm_weight=1.0,
        verbose=False        
        ):
    
    """
    Create a single MiniGrid environment instance.
    Args:
        env_id (str): The environment ID: e.g., "MiniGrid-DoorKey-5x5-v0" or "MiniGrid-Empty-5x5-v0"
        render_mode (str): 'rgb_array' for training, 'human' for visualization
        log_dir (str): Directory to save Monitor logs
        idx (int): Unique index for this environment (useful for multi-process logging)
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
            # WRAPPER: Monitor
            # This wrapper records: Episode Reward, Episode Length and Time
            # It writes to a file (monitor.csv) -> use later to generate graphs
            # if log_dir is not None:
            #     os.makedirs(log_dir, exist_ok=True)
            #     log_path = os.path.join(log_dir, str(idx))
            #     env = Monitor(env, filename=log_path)            
        return env

    return _init


if __name__ == "__main__":
    
    env_id_door_key = "MiniGrid-DoorKey-5x5-v0"
    env_id_empty = "MiniGrid-Empty-5x5-v0" 
    print(f"--- Inspecting: {env_id_empty} ---")
    
    # Create one instance
    test_env = make_minigrid_env(env_id=env_id_empty, render_mode="human")()
    obs, info = test_env.reset()
    
    print(f"Observation Shape: {obs.shape}")
    print(f"Action Space:      {test_env.action_space}")
    
    # =======> for the minigrid doorkey env
    # print("""Action space: Discrete(7)
    #         0 = Turn Left       # Rotate 90° counterclockwise
    #         1 = Turn Right      # Rotate 90° clockwise
    #         2 = Move Forward    # Move one cell in the direction you're facing
    #         3 = Pick Up         # Pick up an object (like the key)
    #         4 = Drop            # Drop the object you're carrying
    #         5 = Toggle          # Interact with objects (open/close doors)
    #         6 = Done            # Signal task completion (rarely used)""")
    # print("\n--- Understanding the Input (7x7x3) ---")
    # print("The observation is a 7x7 grid centered around the agent, with 3 channels of info:")
    # print("Channel 0 (Object IDs): What object is in each cell (2 = Floor / 4 = Keys / ...)")
    # print("Channel 1 (Colors):     The color of each object (0=Red, 1=Green, 2=Blue, ...)")
    # print("Channel 2 (State):      The state of objects 0=Open door, 1=Closed door, 2=Locked door")
    
    for step in range(5):
        action = test_env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")
        if terminated or truncated:
            obs, info = test_env.reset()
                
    test_env.close()