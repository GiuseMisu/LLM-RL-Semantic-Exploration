import os
import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ENV_ID = "MiniGrid-DoorKey-5x5-v0"

def make_minigrid_env(env_id=ENV_ID, render_mode="rgb_array", log_dir=None, idx=0):
    """
    Create a single MiniGrid environment instance.
    
    Args:
        env_id (str): The environment ID.
        render_mode (str): 'rgb_array' for training, 'human' for visualization.
        log_dir (str): Directory to save Monitor logs.
        idx (int): Unique index for this environment (useful for multi-process logging).
    """
    def _init():
        
        env = gym.make(env_id, render_mode=render_mode)
        
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

def get_vectorized_env(env_id=ENV_ID, n_envs=4, log_dir="./data/logs/"):
    """
    Creates a Vectorized Environment (multiple independent copies running in parallel).
    This increases the diversity of data in each batch and stabilizes PPO training.
    """
    # Generate a list of environment factory functions
    env_fns = [make_minigrid_env(env_id, log_dir=log_dir, idx=i) for i in range(n_envs)]
    
    # DummyVecEnv runs them sequentially in the same process (easier for debugging).
    # SubprocVecEnv runs them in separate processes (faster for heavy environments).
    # For MiniGrid 5x5, DummyVecEnv is actually often faster due to low overhead.
    return DummyVecEnv(env_fns)

if __name__ == "__main__":
    print(f"--- Inspecting {ENV_ID} ---")
    
    # Create one instance
    test_env = make_minigrid_env(render_mode="human")()
    obs, info = test_env.reset()
    
    print(f"Observation Shape: {obs.shape}")
    print(f"Action Space:      {test_env.action_space}")
    print("""Action space: Discrete(7)
            0 = Turn Left       # Rotate 90° counterclockwise
            1 = Turn Right      # Rotate 90° clockwise
            2 = Move Forward    # Move one cell in the direction you're facing
            3 = Pick Up         # Pick up an object (like the key)
            4 = Drop            # Drop the object you're carrying
            5 = Toggle          # Interact with objects (open/close doors)
            6 = Done            # Signal task completion (rarely used)""")
    
    print("\n--- Understanding the Input (7x7x3) ---")
    print("The observation is a 7x7 grid centered around the agent, with 3 channels of info:")
    print("Channel 0 (Object IDs): What object is in each cell (2 = Floor / 4 = Keys / ...)")
    print("Channel 1 (Colors):     The color of each object (0=Red, 1=Green, 2=Blue, ...)")
    print("Channel 2 (State):      The state of objects 0=Open door, 1=Closed door, 2=Locked door")
    
    print("\n--- Interaction Test ---")
    
    for step in range(5):
        action = test_env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")
        if terminated or truncated:
            obs, info = test_env.reset()
                
    test_env.close()