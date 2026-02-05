import os
import warnings
# ---  SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../"))

from src.methods.llm_guided.llm_clients.Qwen3Code480b import Qwen3CoderClient
from src.methods.llm_guided.EurekaApproach.eureka_search import EurekaSearch
from src.common.env_setup import make_minigrid_env
from src.common.policy_evaluation import evaluate_policy


if __name__ == "__main__":

    try:
        # Initialize the client
        llm = Qwen3CoderClient(            
            temperature=0.1   # really low temp for qwen-coder
            )
        print(f"Model: {llm.model_name}")
    except Exception as e:
        print(f"Setup Failed: {e}")
        sys.exit(1)

    # Run Search
    baseline_rl = 'PPO' # 'PPO' or 'RecurrentPPO' 
    env_id = "MiniGrid-DoorKey-8x8-v0"

    eureka = EurekaSearch(
        env_id=env_id,
        llm_model=llm,

        reflection_iterations=3, # numero di volte che provi a miglioreare la reward heuristic returned by llm
        
        training_epochs=50,  # numb of epochs to train the model with each candidate reward function
        train_max_steps= 650, # max steps per training episode
        num_eval_episodes=50, # numb of episodes to evaluate each candidate reward function
        pure_rl_baseline=baseline_rl, 

        batch_size=4096,           # 8x8 config
        rollout_iterations=16384   # 8x8 config 

    )

    best_code = eureka.find_best_RwdFunc()

    final_policy = eureka.train_final_model(
        final_train_epochs=100,
        final_train_max_steps=650,
        reward_code_str=best_code
    )

    # load a trained version of the environment
    final_policy.load()

    print("\n=========== EVALUATION PHASE ===========\n")
    eval_env = make_minigrid_env(env_id=env_id, 
                                 render_mode="rgb_array", 
                                 max_steps=650
                                 )()

    final_policy.eval()
    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(eval_env, 
                            final_policy, 
                            n_episodes=50
                            )

