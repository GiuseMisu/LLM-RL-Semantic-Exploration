import os
import warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")

import sys
sys.path.append('../../../') 
sys.path.append('./')

from src.common.env_setup import make_minigrid_env
from ppo.ppo_config import PPO

from src.common.policy_evaluation import evaluate_policy

def main():

    env_id = "MiniGrid-DoorKey-8x8-v0"
    #env_id = "MiniGrid-DoorKey-8x8-v0"

    #==============seed ti garantisce sempre steso env config ============
    # NON USARE SEED DURING TRAINGING MA SOLO IN VAL SE VUOI VEDERE UNO SPECIFICO SCENARIO
    #======================================================================
    # in 5x5 seed = 0 è quello facile con porta tutto sopra / seed = 1 é quello difficile con porta in mezzo   

    env = make_minigrid_env(env_id=env_id, 
                            render_mode="rgb_array", 
                            max_steps=650 # 250 for 5x5, but 650 for 8x8
                            )()
    
    print("\n=========== TRAINING PHASE===========\n")
    # Define the Policy
    policy = PPO(
                env = env, 
                # done automatically inside the code output_dim= 4, 
                epochs = 30, 
                gamma = 0.99, 
                epsilon = 0.2,
                model_name=f"PPO_{env_id.split('-')[1]}_llm_guided"#"PPO"
                )

    policy.batch_size = 4096  # for 8x8 /  2048 # for 5x5

    # rollout buffer size to match or exceed the batch size
    # rollout buffer size to match or exceed the batch size
    policy.rollout.iterations = 16384  # for 8x8 / 4096 # for 5x5 

    # Train the environment
    # policy.trainer(
    #     early_stopping_threshold = 0.90,  # average reward threshold for early stopping 
    #     window_size = 10  # Number of epochs to average over
    #     )    
    
    # load a trained version of the environment
    policy.load()

    print("\n=========== EVALUATION PHASE ===========\n")
    eval_env = make_minigrid_env(env_id=env_id, 
                                 render_mode="rgb_array", 
                                 max_steps=50
                                 )()

    policy.eval()
    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(eval_env, 
                            policy, 
                            n_episodes=10
                            )

if __name__ == "__main__":
    main()