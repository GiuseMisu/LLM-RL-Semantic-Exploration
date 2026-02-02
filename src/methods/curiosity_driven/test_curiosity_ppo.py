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
from src.methods.curiosity_driven.rnd_ppo_config import RNDPPO

from src.common.policy_evaluation import evaluate_policy

def main():

    #env_id = "MiniGrid-DoorKey-8x8-v0"
    env_id = "MiniGrid-DoorKey-5x5-v0"

    #==============seed ti garantisce sempre steso env config ============
    # NON USARE SEED DURING TRAINGING MA SOLO IN VAL SE VUOI VEDERE UNO SPECIFICO SCENARIO
    #======================================================================
    # in 5x5 seed = 0 è quello facile con porta tutto sopra / seed = 1 é quello difficile con porta in mezzo   

    # Create environment using your env_setup.py
    env = make_minigrid_env(env_id=env_id, 
                            render_mode="rgb_array", 
                            max_steps=250 # 250 for 5x5, but 650 for 8x8
                            )()
    
    print("\n=========== TRAINING PHASE===========\n")
    # Define the Policy
    policy = RNDPPO(env = env, 
                          # done automatically inside the code output_dim= 4, 
                          epochs = 5, #100
                          gamma = 0.99, 
                          gamma_intrinsic = 0.99, # late (high value) or early (lower value) exploration
                          epsilon = 0.2,
                          model_name="RNDPPO", 

                          intrinsic_reward_coeff=0.01, # if high helps to explore more => more weight to intrinsic reward then env reward
                                                      # if low  helps to exploit more => more weight to env reward
                          
                          track_stats=False
                          )

    policy.batch_size = 4096

    # rollout buffer size to match or exceed the batch size
    policy.rollout.iterations = 8192 #16384  # for 8x8 / 4096 # for 5x5 
    
    # Train the environment
    policy.trainer(
        early_stopping_threshold=None
        # early_stopping_threshold = 0.90,  # average reward threshold for early stopping 
        # window_size = 15  # Number of epochs to average over
        )    
    
    # load a trained version of the environment
    policy.load()

    # print("\n\nEvaluating the trained policy")
    # eval_env = make_minigrid_env(env_id=env_id, 
    #                              render_mode="rgb_array", 
    #                              max_steps=50
    #                              )()

    # policy.eval()
    
    # # Evaluate over multiple episodes for statistics
    # stats = evaluate_policy(eval_env, 
    #                         policy, 
    #                         n_episodes=10 # evaluation over 10 episodes
    #                         )
    

if __name__ == "__main__":
    main()