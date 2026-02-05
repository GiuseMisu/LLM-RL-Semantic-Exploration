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
from ppo.ppo_config import PPO, RecurrentPPO

from src.common.policy_evaluation import evaluate_policy

def main():

    env_id = "MiniGrid-Empty-5x5-v0"
    #env_id = "MiniGrid-DoorKey-8x8-v0"
    #env_id = "MiniGrid-Empty-16x16-v0" 
     
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
    policy = RecurrentPPO(env = env, 
                          # done automatically inside the code output_dim= 4, 
                          epochs = 25, 
                          gamma = 0.99, 
                          epsilon = 0.2,
                          encode_dim=128,  # CNN output
                          hidden_dim=128,    # LSTM hidden size
                          sequence_length=32, #32 per 8x8 env,    # TBPTT length
                          recurrence = "lstm",
                          model_name="RecurrentPPO",

                          track_stats=False
                          )

    policy.batch_size = 2048 # 4096 for 8x8 /  2048 # for 5x5

    # rollout buffer size to match or exceed the batch size
    policy.rollout.iterations = 4096  # 16384 for 8x8 / 4096 # for 5x5 

    # Train the environment
    policy.trainer(
        early_stopping_threshold = 0.95,  # average ENV_RWD threshold for early stopping 
        window_size = 10  # Number of epochs to average over
        )    
    
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
                            n_episodes=10 # evaluation over 10 episodes
                            )
    

if __name__ == "__main__":
    main()