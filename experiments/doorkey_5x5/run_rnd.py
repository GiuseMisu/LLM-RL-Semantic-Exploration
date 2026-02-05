"""
Train RND-PPO (Curiosity-Driven) on DoorKey-5x5 environment
Final experimental comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from experiments import (
    make_minigrid_env,
    evaluate_policy,
    RNDPPO,
    ENV_CONFIGS,
    SHARED_PPO_PARAMS,
    RND_PARAMS,
    EVALUATION_PARAMS,
    move_experiment_files
)


def main():
    # Load configuration
    config = ENV_CONFIGS["doorkey_5x5"]
    
    # Create environment
    env = make_minigrid_env(
        env_id=config["env_id"],
        render_mode="rgb_array",
        use_llm_rewards=False,
        max_steps=config["max_steps"]
    )()
        
    # Initialize RND-PPO with shared and RND-specific parameters
    policy = RNDPPO(
        env=env,
        epochs=config["epochs"],
        model_name="RNDPPO",
        track_stats=True,
        **SHARED_PPO_PARAMS,  # gamma, epsilon
        **RND_PARAMS  # gamma_intrinsic, intrinsic_reward_coeff, rnd_dim
    )
    
    # Set environment-specific parameters
    policy.batch_size = config["batch_size"]
    policy.rollout.iterations = config["rollout_iterations"]
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting RND-PPO training on DoorKey-5x5")
    print(f"{'='*60}\n")
    
    policy.trainer(
        early_stopping_threshold=config["early_stopping_threshold"],
        window_size=config["early_stopping_window"]
    )
    
    policy.load()

    print("\n=========== EVALUATION PHASE ===========\n")
    eval_env = make_minigrid_env(
        env_id=config["env_id"], 
        render_mode="rgb_array", 
        use_llm_rewards=False,
        max_steps=config["max_steps"]
    )()

    policy.eval()
    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(
        eval_env, 
        policy, 
        **EVALUATION_PARAMS
    )

    # Move experiment files to appropriate directories
    move_experiment_files("RNDPPO", "DOORKEY_5x5")
        
if __name__ == "__main__":
    main()


# Episode 1/20: Reward=0.97, Steps=8
# Episode 2/20: Reward=0.96, Steps=10
# Episode 3/20: Reward=0.97, Steps=8
# Episode 4/20: Reward=0.97, Steps=8
# Episode 5/20: Reward=0.96, Steps=10
# Episode 6/20: Reward=0.96, Steps=10
# Episode 7/20: Reward=0.97, Steps=9
# Episode 8/20: Reward=0.96, Steps=12
# Episode 9/20: Reward=0.97, Steps=8
# Episode 10/20: Reward=0.96, Steps=11
# Episode 11/20: Reward=0.97, Steps=9
# Episode 12/20: Reward=0.97, Steps=9
# Episode 13/20: Reward=0.97, Steps=9
# Episode 14/20: Reward=0.97, Steps=9
# Episode 15/20: Reward=0.97, Steps=9
# Episode 16/20: Reward=0.96, Steps=12
# Episode 17/20: Reward=0.97, Steps=9
# Episode 18/20: Reward=0.96, Steps=10
# Episode 19/20: Reward=0.96, Steps=10
# Episode 20/20: Reward=0.97, Steps=9

# ==================================================
# EVALUATION STATISTICS
# ==================================================
# Mean Reward:    0.966 +/- 0.004
# Min Reward:     0.957
# Max Reward:     0.971
# Mean Length:    9.4 +/- 1.2 steps
# Success Rate:   100.0% (20/20)
# ==================================================