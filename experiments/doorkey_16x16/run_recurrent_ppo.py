"""
Train RecurrentPPO (with LSTM) on doorkey_16x16 environment
Final experimental comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from experiments import (
    make_minigrid_env,
    evaluate_policy,
    RecurrentPPO,
    ENV_CONFIGS,
    SHARED_PPO_PARAMS,
    RECURRENT_PARAMS,
    EVALUATION_PARAMS,
    move_experiment_files
)


def main():
    
    config = ENV_CONFIGS["doorkey_16x16"]
    
    # Create environment
    env = make_minigrid_env(
        env_id=config["env_id"],
        render_mode="rgb_array",
        use_llm_rewards=False,
        max_steps=config["max_steps"]
    )()
        
    # Initialize RecurrentPPO with shared and recurrent-specific parameters
    policy = RecurrentPPO(
        env=env,
        epochs=config["epochs"],
        model_name="RecurrentPPO",
        save_pkl_model=True,
        track_stats=True,
        **SHARED_PPO_PARAMS,  # gamma, epsilon
        **RECURRENT_PARAMS  # hidden_dim, sequence_length, encode_dim
    )
    
    # Set environment-specific parameters
    policy.batch_size = config["batch_size"]
    policy.rollout.iterations = config["rollout_iterations"]
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting RecurrentPPO training on DOORKEY_16x16")
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

    #policy.eval()
    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(
        eval_env, 
        policy, 
        **EVALUATION_PARAMS
    )
    
    # Move experiment files to appropriate directories
    move_experiment_files("RecurrentPPO_lstm", "DOORKEY_16x16")

        
if __name__ == "__main__":
    main()


# ==================================================
# EVALUATION STATISTICS
# ==================================================
# Mean Reward:    0.000 +/- 0.000
# Min Reward:     0.000
# Max Reward:     0.000
# Mean Length:    1024.0 +/- 0.0 steps
# Success Rate:   0.0% (0/20)
# ==================================================