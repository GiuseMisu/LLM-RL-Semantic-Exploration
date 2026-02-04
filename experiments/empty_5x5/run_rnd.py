"""
Train RND-PPO (Curiosity-Driven) on empty_5x5 environment
Final experimental comparison
"""
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
    
    config = ENV_CONFIGS["empty_5x5"]
    
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
    print(f"Starting RND-PPO training on Empty-5x5")
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
    move_experiment_files("RNDPPO", "EMPTY_5x5")
        
if __name__ == "__main__":
    main()