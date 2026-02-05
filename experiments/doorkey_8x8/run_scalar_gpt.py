"""
Train PPO with LLM-Scalar Rewards using GPT-OSS on DoorKey-8x8
Final experimental comparison
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from experiments import (
    make_minigrid_env,
    evaluate_policy,
    PPO,
    ENV_CONFIGS,
    SHARED_PPO_PARAMS,
    LLM_PARAMS,
    EVALUATION_PARAMS,
    move_experiment_files,
    # Scalar approach components
    RobustCachedLLMClient,
    DOOR_KEY_SYSTEM_PROMPT,
    get_DOORKEY_description,
    GPT_OSS_Client
)


def main():
    # Load configuration
    config = ENV_CONFIGS["doorkey_8x8"]
    llm_cfg = LLM_PARAMS
    
    # === Initialize LLM Client ===
    try:
        real_client = GPT_OSS_Client(
            system_prompt=DOOR_KEY_SYSTEM_PROMPT,
            reasoning=True,
            temperature=0.3
        )
        print(f"\nInitialized LLM: {real_client.model_name}")
    except Exception as e:
        print(f"LLM Setup Failed: {e}")
        sys.exit(1)

    # === Setup Cache Path ===
    # Cache is stored in src/methods/llm_guided/ScalarApproach/cache/
    cache_dir = os.path.join(os.path.dirname(__file__), 
                             "../../src/methods/llm_guided/ScalarApproach/cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "doorkey_gpt_oss_cache.json")
    
    # Wrap with caching and guardrails
    llm_client = RobustCachedLLMClient(
        real_client,
        cache_path=cache_path,
        voting_samples=llm_cfg["voting_samples"],
        mode=config["env_id"]
    )

    print(f"\n{'='*60}")
    print(f"Starting PPO with LLM-Scalar Rewards (GPT-OSS)")
    print(f"Environment: DoorKey-8x8")
    print(f"{'='*60}\n")

    # === Create Environment with LLM rewards ===
    env = make_minigrid_env(
        env_id=config["env_id"],
        render_mode="rgb_array",
        use_llm_rewards=True,
        llm_client=llm_client,
        textualizer_fn=get_DOORKEY_description,
        llm_weight=llm_cfg["llm_weight"],
        verbose=False,
        max_steps=config["max_steps"]
    )()

    # === Initialize PPO ===
    policy = PPO(
        env=env,
        epochs=config["epochs"],
        model_name="Scalar_GPT",
        save_pkl_model=True,
        track_stats=True,
        **SHARED_PPO_PARAMS
    )
    
    # Set environment-specific parameters
    policy.batch_size = config["batch_size"]
    policy.rollout.iterations = config["rollout_iterations"]

    # === Train ===
    policy.trainer(
        early_stopping_threshold=config["early_stopping_threshold"],
        window_size=config["early_stopping_window"]
    )

    # Finalize and print LLM statistics
    if hasattr(env, 'finalize_episode'):
        env.finalize_episode()
    if hasattr(env, 'print_statistics_summary'):
        env.print_statistics_summary()
    llm_client.print_stats_summary()

    # Load best model
    policy.load()

    # === Evaluation Phase ===
    print("\n=========== EVALUATION PHASE ===========\n")
    
    # Create fresh eval environment (without LLM rewards for fair comparison)
    eval_env = make_minigrid_env(
        env_id=config["env_id"],
        render_mode="rgb_array",
        use_llm_rewards=llm_client,
        max_steps=config["max_steps"]
    )()

    policy.eval()
    
    stats = evaluate_policy(
        eval_env,
        policy,
        **EVALUATION_PARAMS
    )

    # Move experiment files to results directories
    move_experiment_files(
        model_name="Scalar_GPT",
        environment="DOORKEY_8x8"
    )


if __name__ == "__main__":
    main()
