"""
Train Eureka (LLM-Generated Reward Functions) with Qwen on DoorKey-16x16
Final experimental comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from experiments import (
    make_minigrid_env,
    evaluate_policy,
    ENV_CONFIGS,
    EUREKA_PARAMS_16x16,
    EVALUATION_PARAMS,
    move_experiment_files,
    EurekaSearch,
    GPT_OSS_Client
)

def main():
    # Load configuration
    config = ENV_CONFIGS["doorkey_16x16"]
    eureka_cfg = EUREKA_PARAMS_16x16
    
    # Initialize LLM client
    try:
        llm = GPT_OSS_Client(
            reasoning=True, 
            temperature=0.1 #more deterministic        
            )
        print(f"\nInitialized LLM: {llm.model_name}")
    except Exception as e:
        print(f"LLM Setup Failed: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Starting Eureka Reward Function Search with GPT-OSS")
    print(f"Environment: DoorKey-16x16")
    print(f"{'='*60}\n")

    # Initialize Eureka Search
    eureka = EurekaSearch(
        env_id=config["env_id"],
        llm_model=llm,
        reflection_iterations=eureka_cfg["reflection_iterations"],
        training_epochs=eureka_cfg["training_epochs"],
        train_max_steps=config["max_steps"],
        num_eval_episodes=eureka_cfg["num_eval_episodes"],
        pure_rl_baseline='PPO', #--> Using PPO as the pure RL baseline
        batch_size=config["batch_size"],
        rollout_iterations=config["rollout_iterations"]
    )

    # Phase 1: Search for best reward function
    print("\n[PHASE 1] Searching for optimal reward function\n")
    best_code = eureka.find_best_RwdFunc()
    
    if best_code is None:
        print("\nFailed to generate valid reward function")
        sys.exit(1)

    # Phase 2: Final training with best reward function
    print(f"\n{'='*60}")
    print("[PHASE 2] Training final model with best reward function")
    print(f"{'='*60}\n")
    
    final_policy = eureka.train_final_model(
        final_train_epochs=config["epochs"],
        final_train_max_steps=config["max_steps"],
        reward_code_str=best_code
    )

    # Load the best trained model
    final_policy.load()

    # Phase 3: Evaluation
    print("\n=========== EVALUATION PHASE ===========\n")
    eval_env = make_minigrid_env(
        env_id=config["env_id"],
        render_mode="rgb_array",
        eureka_reward_code=best_code,
        max_steps=config["max_steps"]
    )()

    final_policy.eval()
    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(
        eval_env,
        final_policy,
        **EVALUATION_PARAMS
    )

    # Move all experiment files to results directories
    move_experiment_files(
        model_name="Eureka_GPT_OSS",
        environment="DOORKEY_16x16",
        source_pattern="PPO_FINAL__",
        include_reward_function=True
    )
    
if __name__ == "__main__":
    main()


