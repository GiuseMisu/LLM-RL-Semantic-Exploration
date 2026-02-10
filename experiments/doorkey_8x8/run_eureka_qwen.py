"""
Train Eureka (LLM-Generated Reward Functions) with Qwen on DoorKey-8x8
Final experimental comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from experiments import (
    make_minigrid_env,
    evaluate_policy,
    ENV_CONFIGS,
    EUREKA_PARAMS,
    EVALUATION_PARAMS,
    EurekaSearch,
    Qwen3CoderClient
)


def main():
    # Load configuration
    config = ENV_CONFIGS["doorkey_8x8"]
    eureka_cfg = EUREKA_PARAMS
    
    # Initialize LLM client
    try:
        llm = Qwen3CoderClient(
            temperature=0.1  # Low temperature for code generation
        )
        print(f"\nInitialized LLM: {llm.model_name}")
    except Exception as e:
        print(f"LLM Setup Failed: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Starting Eureka Reward Function Search with Qwen")
    print(f"Environment: DoorKey-8x8")
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

    
if __name__ == "__main__":
    main()



# Episode 1/20: Reward=11.04, Steps=11
# Episode 2/20: Reward=11.06, Steps=15
# Episode 3/20: Reward=11.05, Steps=19
# Episode 4/20: Reward=11.05, Steps=16
# Episode 5/20: Reward=11.04, Steps=14
# Episode 6/20: Reward=11.06, Steps=17
# Episode 7/20: Reward=11.04, Steps=13
# Episode 8/20: Reward=11.04, Steps=17
# Episode 9/20: Reward=11.04, Steps=15
# Episode 10/20: Reward=11.04, Steps=14
# Episode 11/20: Reward=11.04, Steps=12
# Episode 12/20: Reward=11.03, Steps=14
# Episode 13/20: Reward=11.06, Steps=21
# Episode 14/20: Reward=11.07, Steps=20
# Episode 15/20: Reward=11.06, Steps=18
# Episode 16/20: Reward=11.05, Steps=17
# Episode 17/20: Reward=11.05, Steps=16
# Episode 18/20: Reward=11.05, Steps=14
# Episode 19/20: Reward=11.06, Steps=17
# Episode 20/20: Reward=11.06, Steps=18

# ==================================================
# EVALUATION STATISTICS
# ==================================================
# Mean Reward:    11.049 +/- 0.010
# Min Reward:     11.028
# Max Reward:     11.071
# Mean Length:    15.9 +/- 2.5 steps
# Success Rate:   100.0% (20/20)
# ==================================================