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
    EurekaSearch,
    # NOTE : don't need a real LLM client, we're reusing a saved reward function
)


def main():
    
    config = ENV_CONFIGS["doorkey_16x16"]
    eureka_cfg = EUREKA_PARAMS_16x16
    # will reuse a previously generated reward function from the 8x8 experiment
    # No LLM queries will be made here.

    # Load the reward function generated on the 8x8 environment
    reward_src_path = os.path.join(os.path.dirname(__file__), "../../results/reward_functions/doorkey_8x8/Eureka_Qwen_DOORKEY_8x8_reward.py")
    try:
        with open(reward_src_path, "r", encoding="utf-8") as f:
            best_code = f.read()
        print(f"Loaded reward function from: {reward_src_path}")
    except Exception as e:
        print(f"Failed to load reward function from {reward_src_path}: {e}")
        sys.exit(1)

    # LLM-like object so EurekaSearch can initialize
    class _DummyLLM:
        def __init__(self):
            self.model_name = "qwen-from-8x8"

    # Initialize EurekaSearch (only used here to access training utilities)
    eureka = EurekaSearch(
        env_id=config["env_id"],
        llm_model=_DummyLLM(),
        reflection_iterations=eureka_cfg["reflection_iterations"],
        training_epochs=eureka_cfg["training_epochs"],
        train_max_steps=config["max_steps"],
        num_eval_episodes=eureka_cfg["num_eval_episodes"],
        pure_rl_baseline='PPO', #--> Using PPO as the pure RL baseline
        batch_size=config["batch_size"],
        rollout_iterations=config["rollout_iterations"]
    )

    # Phase 2: Final training with best reward function
    print(f"\n{'='*60}")
    print("Training final model on 16x16 ENV with reward function from SMALLER ENVIRONMENT (8x8)")
    print(f"{'='*60}\n")
    
    final_policy = eureka.train_final_model(
        final_train_epochs=config["epochs"],
        final_train_max_steps=config["max_steps"],
        reward_code_str=best_code
    )

    # Load the best trained model
    final_policy.load()

    # # Phase 3: Evaluation
    # print("\n=========== EVALUATION PHASE ===========\n")
    eval_env = make_minigrid_env(
        env_id=config["env_id"],
        render_mode="rgb_array",
        eureka_reward_code=best_code,
        max_steps=config["max_steps"]
    )()

    
    # Evaluate over multiple episodes for statistics
    stats = evaluate_policy(
        eval_env,
        final_policy,
        save_gif=False,
        n_episodes=100,
        #**EVALUATION_PARAMS
    )

    
if __name__ == "__main__":
    main()


