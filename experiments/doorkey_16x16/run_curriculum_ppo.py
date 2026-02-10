"""
Curriculum Learning PPO on DoorKey - progressive training:
    5x5  →  6x6  →  8x8  →  16x16

Each stage trains until early-stopping (avg env reward >= 0.92 over 15 epochs),
then validates on the same env size (30 episodes).  
If validation success_rate > 0.95 the model is promoted to the next larger env. 
The 16x16 is the final stage - it trains to completion.

Resume support:
    python -m experiments.doorkey_16x16.run_curriculum_ppo --resume 3
    (skips stages 0-2 and loads the 8x8 checkpoint to start 16x16 directly)
    Stage indices: 0=5x5, 1=6x6, 2=8x8, 3=16x16
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from experiments import (
    make_minigrid_env,
    evaluate_policy,
    SHARED_PPO_PARAMS,
    EVALUATION_PARAMS,
)

from src.methods.pure_rl.curriculum_learnign.curriculum_trainer import (
    CurriculumTrainer,
    DEFAULT_CURRICULUM_STAGES,
)


def main():

    # ── CLI argument for resuming from a specific stage ──
    parser = argparse.ArgumentParser(description="Curriculum PPO on DoorKey")
    parser.add_argument(
        "--resume", type=int, default=None, metavar="STAGE",
        help="Resume from this stage index (0=5x5, 1=6x6, 2=8x8, 3=16x16). "
             "Loads the checkpoint from the previous stage automatically."
    )
    args = parser.parse_args()

    stages = DEFAULT_CURRICULUM_STAGES          # 5x5 -> 6x6 -> 8x8 -> 16x16
    promotion_threshold = 0.95                  # validation success-rate to advance
    validation_episodes = 30                    # episodes for validation checks

    # ── Create & run curriculum trainer ──
    trainer = CurriculumTrainer(
        stages=stages,
        ppo_params=SHARED_PPO_PARAMS,           # gamma=0.99, epsilon=0.2
        promotion_threshold=promotion_threshold,
        validation_episodes=validation_episodes,
        model_dir="results/models/curriculum",
        log_dir="logs/Curriculum_DoorKey16x16",
        resume_from_stage=args.resume,
    )

    final_ckpt = trainer.run()

    # - Final evaluation on DoorKey-16x16 with the best model ──
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION  -  DoorKey-16x16")
    print("=" * 60 + "\n")

    final_stage = stages[-1]   # last config: 16x16 config
    eval_env = make_minigrid_env(
        env_id=final_stage["env_id"],
        render_mode="rgb_array",
        use_llm_rewards=False,
        max_steps=final_stage["max_steps"],
    )()

    # Build a fresh policy and load the curriculum checkpoint
    from src.methods.pure_rl.ppo.ppo_config import PPO

    policy = PPO(
        env=eval_env,
        epochs=1,                               # not training, just need structure
        model_name="CurriculumPPO",
        save_pkl_model=False,
        track_stats=False,
        **SHARED_PPO_PARAMS,
    )

    if final_ckpt is not None and os.path.exists(final_ckpt):
        import torch
        state_dict = torch.load(final_ckpt, weights_only=True, map_location=policy.device)
        policy.load_state_dict(state_dict)
        print(f"[EVAL] Loaded checkpoint: {final_ckpt}")
    else:
        print("[EVAL] No curriculum checkpoint found - evaluating untrained policy")

    stats = evaluate_policy(
        eval_env,
        policy,
        **EVALUATION_PARAMS,
    )



if __name__ == "__main__":
    main()
