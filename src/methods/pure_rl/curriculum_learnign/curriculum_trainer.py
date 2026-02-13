"""
Curriculum Learning Trainer for PPO on MiniGrid DoorKey environments.

Trains PPO progressively across a sequence of growing environments:
  5x5 -> 6x6 -> 8x8 -> 16x16

For each stage:
  1. Create a new PPO agent on the current env (loading weights from the previous stage if available)
  2. Train with early stopping (avg env reward >= threshold over N consecutive epochs)
  3. Validate on a held-out set of episodes (same env dimension)
  4. If validation success_rate > promotion_threshold, move to the next larger env
  5. The last stage (16x16) trains to completion without promotion

The CNN architecture (MiniGridCNN) uses 7x7x3 partial observations (agent-centric view),
so the same network works across all grid sizes — no architecture change needed.
"""

import os
import torch
from torch.nn import functional as F

from src.common.env_setup import make_minigrid_env
from src.common.policy_evaluation import evaluate_policy
from src.methods.pure_rl.ppo.ppo_config import PPO


DEFAULT_CURRICULUM_STAGES = [
    {
        "env_id": "MiniGrid-DoorKey-5x5-v0",
        "label": "5x5",
        "max_steps": 250,
        "epochs": 100,
        "batch_size": 2048,
        "rollout_iterations": 4096,
        "early_stopping_threshold": 0.92,
        "early_stopping_window": 15,
    },
    {
        "env_id": "MiniGrid-DoorKey-6x6-v0",
        "label": "6x6",
        "max_steps": 400,
        "epochs": 125,
        "batch_size": 4096,
        "rollout_iterations": 8192,
        "early_stopping_threshold": 0.92,
        "early_stopping_window": 15,
    },
    {
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "label": "8x8",
        "max_steps": 640,
        "epochs": 150,
        "batch_size": 4096,
        "rollout_iterations": 16384,
        "early_stopping_threshold": 0.92,
        "early_stopping_window": 15,
    },
    {
        "env_id": "MiniGrid-DoorKey-16x16-v0",
        "label": "16x16",
        "max_steps": 1024,
        "epochs": 300,
        "batch_size": 8192,
        "rollout_iterations": 32768,
        "early_stopping_threshold": 0.92,
        "early_stopping_window": 15,
    },
]


class CurriculumTrainer:
    """
    curriculum learning across a sequence of MiniGrid environments.

    Parameters
    ----------
    stages : list[dict]
        Ordered list of stage configurations (smallest to largest).
        Each dict contain: 
        env_id, label, max_steps,
        epochs, batch_size, rollout_iterations,
        early_stopping_threshold, early_stopping_window.
    ppo_params : dict
        Shared PPO hyperparameters (gamma, epsilon, …).
    promotion_threshold : float
        Validation success-rate required to advance to the next stage.
    validation_episodes : int
        Number of episodes used for validation after early stopping.
    model_dir : str
        Directory where stage checkpoints are saved.
    """

    def __init__(
        self,
        stages=None,
        ppo_params=None,
        promotion_threshold: float = 0.95,
        validation_episodes: int = 30,
        model_dir: str = "results/models/curriculum",
        log_dir: str = "logs",
        resume_from_stage: int | None = None,
    ):
        self.stages = stages or DEFAULT_CURRICULUM_STAGES
        self.ppo_params = ppo_params or {"gamma": 0.99, "epsilon": 0.2}
        self.promotion_threshold = promotion_threshold
        self.validation_episodes = validation_episodes
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.resume_from_stage = resume_from_stage

        # will hold the path of the last saved checkpoint
        self._previous_model_path = None

    def run(self):
        """Execute the full curriculum from stage 0 (or resume_from_stage) to the last stage."""
        os.makedirs(self.model_dir, exist_ok=True)

        start_idx = 0
        if self.resume_from_stage is not None:
            start_idx = self.resume_from_stage
            # Look for the checkpoint of the stage *before* the one we resume from
            # so we can transfer its weights into the resumed stage.
            if start_idx > 0:
                prev_label = self.stages[start_idx - 1]["label"]
                prev_ckpt = os.path.join(
                    self.model_dir,
                    f"CurriculumPPO_DoorKey_{prev_label}_best.pkl",
                )
                if os.path.exists(prev_ckpt):
                    self._previous_model_path = prev_ckpt
                    print(f"[RESUME] Found checkpoint from stage {start_idx - 1} "
                          f"({prev_label}): {prev_ckpt}")
                else:
                    # Try auto-detecting the latest available checkpoint
                    detected = self._find_latest_checkpoint()
                    if detected is not None:
                        self._previous_model_path = detected
                        print(f"[RESUME] Auto-detected latest checkpoint: {detected}")
                    else:
                        print(f"[RESUME] WARNING: No checkpoint found for stage "
                              f"{start_idx - 1} ({prev_label}). "
                              f"Training stage {start_idx} from scratch.")

            print(f"[RESUME] Starting curriculum from stage {start_idx} "
                  f"({self.stages[start_idx]['label']})\n")

        for idx in range(start_idx, len(self.stages)):
            stage = self.stages[idx]
            is_last_stage = (idx == len(self.stages) - 1)
            label = stage["label"]

            print("\n" + "=" * 70)
            print(f"  CURRICULUM STAGE {idx + 1}/{len(self.stages)}  —  DoorKey-{label}")
            print("=" * 70 + "\n")

            # 1 - Build environment & PPO agent for this stage
            env = make_minigrid_env(
                env_id=stage["env_id"],
                render_mode="rgb_array",
                use_llm_rewards=False,
                max_steps=stage["max_steps"],
            )()

            policy = PPO(
                env=env,
                epochs=stage["epochs"],
                model_name=f"CurriculumPPO_stage{idx}",
                save_pkl_model=True,
                track_stats=True,
                **self.ppo_params,
            )
            policy.batch_size = stage["batch_size"]
            policy.rollout.iterations = stage["rollout_iterations"]

            # 2 - Transfer weights from previous stage (if any)
            if self._previous_model_path is not None:
                print(f"[CURRICULUM] Loading weights from previous stage: {self._previous_model_path}")
                self._transfer_weights(policy, self._previous_model_path)

            # 3 - Train with early stopping
            print(f"[CURRICULUM] Training on {stage['env_id']}  |  "
                  f"early_stop={stage['early_stopping_threshold']} "
                  f"window={stage['early_stopping_window']}")

            training_history = policy.trainer(
                early_stopping_threshold=stage["early_stopping_threshold"],
                window_size=stage["early_stopping_window"],
            )

            # 4 - Save the best checkpoint for this stage
            stage_ckpt = os.path.join(
                self.model_dir,
                f"CurriculumPPO_DoorKey_{label}_best.pkl",
            )
            # The trainer already saved the best model via policy.save(); copy it
            self._save_stage_checkpoint(policy, stage_ckpt)
            print(f"[CURRICULUM] Stage checkpoint saved → {stage_ckpt}")

            # 5 - Validation on same-dimension env
            val_stats = self._validate(policy, stage)
            val_success = val_stats["success_rate"]
            print(f"[CURRICULUM] Validation success rate on {label}: {val_success:.2%}")

            # 6 - Decide promotion
            if is_last_stage:
                print(f"\n[CURRICULUM] Final stage ({label}) complete - no further promotion.")
                self._previous_model_path = stage_ckpt
                break

            if val_success >= self.promotion_threshold:
                print(f"[CURRICULUM] Promoted!  {val_success:.2%} >= {self.promotion_threshold:.2%}")
                self._previous_model_path = stage_ckpt
            else:
                print(f"[CURRICULUM] Validation below threshold "
                      f"({val_success:.2%} < {self.promotion_threshold:.2%}).  "
                      f"Stopping curriculum at stage {label}.")
                break

        print("\n" + "=" * 70)
        print("  CURRICULUM LEARNING FINISHED")
        print("=" * 70)
        return self._previous_model_path


    #  Internal helpers
    def _find_latest_checkpoint(self) -> str | None:
        """
        Scan model_dir for existing stage checkpoints and return the path
        of the most advanced one (by stage index), or None if nothing found.
        """
        latest_path = None
        for idx, stage in enumerate(self.stages):
            candidate = os.path.join(
                self.model_dir,
                f"CurriculumPPO_DoorKey_{stage['label']}_best.pkl",
            )
            if os.path.exists(candidate):
                latest_path = candidate
        return latest_path

    @staticmethod
    def _transfer_weights(policy: PPO, checkpoint_path: str):
        """
        Load weights from a checkpoint into *policy*.
        Because all stages share the same architecture (MiniGridCNN 7x7x3
        partial obs -> BaseNet actor/critic), a direct state_dict load works.
        Resets the optimizer so the new stage starts fresh.
        """
        saved_state = torch.load(checkpoint_path, weights_only=True, map_location=policy.device)
        policy.load_state_dict(saved_state)
        # Re-initialise the optimizer so momentum / adaptive terms start fresh
        from torch.optim import Adam
        policy.optimizer = Adam(policy.parameters(), lr=policy.lr)
        print(f"[TRANSFER] Weights loaded and optimizer reset.")

    @staticmethod
    def _save_stage_checkpoint(policy: PPO, path: str):
        """Save the current model state to *path*."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        policy.to("cpu")
        torch.save(policy.state_dict(), path)
        policy.to(policy.device)

    def _validate(self, policy: PPO, stage: dict) -> dict:
        """
        Run validation episodes on an independent env instance
        of the same dimension and return statistics.
        """
        print(f"\n--- Validation on {stage['env_id']}  "
              f"({self.validation_episodes} episodes) ---")

        val_env = make_minigrid_env(
            env_id=stage["env_id"],
            render_mode="rgb_array",
            use_llm_rewards=False,
            max_steps=stage["max_steps"],
        )()

        # Load the best model saved during training
        policy.load()

        stats = evaluate_policy(
            val_env,
            policy,
            n_episodes=self.validation_episodes,
            save_gif=False,
        )

        # Restore training mode for potential next stage
        policy.train()
        val_env.close()
        return stats
