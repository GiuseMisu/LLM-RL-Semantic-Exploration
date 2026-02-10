"""
Curriculum Learning utilities for PPO on MiniGrid environments.
Progressive training across growing environment sizes.
"""

from src.methods.pure_rl.curriculum_learnign.curriculum_trainer import CurriculumTrainer

__all__ = ['CurriculumTrainer']
