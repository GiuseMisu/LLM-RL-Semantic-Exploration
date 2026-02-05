"""
Experiments package - centralizes common imports for all experiment scripts
"""
import sys
import os

# project root
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

# Common imports for all experiments
from src.common.env_setup import make_minigrid_env
from src.common.policy_evaluation import evaluate_policy
from experiments.experiment_config import ENV_CONFIGS, SHARED_PPO_PARAMS, RECURRENT_PARAMS, RND_PARAMS, EUREKA_PARAMS, EVALUATION_PARAMS
from experiments.experiment_utils import move_experiment_files

# PPO variants
from src.methods.pure_rl.ppo.ppo_config import PPO, RecurrentPPO
from src.methods.curiosity_driven.rnd_ppo_config import RNDPPO

# Eureka approach
from src.methods.llm_guided.EurekaApproach.eureka_search import EurekaSearch
from src.methods.llm_guided.llm_clients.Qwen3Code480b import Qwen3CoderClient
from src.methods.llm_guided.llm_clients.DeepSeek671b import DeepSeekCloud671b_Client

__all__ = [
    'make_minigrid_env',
    'evaluate_policy',
    'ENV_CONFIGS',
    'SHARED_PPO_PARAMS',
    'RECURRENT_PARAMS',
    'RND_PARAMS',
    'EUREKA_PARAMS',
    'EVALUATION_PARAMS',
    'move_experiment_files',
    'PPO',
    'RecurrentPPO',
    'RNDPPO',
    'EurekaSearch',
    'Qwen3CoderClient',
    'DeepSeekCloud671b_Client',
]
