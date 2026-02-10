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
from experiments.experiment_config import ENV_CONFIGS, SHARED_PPO_PARAMS, RECURRENT_PARAMS, RND_PARAMS, RND_PARAMS_16x16, EUREKA_PARAMS, EUREKA_PARAMS_16x16, LLM_PARAMS, EVALUATION_PARAMS

# PPO variants
from src.methods.pure_rl.ppo.ppo_config import PPO, RecurrentPPO
from src.methods.curiosity_driven.rnd_ppo_config import RNDPPO

# Eureka approach
from src.methods.llm_guided.EurekaApproach.eureka_search import EurekaSearch
from src.methods.llm_guided.llm_clients.Qwen3Code480b import Qwen3CoderClient
from src.methods.llm_guided.llm_clients.DeepSeek671b import DeepSeekCloud671b_Client

# Scalar approach (LLM-guided rewards)
from src.methods.llm_guided.ScalarApproach.cached_llm import RobustCachedLLMClient
from src.methods.llm_guided.ScalarApproach.scalar_prompts import DOOR_KEY_SYSTEM_PROMPT, EMPTY_SYSTEM_PROMPT
from src.methods.llm_guided.ScalarApproach.DoorKey_Textualizer import get_DOORKEY_description
from src.methods.llm_guided.ScalarApproach.Empty_Textualizer import get_EMPTY_description
from src.methods.llm_guided.llm_clients.hermes3 import HermesLLMClient
from src.methods.llm_guided.llm_clients.gpt_oss import GPT_OSS_Client

__all__ = [
    'make_minigrid_env',
    'evaluate_policy',
    'ENV_CONFIGS',
    'SHARED_PPO_PARAMS',
    'RECURRENT_PARAMS',
    'RND_PARAMS',
    'RND_PARAMS_16x16',
    'EUREKA_PARAMS',
    'EUREKA_PARAMS_16x16',
    'LLM_PARAMS',
    'EVALUATION_PARAMS',
    'PPO',
    'RecurrentPPO',
    'RNDPPO',
    'EurekaSearch',
    'Qwen3CoderClient',
    'DeepSeekCloud671b_Client',
    # Scalar approach
    'RobustCachedLLMClient',
    'DOOR_KEY_SYSTEM_PROMPT',
    'EMPTY_SYSTEM_PROMPT',
    'get_DOORKEY_description',
    'get_EMPTY_description',
    'HermesLLMClient',
    'GPT_OSS_Client',
]
