# =======================================================
# Training script runs PPO with LLM-augmented rewards
# =======================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import warnings
# ---  SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")

import torch
from src.common.env_setup import make_minigrid_env
from src.methods.pure_rl.ppo.ppo_config import PPO, RecurrentPPO

# Import LLM components
from src.methods.llm_guided.cached_llm import RobustCachedLLMClient
from src.methods.llm_guided.llm_shared_utils import DOOR_KEY_SYSTEM_PROMPT, EMPTY_SYSTEM_PROMPT
from src.methods.llm_guided.DoorKey_Textualizer import get_DOORKEY_description
from src.methods.llm_guided.Empty_Textualizer import get_EMPTY_description

# Choose LLM Client 
from src.methods.llm_guided.phi3_5 import Phi35LLMClient
# from src.methods.llm_guided.gemini import GeminiLLMClient



def train_ppo_with_llm(
    env_id="MiniGrid-DoorKey-5x5-v0",
    use_llm=True,
    llm_backend='phi',  # 'phi' or 'gemini'
    llm_weight=1.0,
    epochs=1000,
    max_steps=250,
    cache_name=None,
    verbose=False,
    voting_samples=3
):
    
    llm_client = None
    textualizer_fn = None
    
    if use_llm:
        if "DoorKey" in env_id:
            system_prompt = DOOR_KEY_SYSTEM_PROMPT
            textualizer_fn = get_DOORKEY_description
            name_cache = "doorkey_"+llm_backend+"_cache.json"
            cache_name = cache_name or name_cache
        elif "Empty" in env_id:
            system_prompt = EMPTY_SYSTEM_PROMPT
            textualizer_fn = get_EMPTY_description
            name_cache = "empty_"+llm_backend+"_cache.json"
            cache_name = cache_name or name_cache
        else:
            raise ValueError(f"Unknown environment: {env_id}")
        
        # Initialize LLM
        if llm_backend == 'phi':
            real_client = Phi35LLMClient(debug=False, system_prompt=system_prompt)
        elif llm_backend == 'gemini':
            from src.methods.llm_guided.gemini import GeminiLLMClient
            real_client = GeminiLLMClient(debug=False, system_prompt=system_prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")
        
        # Wrap with caching and guardrails
        llm_client = RobustCachedLLMClient(
            real_client,
            cache_path=cache_name,
            voting_samples=voting_samples,
            mode = env_id 
        )
            
    # === Create Environment ===
    env_fn = make_minigrid_env(
        env_id=env_id,
        render_mode="rgb_array",
        use_llm_rewards=use_llm, # the environment creation depends on LLM usage or not
        llm_client=llm_client,
        textualizer_fn=textualizer_fn,
        llm_weight=llm_weight,
        verbose=verbose,
        max_steps=max_steps #250
    )
    env = env_fn()
    
    # === Setup PPO ===
    policy = RecurrentPPO(env = env,  
                          epochs = epochs, 
                          gamma = 0.99, 
                          epsilon = 0.2,
                          encode_dim=128,  # CNN output
                          hidden_dim=64,    # LSTM hidden size
                          recurrence = "lstm",
                          model_name=f"RecurrentPPO_{env_id.split('-')[1]}_llm_guided",
                          )
    
    # === Train ===
    policy.trainer(
        early_stopping_threshold= 195,  # Stop if avg reward reaches 95%
        window_size=10  # Average over last 10 epochs
    )

    # IMPORTANT: Finalize the last episode (otherwise it's not saved)
    if use_llm and hasattr(env, 'finalize_episode'):
        env.finalize_episode()

    if use_llm and hasattr(env, 'print_statistics_summary'):
        env.print_statistics_summary()
        
        # Print cache stats
        print(f"LLM Cache Statistics:")
        print(f"  Hits:                   {llm_client.stats['hits']}")
        print(f"  Misses:                 {llm_client.stats['misses']}")
        print(f"  Guardrail Corrections:  {llm_client.stats['corrected_by_guardrail']}")
        print(f"{'='*60}\n")
    return policy, env


if __name__ == "__main__":
    # === EXPERIMENT 1: Pure RL (Baseline) ===
    # policy_pure, env_pure = train_ppo_with_llm(
    #     env_id="MiniGrid-DoorKey-5x5-v0",
    #     use_llm=False, # di default la funzione usa LLM
    #     epochs=2
    # )
    
    # # === EXPERIMENT 2: LLM-Guided (Additive Rewards) ===
    policy_llm, env_llm = train_ppo_with_llm(
        env_id="MiniGrid-DoorKey-5x5-v0",
        use_llm=True,
        llm_backend='phi',
        llm_weight=1.0, 
        epochs=100,
        max_steps=250,
        verbose=False, 
        voting_samples=3
    )