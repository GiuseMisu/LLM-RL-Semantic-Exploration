# =======================================================
# Training script runs PPO with LLM-augmented rewards
# =======================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from src.common.env_setup import make_minigrid_env
from src.methods.pure_rl.ppo.ppo_config import PPO

# Import LLM components
from src.methods.llm_guided.ScalarApproach.cached_llm import RobustCachedLLMClient

from src.methods.llm_guided.ScalarApproach.scalar_prompts import DOOR_KEY_SYSTEM_PROMPT, EMPTY_SYSTEM_PROMPT
from src.methods.llm_guided.ScalarApproach.DoorKey_Textualizer import get_DOORKEY_description
from src.methods.llm_guided.ScalarApproach.Empty_Textualizer import get_EMPTY_description

# Choose LLM Client 
from src.methods.llm_guided.llm_clients.phi3_5 import Phi35LLMClient
from src.methods.llm_guided.llm_clients.deepseek_r1 import DeepSeekLLMClient
from src.methods.llm_guided.llm_clients.hermes3 import HermesLLMClient
from src.methods.llm_guided.llm_clients.DeepSeek671b import DeepSeekCloud671b_Client



def train_ppo_with_llm(
    env_id="MiniGrid-DoorKey-5x5-v0",
    use_llm=True,
    llm_backend='phi',  # 'phi' or else
    llm_weight=1.0,
    epochs=1000,
    max_steps=250,
    cache_name=None,
    verbose=False,
    voting_samples=3,
    load : bool  = False
):
    
    llm_client = None
    textualizer_fn = None
    
    if use_llm:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        if "DoorKey" in env_id:
            system_prompt = DOOR_KEY_SYSTEM_PROMPT
            textualizer_fn = get_DOORKEY_description
            default_cache = "doorkey_"+llm_backend+"_cache.json"
        elif "Empty" in env_id:
            system_prompt = EMPTY_SYSTEM_PROMPT
            textualizer_fn = get_EMPTY_description
            default_cache = "empty_"+llm_backend+"_cache.json"
        else:
            raise ValueError(f"Unknown environment: {env_id}")
        
        # Build full cache path
        target_filename = cache_name or default_cache
        if not os.path.isabs(target_filename):
            cache_name = os.path.join(cache_dir, target_filename)
        else:
            cache_name = target_filename
        
        # Initialize LLM
        if llm_backend == 'phi':
            real_client = Phi35LLMClient(system_prompt=system_prompt)
        elif llm_backend == 'deepseek':
            real_client = DeepSeekLLMClient(system_prompt=system_prompt)  
        elif llm_backend == 'deepseek671b':
            real_client = DeepSeekCloud671b_Client(system_prompt=system_prompt, 
                                                   reasoning=True,
                                                   temperature=0.3)
        elif llm_backend == 'hermes':
            real_client = HermesLLMClient(debug=False, system_prompt=system_prompt)
        elif llm_backend == 'gpt':
            from src.methods.llm_guided.llm_clients.gpt_oss import GPT_OSS_Client
            real_client = GPT_OSS_Client(system_prompt=system_prompt, reasoning=True, temperature=0.3)
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
        max_steps=max_steps 
    )
    env = env_fn()
    
    # === Setup PPO ===

    policy = PPO(
        env=env,
        gamma=0.99,
        epsilon=0.2,
        epochs=epochs,
        model_name=f"PPO_{env_id.split('-')[1]}_llm_guided",
    )

    policy.batch_size = 1024
    policy.rollout.iterations = 2048
    
    if load:
        policy.load()

    # === Train ===
    policy.trainer(
        early_stopping_threshold = 0.95,  # average ENV_RWD threshold for early stopping 
        window_size=10  # Average over last 10 epochs
    )


    # IMPORTANT: Finalize the last episode (otherwise it's not saved)
    if use_llm and hasattr(env, 'finalize_episode'):
        env.finalize_episode()

    if use_llm and hasattr(env, 'print_statistics_summary'):
        env.print_statistics_summary()
        # Print cache & guardrail stats using the new method
        llm_client.print_stats_summary()
        
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
        env_id="MiniGrid-DoorKey-8x8-v0",
        use_llm=True,
        llm_backend='gpt', # 'phi' or 'deepseek' or 'deepseek671b'
        llm_weight=1.0, 
        epochs=5,
        max_steps=640,
        verbose=True, 
        voting_samples=1
    )