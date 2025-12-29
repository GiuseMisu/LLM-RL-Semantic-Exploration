import sys
import os

# Adjust path to find your src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

# CORRECT IMPORTS (Fixing the errors found in your main block)
from src.methods.llm_guided.cached_llm import RobustCachedLLMClient
from src.methods.llm_guided.phi3_5 import Phi35LLMClient
from src.methods.llm_guided.llm_shared_utils import DOOR_KEY_SYSTEM_PROMPT, EMPTY_SYSTEM_PROMPT


if __name__ == "__main__":
    print("===============================================")
    print("  VERIFYING DECOUPLING of DOORKEY/EMPTY MODES ")
    print("===============================================")

    print("\n[SCENARIO A] Testing DoorKey Mode")
    try:
        # 1. Init Real Client with DoorKey Prompt
        dk_real = Phi35LLMClient(debug=True, system_prompt=DOOR_KEY_SYSTEM_PROMPT)
        # 2. Wrap with Caching/Guardrails
        dk_wrapper = RobustCachedLLMClient(dk_real, cache_path="test_DOORKEY_real.json", voting_samples=3)
        
        # Check Mode Detection
        print(f"   -> Detected Mode: {dk_wrapper.mode}")
        if dk_wrapper.mode != "DOORKEY":
            print("   -> ERROR: Failed to detect DOORKEY mode")

        # 3. Real Observation
        obs_dk = "{ 'Agent': { 'pos': (1, 1), 'facing': 'East', 'inventory': 'None' }, 'Key': 'loc=(2, 1), dist=1, dir=Front <REACHABLE>', 'Door': 'Not Found', 'Goal': 'Unknown' }"
        reward = dk_wrapper.robust_get_reward(obs_dk, verbose=True)
        print(f"   -> Result: {reward}")

    except Exception as e:
        print(f"   -> CRITICAL FAIL in DoorKey Test: {e}")


    print("\n[SCENARIO B] Testing Empty Mode")
    try:
        # 1. Init Real Client with EMPTY Prompt
        empty_real = Phi35LLMClient(debug=True, system_prompt=EMPTY_SYSTEM_PROMPT)
        
        # 2. Wrap with Caching/Guardrails
        empty_wrapper = RobustCachedLLMClient(empty_real, cache_path="test_EMPTY_real.json", voting_samples=3)

        # Check Mode Detection
        print(f"   -> Detected Mode: {empty_wrapper.mode}")
        if empty_wrapper.mode != "EMPTY":
            print("   -> ERROR: Failed to detect EMPTY mode")

        # 3. Real Observation (Goal is visible but far)
        # This tests if the LLM understands the Empty textual format
        obs_empty = "{ 'Agent': { 'pos': (1, 1), 'facing': 'East' }, 'Goal': 'loc=(5, 1), dist=4, dir=Front' }"
        reward_empty = empty_wrapper.robust_get_reward(obs_empty, verbose=True)
        print(f"   -> Result: {reward_empty}")

    except Exception as e:
        print(f"   -> CRITICAL FAIL in Empty Test: {e}")