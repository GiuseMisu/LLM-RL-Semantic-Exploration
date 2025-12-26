import ollama
import sys
import json
import re
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from src.methods.llm_guided.llm_shared_utils import SYSTEM_PROMPT


MODEL_NAME = "qwen2.5:1.5b"

def get_llm_reward(mock_observation):
    #Sends the system prompt + observation to the local LLM

    print(f"\nScanning State: {mock_observation}")
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format='json', # Forces the LLM to output clean JSON
            messages=[
                # --- THIS IS HOW YOU PASS THE SYSTEM PROMPT ---
                {'role': 'system', 'content': SYSTEM_PROMPT}, 
                
                # The actual observation from your textualizer
                {'role': 'user', 'content': f"Observation: {mock_observation}"} 
            ]
        )
        
        # Parse the result
        content = response['message']['content']
        data = json.loads(content)
        
        print("-" * 40)
        print(f"REWARD: {data.get('reward')}")
        print(f"REASON: {data.get('reasoning')}")
        print("-" * 40)
        return data.get('reward')
        
    except Exception as e:
        print(f"Error: {e}")

def run_test():
    print(f"--- CONNECTING TO OLLAMA ({MODEL_NAME}) ---")
    
    try:
        # Get list of models
        models_info = ollama.list()
        
        # --- Handle different library versions (Dictionary vs Object) ---
        available_models = []
        for m in models_info['models']:
            # If it's a dictionary (standard), get 'model'. If missing, try 'name'.
            if isinstance(m, dict):
                name = m.get('model') or m.get('name')
            # If it's an object (newer versions), use dot notation
            else:
                name = getattr(m, 'model', getattr(m, 'name', str(m)))
            
            if name:
                available_models.append(name)

        print(f"   Found models: {available_models}")
        
        # Check if our model is there (ignoring :latest tag differences if needed)
        if not any(MODEL_NAME in m for m in available_models):
            print(f"[WARNING]: '{MODEL_NAME}' might be missing or named differently.")
        
    except Exception as e:
        print(f"Connection Warning: {e}")
        print(" (Continuing anyway, as this might just be a version listing error...)")

    # print("\n--- 2. LOGIC TEST (Math) ---")
    # prompt_1 = "If I have 3 apples and eat one, then buy two more, how many do I have?"
    # print(f"Query: {prompt_1}")
    
    # try:
    #     response = ollama.chat(model=MODEL_NAME, messages=[
    #         {'role': 'user', 'content': prompt_1}
    #     ])
    #     print(f"Answer: {response['message']['content']}")
    # except Exception as e:
    #     print(f"Inference Failed: {e}")

    # --- TEST: Project Simulation (JSON Output) ---
    print("\n--- PROJECT SIMULATION (RL Agent Reward) ---")
    #mock_obs_without_dist = "{ 'Agent': { 'pos': (1, 2), 'facing': 'West', 'inventory': 'None' }, 'Key': 'pos=(1, 3)', 'Door': 'pos=(2, 2), state=Locked', 'Goal_Dist': pos=(3, 3) }" 
    #reward = get_llm_reward(mock_obs_without_dist)
    #print(f"Computed Reward: {reward}")
    
    mock_obs_wit_dist = "{ 'Agent': { 'pos': (1, 2), 'facing': 'South', 'inventory': 'None' }, 'Key': 'loc=(1, 3), dist=1, dir=Front <REACHABLE>', 'Door': 'loc=(2, 2), dist=1, dir=Left, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Left' }"
    reward = get_llm_reward(mock_obs_wit_dist)
    print(f"Computed Reward: {reward}")

if __name__ == "__main__":
    run_test()


