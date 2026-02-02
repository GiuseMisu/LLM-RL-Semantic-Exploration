import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../"))
from src.methods.llm_guided.llm_clients.gpt_oss import GPT_OSS_Client

if __name__ == "__main__":
    
    try:
        # Initialize the client
        client = GPT_OSS_Client(
            reasoning=True, 
            temperature=0.3        
            )
        print(f"Model: {client.model_name}")
    except Exception as e:
        print(f"Setup Failed: {e}")
        sys.exit(1)

    # --- HELPER FUNCTION FOR CONSISTENT TESTING ---
    def run_test(case_name, obs):
        print(f"\n\n--- {case_name}")
        # Note: We call .get_reward() which is inherited from BaseLLMClient
        # It handles the timing, JSON repair, parsing, and printing automatically.
        client.get_reward(obs, verbose = True, generate_explanation=True)
        sys.stdout.flush()

    # --- TEST CASES ---
    # CASE 1: Perfect Scenario (Should get 0.5)
    obs_1 = "{ 'Agent': { 'pos': (1, 2), 'facing': 'South', 'inventory': 'None' }, 'Key': 'loc=(1, 3), dist=1, dir=Front <REACHABLE>', 'Door': 'loc=(2, 2), dist=1, dir=Left, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Left' }"
    run_test("TEST CASE 1: Key Reachable", obs_1)

    # CASE 3: Ready to Open Door
    # Agent HAS the key. Key status is "In Inventory (Carried)". Door is reachable.
    obs_3 = "{ 'Agent': { 'pos': (1, 2), 'facing': 'East', 'inventory': 'yellow key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(2, 2), dist=1, dir=Front <REACHABLE>, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Front-Right' }"
    run_test("TEST CASE 3: Ready to Unlock", obs_3)

    # CASE 4: Holding Key but Wrong Direction
    # Agent HAS the key, but is facing the Wall (West) instead of the Door (East). Door is Behind.
    obs_4 = "{ 'Agent': { 'pos': (1, 2), 'facing': 'West', 'inventory': 'yellow key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(2, 2), dist=1, dir=Behind, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Behind' }"
    run_test("TEST CASE 4: Wrong Facing (Has Key)", obs_4)

    # CASE 5: Phase 3 Transition - Door is Open
    # Door state is 'Open'. Agent is standing ON the door (dist=0, dir=Here).
    obs_5 = "{ 'Agent': { 'pos': (2, 2), 'facing': 'East', 'inventory': 'yellow key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(2, 2), dist=0, dir=Here, state=Open', 'Goal': 'loc=(3, 3), dist=2, dir=Front-Right' }"
    run_test("TEST CASE 5: Door Open (Passing Through)", obs_5)

    # CASE 6: Goal Reached
    # Agent overlaps with Goal.
    obs_6 = "{ 'Agent': { 'pos': (3, 3), 'facing': 'South', 'inventory': 'yellow key' }, 'Key': 'In Inventory (Carried)', 'Door': 'loc=(2, 2), dist=2, dir=Behind, state=Open', 'Goal': 'loc=(3, 3), dist=0, dir=Here' }"
    run_test("TEST CASE 6: Goal Reached", obs_6)