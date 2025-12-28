import time
import sys
import os

# Adjust path to find the sibling file 'gemini.py'
# Assumes this file is in src/methods/llm_guided/llm_tests/
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from gemini import GeminiLLMClient

if __name__ == "__main__":
    
    print("--- INITIALIZING GEMINI CLIENT ---")
    
    try:
        # Initialize the client
        MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
        client = GeminiLLMClient(debug=True, model_name=MODEL_NAME)
        print(f"Model: {client.model_name}")
    except Exception as e:
        print(f"Setup Failed: {e}")
        print("Tip: Make sure GEMINI_API_KEY is set.")
        sys.exit(1)

    # --- HELPER FUNCTION FOR CONSISTENT TESTING ---
    def run_test(case_name, obs):
        print(f"\n\n--- {case_name}")
        # Calls .get_reward() inherited from BaseLLMClient
        # It handles timing, JSON repair, parsing, and printing.
        client.get_reward(obs, verbose = True, generate_explanation=False)
        sys.stdout.flush()
    
    
    # --- TEST CASES ---
    # CASE 1: Perfect Scenario (Should get 0.5)
    obs_1 = "{ 'Agent': { 'pos': (1, 2), 'facing': 'South', 'inventory': 'None' }, 'Key': 'loc=(1, 3), dist=1, dir=Front <REACHABLE>', 'Door': 'loc=(2, 2), dist=1, dir=Left, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Left' }"
    run_test("TEST CASE 1: Key Reachable", obs_1)

    # CASE 2: Distraction Scenario (Should get low reward or negative)
    # Agent is near the door but doesn't have the key. 
    # 1.5B models often fail this and give 0.5 because they see "dist=1" for door.
    obs_2 = "{ 'Agent': { 'pos': (2, 3), 'facing': 'North', 'inventory': 'None' }, 'Key': 'loc=(5, 5), dist=4, dir=Behind', 'Door': 'loc=(2, 2), dist=1, dir=Front <REACHABLE>, state=Locked', 'Goal': 'Unknown' }"
    run_test("TEST CASE 2: Distracted by Door", obs_2)

    # CASE 3: Ready to Open Door
    # Scenario: Agent has the Key, is standing next to the Door (2,2), facing it.
    # Logic: Inventory='Key' -> Phase 2. Door is <REACHABLE>. Reward should be 0.5.
    obs_3 = "{ 'Agent': { 'pos': (1, 2), 'facing': 'East', 'inventory': 'Key' }, 'Key': 'loc=None', 'Door': 'loc=(2, 2), dist=1, dir=Front <REACHABLE>, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Front-Right' }"
    run_test("TEST CASE 3: Ready to Unlock", obs_3)

    # CASE 4: Holding Key but Wrong Direction
    # Scenario: Agent has Key at (1,2) but is facing West (Wall) instead of East (Door).
    # Logic: Door is NOT <REACHABLE> (dir=Behind). Reward should be low (0.1).
    obs_4 = "{ 'Agent': { 'pos': (1, 2), 'facing': 'West', 'inventory': 'Key' }, 'Key': 'loc=None', 'Door': 'loc=(2, 2), dist=1, dir=Behind, state=Locked', 'Goal': 'loc=(3, 3), dist=3, dir=Behind' }"
    run_test("TEST CASE 4: Wrong Facing", obs_4)

    # CASE 5: Phase 3 Transition - Door is Open
    # Scenario: Door is Open. Agent is walking through.
    # Logic: Door state='Open' -> Phase 2 Complete. Reward should be 0.7.
    obs_5 = "{ 'Agent': { 'pos': (2, 2), 'facing': 'East', 'inventory': 'Key' }, 'Key': 'loc=None', 'Door': 'loc=(2, 2), dist=0, dir=Here, state=Open', 'Goal': 'loc=(3, 3), dist=2, dir=Front-Right' }"
    run_test("TEST CASE 5: Door Open", obs_5)

    # CASE 6: Goal Reached
    # Scenario: Agent overlaps with Goal.
    # Logic: Dist to Goal is 0. Reward should be 1.0.
    obs_6 = "{ 'Agent': { 'pos': (3, 3), 'facing': 'South', 'inventory': 'Key' }, 'Key': 'loc=None', 'Door': 'loc=(2, 2), dist=2, dir=Behind, state=Open', 'Goal': 'loc=(3, 3), dist=0, dir=Here <REACHABLE>' }"
    run_test("TEST CASE 6: Goal Reached", obs_6)