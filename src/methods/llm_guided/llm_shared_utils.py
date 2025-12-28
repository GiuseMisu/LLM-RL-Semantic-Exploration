import os
import re
import json
import sys
import time
from abc import ABC, abstractmethod

# --- 1. THE SYSTEM PROMPT ---
# This acts as the "Rulebook" for the LLM. 
# It converts the LLM into a Reward Function.

SYSTEM_PROMPT = """
You are an expert Reward Function for a Reinforcement Learning agent in the MiniGrid-DoorKey environment.
Your goal is to guide the agent towards the solution by providing a SCALAR REWARD (between -0.1 and 1.0).

THE TASK:
1. Locate the Key.
2. Pick up the Key.
3. Locate the Door.
4. Unlock/Open the Door.
5. Reach the Goal.

CRITICAL COORDINATE RULES:
- The Grid Origin (0,0) is TOP-LEFT.
- X increases to the Right (East).
- Y increases DOWNWARDS (South). 
- "dir" in the input gives you the correct direction relative to the agent. Trust it.

INTERACTION RULES:
To PICKUP an object or TOGGLE/OPEN a door, you must be in the adjacent cell facing it (dir=Front). 
You cannot interact with objects to your Left, Right, or Behind.

INPUT FORMAT:
You will receive a structured JSON description.

OUTPUT FORMAT:
Output exactly ONE JSON object representing the IMMEDIATE current state only.
Do not simulate future steps. Do not output multiple JSONs.
Do not include comments (//) inside the JSON object.

### EXAMPLE INPUT:
"{ 'Agent': { 'pos': (1, 1), 'facing': 'East', 'inventory': 'None' }, 'Key': 'loc=(2, 1), dist=1, dir=Front <REACHABLE>' }"

### EXAMPLE OUTPUT:
{
  "check_inventory": "None",
  "check_facing": "Key is Front",
  "reasoning": "The agent sees the key immediately in front (reachable). Reward is max for Phase 1.",
  "reward": 0.5
}

SCORING GUIDELINES:

PHASE 1: FINDING THE KEY (If Inventory is 'None')
- 0.1: Wandering, not seeing the Key.
- 0.3: The Key is visible (in 'Key' field) but not close.
- 0.5: The Key is marked <REACHABLE>. (IMMEDIATE REWARD).
- -0.1: Moving away from the Key or focusing on the Door while Inventory is None.

PHASE 2: OPENING THE DOOR (If Inventory has 'Key')
- 0.1: Wandering with Key.
- 0.5: Standing adjacent to the Door (Door is <REACHABLE>).
- 0.7: The Door is Open/Unlocked.

PHASE 3: GOAL
- 1.0: Reached the Goal.
"""

def clean_json_text(text):
    """
    Extracts ONLY the first JSON object found in the text.
    Stops immediately after the first closing brace '}'.
    """
    # This regex finds the first occurrence of { ... } 
    # .*? is non-greedy, meaning it stops at the first } it sees.
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        return match.group(0)
    # If no JSON found, return original text (will likely fail json.loads)
    return text

class BaseLLMClient(ABC):
    """
    Abstract Base Class for all LLM clients
    Ensures that different LLMs all look the same to the RL agent.
    """    
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        """
        Subclasses must implement this. 
        It should return the raw string from the API.
        """
        pass

    def get_reward(self, observation: str, verbose: bool = False, generate_explanation: bool = False) -> float:
        """
        The MAIN method. It handles the full pipeline:
        Fetch Raw -> Repair JSON -> Clean -> Parse -> Print -> Return Float
        Args:
            observation (str): The observation string
            generate_explanation (bool): If False, stops generation at '}' to save speed/tokens.
        """

        if verbose:
            print(f"\nScanning State: {observation}")
            #start_time = time.time() #[timer]

        try:
            # 1. Call the specific API (Gemini, OpenAI, etc.)
            raw_text = self._get_raw_response(observation, generate_explanation)
            
            # 2. Repair JSON if cut off
            if not generate_explanation and raw_text:
                # If used the stop token '}', the model stops writing BEFORE 
                # sending it (or right at it), so we often need to add it back.
                stripped = raw_text.strip()
                if not stripped.endswith("}"):
                    raw_text = stripped + "}"

            # 3. Regex Clean
            cleaned_text = clean_json_text(raw_text)

            # 4. Parse JSON
            data = json.loads(cleaned_text)

            # 5. Standardized Printing
            if verbose:
                print(f"\n[Raw LLM Output]:\n{cleaned_text}")
                print("-" * 40)
                print(f"CHECK INV: {data.get('check_inventory')}")
                print(f"REWARD:    {data.get('reward')}")
                if generate_explanation:
                    print(f"REASON:    {data.get('reasoning')}")
                print("-" * 40)
                sys.stdout.flush() # FORCE PRINT TO TERMINAL => AVOID BUFFERING ISSUES
                #[timer]
                # end_time = time.time()
                # print(f"Execution time: {end_time - start_time:.4f} seconds")
            
            return float(data.get('reward', 0.0))

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[ERROR] JSON Parsing Failed: {e}")
            print(f"[Raw Text was]: {raw_text}")
            return 0.0
        except Exception as e:
            print(f"[ERROR] General Failure: {e}")
            return 0.0