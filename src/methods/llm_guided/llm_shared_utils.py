import os
import re
import json
import sys
import time
from abc import ABC, abstractmethod

DOOR_KEY_SYSTEM_PROMPT = """
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
- "dir" in the input gives you the correct direction relative to the agent (Front, Left, Right, Behind).

INTERACTION RULES:
To PICKUP an object or TOGGLE/OPEN a door, you must be in the adjacent cell facing it (dir=Front). 
You cannot interact with objects to your Left, Right, or Behind.

INPUT FORMAT:
You will receive a structured JSON description of the current environment state.

OUTPUT FORMAT:
Output exactly A SINGLE JSON object, with four fields:
- "check_inventory": The current inventory of the agent (e.g., 'Key' or 'None').
- "check_facing": Where the agent is currently facing (e.g. North, East, South, West).
- "reasoning": A brief explanation (1-2 sentences) of your reward decision.
- "reward": A scalar float value between -0.1 and 1.0 representing the reward for this state.

Do not output multiple JSONs.
Do not simulate future steps. 
IMPORTANT: Do not include comments (//) inside the JSON object.

### EXAMPLE INPUT:
{ 'Agent': { 'pos': (1, 1), 'facing': 'West', 'inventory': 'None' }, 'Key': 'loc=(1, 3), dist=2, dir=Left', 'Door': 'loc=(2, 2), dist=2, dir=Behind, state=Locked', 'Goal': 'loc=(3, 3), dist=4, dir=Behind' }

### EXAMPLE OUTPUT:
{
  "check_inventory": "None",
  "check_facing": "Key is Left",
  "reasoning": "The agent has no key in inventory. The key is visible but not adjacent (dist=2), so it cannot be picked up. The goal and door are not reachable, and the agent is not focusing on the door.",
  "reward": 0.3
}

SCORING GUIDELINES (EVALUATE IN THIS ORDER):

1. PHASE 3: GOAL (HIGHEST PRIORITY)
   - CHECK THIS FIRST. Irrespective of inventory.
   - 1.0: If 'Goal' is marked dist=0 (Here).
   - 0.9: If 'Goal' is adjacent (dir=Front <REACHABLE>).

2. PHASE 2: OPENING THE DOOR
   - Condition: Inventory has 'Key' AND Goal is NOT reachable.
   - 0.7: The Door is Open/Unlocked (state=Unlocked or state=Open).
   - 0.5: Standing adjacent to the Door (dir=Front <REACHABLE>) but it is not opened (state=Locked).
   - 0.1: Wandering with Key.

3. PHASE 1: FINDING THE KEY
   - Condition: Inventory is 'None'.
   - 0.5: The Key is marked <REACHABLE>. (IMMEDIATE REWARD).
   - 0.3: The Key is visible (in 'Key' field) but not close.
   - 0.1: Wandering, not seeing the Key.
   - -0.1: Moving away from the Key when it is visible.
"""

EMPTY_SYSTEM_PROMPT = """
You are an expert Reward Function for a Reinforcement Learning agent in the MiniGrid-Empty environment.
Your goal is to guide the agent towards the solution by providing a SCALAR REWARD (between -0.1 and 1.0).

THE TASK:
1. Locate the Goal.
2. Move towards the Goal.
3. Reach the Goal.

COORDINATE RULES:
- The Grid Origin (0,0) is TOP-LEFT.
- X increases to the Right (East).
- Y increases DOWNWARDS (South).
- "dir" in the input gives you the correct direction relative to the agent (Front, Left, Right, Behind).

INPUT FORMAT:
You will receive a structured JSON description of the current environment state.

OUTPUT FORMAT:
Output exactly A SINGLE JSON object, with two fields:
- "reasoning": A brief explanation (1-2 sentences) of your reward decision.
- "reward": A scalar float value between -0.1 and 1.0 representing the reward for this state.

Do not output multiple JSONs.
Do not simulate future steps.
IMPORTANT: Do not include comments (//) inside the JSON object.

### EXAMPLE INPUT:
"{ 'Agent': { 'pos': (1, 1), 'facing': 'East' }, 'Goal': 'loc=(3, 1), dist=2, dir=Front' }"
        
### EXAMPLE OUTPUT:
{
  "reasoning": "The Goal is directly in front. Moving forward reduces distance.",
  "reward": 0.5
}
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

def strip_json_comments(text_json):
    """
    Removes comments (// ...) from a JSON string 
    so standard json.loads can parse it
    """
    # Regex to remove // comments until end of line
    # It looks for //, then any character (.) until a newline or end of string
    text_json = re.sub(r'//.*', '', text_json)
    return text_json

class BaseLLMClient(ABC):
    """
    Abstract Base Class for all LLM clients
    Ensures that different LLMs all look the same to the RL agent.
    """    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

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

        try:
            # 1. Call the specific API 
            raw_text = self._get_raw_response(observation, generate_explanation)
            
            # 2. Regex Clean
            # Remove // comments that confuse json.loads
            cleaned_text = strip_json_comments(raw_text)
            #Extract the JSON object part
            cleaned_text = clean_json_text(cleaned_text)
            cleaned_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', cleaned_text)  # Remove non-ASCII chars

            # 3. Repair JSON if cut off (Check for missing brace)
            if not generate_explanation and cleaned_text:
                # If used the stop token '}', the model stops writing BEFORE 
                # sending it (or right at it), so we often need to add it back.
                stripped = cleaned_text.strip()
                if not stripped.endswith("}"):
                    cleaned_text = stripped + "}"

            # 4. Parse JSON
            data = json.loads(cleaned_text)

            # 5. Standardized Printing
            if verbose:                
                #[FOR DEBUG] raw and clean response
                # print(f"\n[Raw LLM Output]:\n{raw_text}") 
                # print(f"\n[Cleaned LLM Output]:\n{cleaned_text}")
                print("-" * 40)
                # the two envs have different keys
                # iterate over keys to handle both DoorKey (check_inventory) and Empty (reasoning)
                priority_keys = ['check_inventory', 'check_facing', 'reasoning', 'reward']
                for key in priority_keys:
                    if key in data:
                        # Print formatted: Key ...... Value
                        print(f"{key.upper().ljust(15)}: {data[key]}") 
                print("-" * 40)
                sys.stdout.flush()
            
            if 'reward' not in data:
                print(f"[ERROR] 'reward' field missing in LLM response.")
                print(f"[Raw Text was]: {repr(raw_text)}")
                print(f"[Attempted to Parse]: {repr(cleaned_text)}")
                return 0.0
            else:
                return float(data.get('reward'))

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[ERROR] JSON Parsing Failed: {e}")
            print(f"[Raw Text was]: {repr(raw_text)}")
            print(f"[Attempted to Parse]: {repr(cleaned_text)}")
            return 0.0
        
        except ConnectionError:
            # This ensures the program crashes immediately if Ollama is down
            print("\n[Error] Could not connect to Ollama\n")
            raise 

        except Exception as e:
            print(f"[ERROR] General Failure: {e}")
            return 0.0