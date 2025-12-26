import re
from abc import ABC, abstractmethod

# --- 1. THE SYSTEM PROMPT ---
# This acts as the "Rulebook" for the LLM. 
# It converts the LLM into a Reward Function.

#inserted chackinv and check facing in prompt to force LLM to "check" the state before reasoning
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
- Therefore: (1, 2) is SOUTH of (1, 1). Do not confuse this with standard graphs.
- "dir" in the input gives you the correct direction relative to the agent. Trust it.

INTERACTION RULES:
To PICKUP an object or TOGGLE/OPEN a door, you must be in the adjacent cell facing it (dir=Front). 
You cannot interact with objects to your Left, Right, or Behind.
Consider the interaction rules when reasoning about rewards.

INPUT FORMAT:
You will receive a structured JSON description.
Example: "Key": "loc=(1, 2), dist=1, dir=South"

OUTPUT FORMAT:
You must strictly follow this JSON format:
{
  "check_inventory": "What is the agent holding?",
  "check_facing": "Is the target In Front or Behind?",
  "reasoning": "Based on the checks above, explain the score.",
  "reward": <float between -0.1 and 1.0>
}

SCORING GUIDELINES:

PHASE 1: FINDING THE KEY (If Inventory is 'None')
- 0.1: Wandering, not seeing the Key.
- 0.3: The Key is visible (in 'Key' field) but not close.
- 0.5: The Key is marked <REACHABLE>. (IMMEDIATE REWARD, STOP HERE).
- -0.1: Moving away from the Key or focusing on the Door while Inventory is None.
- DO NOT CARE ABOUT THE DOOR OR GOAL IN THIS PHASE.

PHASE 2: OPENING THE DOOR (If Inventory has 'Key')
- 0.1: Wandering with Key.
- 0.5: Standing adjacent to the Door (Door is <REACHABLE>).
- 0.7: The Door is Open/Unlocked.

PHASE 3: GOAL
- 1.0: Reached the Goal.

Be generous with intermediate steps (like moving closer to the key) to encourage progress.
"""

class BaseLLMClient(ABC):
    """
    Abstract Base Class for all LLM clients
    Ensures that different LLMs all look the same to the RL agent.
    """    
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def get_llm_response(self, prompt: str) -> str:
        """
        Raw method to send text to the specific LLM API and get text back.
        Must be implemented by the child class (e.g., GeminiClient).
        """
        pass

    def compute_reward(self, state_description: str) -> float:
        """
        The main method called by the RL Agent.
        1. Sends state to LLM.
        2. Parses the response.
        3. Returns the scalar reward.
        """
        try:
            # 1. Call the API
            raw_response = self.get_llm_response(state_description)
            
            # 2. Extract the number
            reward = self._parse_scalar_reward(raw_response)
            
            #=====================
            # --- GUARDRAIL ---
            # in the system promt there is a reward rule: rew > 0.5 only if holding key
            #=====================
            # If LLM gives > 0.5 (implies holding key) but Inventory is None, crush the reward.
            if "'inventory': 'None'" in state_description and reward > 0.4:
                print("[GUARDRAIL] LLM hallucinated holding key => Clamping reward")
                reward = 0.1

            if self.debug:
                print(f"[LLM RAW]: {raw_response}")
                print(f"[LLM PARSED]: {reward}")
                
            return reward

        except Exception as e:
            print(f"Error in LLM Reward Calculation: {e}")
            return 0.0  # Fallback to neutral reward if LLM fails

    def _parse_scalar_reward(self, response_text: str) -> float:
        """
        Robustly extracts a float number from the LLM's answer.
        Handles JSON, raw numbers, or sentences like 'The reward is 0.5'.
        """
        # 1. Try to find the specific JSON field "reward": 0.5
        json_match = re.search(r'"reward"\s*:\s*([-+]?\d*\.?\d+)', response_text)
        if json_match:
            return float(json_match.group(1))

        # 2. Fallback: Find the last floating point number in the text
        print("==> [FALLBACK] Could not find JSON 'reward' field, Trying to extract number from text")
        # (LLMs usually put the final score at the end)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
        if numbers:
            # Return the last number found, assuming it's the score
            return float(numbers[-1])
            
        return 0.0