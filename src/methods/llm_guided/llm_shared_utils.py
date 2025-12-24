import re
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
- Therefore: (1, 2) is SOUTH of (1, 1). Do not confuse this with standard graphs.
- "dir" in the input gives you the correct direction relative to the agent. Trust it.

INPUT FORMAT:
You will receive a structured JSON description.
Example: "Key": "loc=(1, 2), dist=1, dir=South"

OUTPUT FORMAT:
You must strictly follow this JSON format:
{
  "reasoning": "A short sentence explaining why you gave this score.",
  "reward": <float between -0.1 and 1.0>
}

SCORING GUIDELINES:
- 0.0: Useless move (hitting a wall, spinning in place).
- 0.1: Exploring new tiles / moving towards an unseen area.
- 0.3: Moving closer to a visible Key.
- 0.5: PICKING UP the Key (Major sub-goal).
- 0.7: Moving towards the Door (while holding the Key).
- 1.0: OPENING the Door or Reaching the Goal.
- -0.1: Repeating the same state (getting stuck).

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