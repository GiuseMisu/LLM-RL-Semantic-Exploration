import re
import json
import sys
from abc import ABC, abstractmethod


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