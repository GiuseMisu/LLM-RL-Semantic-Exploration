import os
import sys  

from google import genai
from google.genai import types

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.methods.llm_guided.llm_shared_utils import BaseLLMClient, SYSTEM_PROMPT

class GeminiLLMClient(BaseLLMClient):
    def __init__(self, api_key=None, model_name="gemini-1.5-flash", debug=False):
        """
        Args:
            api_key (str): Your Google Gemini API Key
            model_name (str): Name of the Gemini model to use
        """
        super().__init__(debug=debug)
        
        # 1. Authentication
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key is missing! Pass it to init or set GEMINI_API_KEY env var.")
            
        # 2. Client Setup (New SDK Style)
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

        #check system prompt
        if not SYSTEM_PROMPT:
            raise ValueError("SYSTEM_PROMPT is not set properly. Please update it in llm_shared_utils.py.")
        
        # 3. Configuration (System Prompt & Safety)
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1, # Low temp for consistent numerical rewards
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                )
            ]
        )

    def get_llm_response(self, prompt: str) -> str:
        """
        Sends the textualized state to Gemini and returns the raw text response, using the new SDK.
        """
        try:
            # Send message to the model
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config
            )
            
            # Extract text
            if response.text:
                return response.text
            else:
                print("==> [ERROR] Gemini returned EMPTY response text")
                return '{"reward": 0.0, "reasoning": "Error: Empty response"}'
                
        except Exception as e:
            if self.debug:
                print(f"Gemini API Error: {e}")
            return '{"reward": 0.0, "reasoning": "API Failure"}'


if __name__ == "__main__":    
    # Mock input
    mock_state = """{ 'Agent': { 'pos': (1, 1), 'facing': 'West', 'inventory': 'None' }, 'Key': 'pos=(1, 2)', 'Door': 'pos=(2, 2), state=Locked', 'Goal_Dist': pos=(3, 3) }"""
    
    try:
        client = GeminiLLMClient(debug=True, model_name="gemini-2.5-flash-lite-preview-09-2025")
        
        print(f"Sending State:\n{mock_state}\n")
        reward = client.compute_reward(mock_state)
        print("-" * 30)
        print(f"FINAL SCALAR REWARD: {reward}")
        print("-" * 30)
        
    except ValueError as e:
        print(f"Setup Error: {e}")
    except Exception as e:
        print(f"Runtime Error: {e}")