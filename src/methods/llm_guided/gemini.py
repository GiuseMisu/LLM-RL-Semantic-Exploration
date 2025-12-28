import os
import sys  
import json
import re


from google import genai
from google.genai import types

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.methods.llm_guided.llm_shared_utils import BaseLLMClient, SYSTEM_PROMPT


class GeminiLLMClient(BaseLLMClient):
    def __init__(self, api_key=None, model_name="gemini-1.5-flash", debug=False):
        
        super().__init__(debug=debug)
        
        # Authentication
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key is missing!")
            
        # Client Setup (New SDK Style)
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

        #check system prompt
        if not SYSTEM_PROMPT:
            raise ValueError("SYSTEM_PROMPT is not set properly")
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE")
        ]

    def _get_raw_response(self, prompt: str, generate_explanation: bool = False) -> str:
        """
        Implementation of the abstract method.
        Just sends the request and returns the string.
        """
        try:
            #build the config
            run_config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1, 
                safety_settings=self.safety_settings,
                # =====[CRITICAL] Stop sequences=====
                stop_sequences=["}"] if not generate_explanation else None
            )

            # Send message to the model
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=run_config
            )
            
            if response.text:
                return response.text
            return ""

        except Exception as e:
            if self.debug:
                print(f"Gemini API Error: {e}")
            return ""


if __name__ == "__main__":    

    #Setup Test Client
    try:
        MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
        client = GeminiLLMClient(debug=True, model_name=MODEL_NAME)
        print(f"--- TESTING MODEL: {client.model_name} ---")
    except Exception as e:
        print(e)
