import os
import sys
import re
from openai import OpenAI  # Changed from 'ollama' to 'openai'

# Adjust this path if necessary to match your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.methods.llm_guided.llm_shared_utils import BaseLLMClient, DOOR_KEY_SYSTEM_PROMPT

class PerplexityClient(BaseLLMClient):
    
    def __init__(self, model_name="sonar-reasoning-pro", system_prompt=DOOR_KEY_SYSTEM_PROMPT, temperature=0.6):     
        super().__init__(system_prompt=system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the OpenAI client pointing to Perplexity
        # It looks for PERPLEXITY_API_KEY in your environment variables automatically.
        # If you MUST hardcode it (not recommended), use: api_key="pplx-..."
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            print("Warning: PERPLEXITY_API_KEY not found in environment variables.")
        
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.perplexity.ai"
        )
        
        print(f"\n\n[Perplexity API] \n- Model = {self.model_name} \n- Temperature = {self.temperature}")

    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        try:
            # Perplexity models:
            # - 'sonar-reasoning-pro': Best for logic (Simulates DeepSeek R1 / Reasoning)
            # - 'sonar-pro': Faster, standard search/chat
            
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            
            # API Call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

            # Extract content
            if response.choices and response.choices[0].message.content:
                raw_text = response.choices[0].message.content

                # Perplexity's 'sonar-reasoning' models might include <think> tags 
                # similar to DeepSeek. We keep your cleaning logic just in case.
                cleaned_response = self.clean_json_text_DeepSeek(raw_text)
                return cleaned_response
            
            return ""

        except Exception as e:
            raise ConnectionError(
                f"[Perplexity API Error] {e}\n"
                f"Check your API key and credit balance."
            )
    
    @staticmethod
    def clean_json_text_DeepSeek(text: str) -> str:
        # Kept your existing cleaning logic
        # 1. Remove <think>...</think> blocks if present (Perplexity reasoning models use this too)
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 2. Existing logic to find the first '{' and last '}'
        # This helps if the model chats before giving the JSON
        start = clean_text.find('{')
        end = clean_text.rfind('}')
        
        if start != -1 and end != -1:
            return clean_text[start:end+1]
        
        return clean_text
    
