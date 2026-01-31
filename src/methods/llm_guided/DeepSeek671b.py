import os
import sys
import ollama
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.methods.llm_guided.llm_shared_utils import BaseLLMClient, DOOR_KEY_SYSTEM_PROMPT

class DeepSeekCloud671b_Client(BaseLLMClient):
    
    def __init__(self, model_name="deepseek-v3.1:671b-cloud", system_prompt=DOOR_KEY_SYSTEM_PROMPT, reasoning = True, temperature=0.3):     
        super().__init__(system_prompt=system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning = reasoning
        print(f"\n\n[DeepSeek-Cloud-671b] \n- Reasoning = {self.reasoning} \n- Temperature = {self.temperature}")

        # Verify connection on init
        try:
            ollama.show(self.model_name)
        except Exception:
            print(f"Warning: Model {self.model_name} not found locally (Run 'ollama pull {self.model_name}')")

    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        try:
            
            llm_options = {'temperature': self.temperature}
            
            messages = [
                {'role': 'system', 'content': self.system_prompt}, 
                {'role': 'user', 'content': f"Observation: {prompt}"} 
            ]

            response = ollama.chat(
                model=self.model_name,
                options=llm_options,
                messages=messages, 
                think=self.reasoning
            )
            
            #print("THE RAW RESPONSE FROM DEEPSEEK IS AS SUCH:")
            #print(response)

            if 'message' in response and 'content' in response['message']:
                raw_text = response['message']['content']

                # if self.reasoning == True and response['message']['thinking']:
                #     # [debug] print CoT
                #     print("\n" + "-"*40)
                #     print("CoT of DeepSeek-Cloud-671b:")
                #     print(response['message']['thinking'])
                #     print("="*40 + "\n")

                # clean the specific DeepSeek artifacts here
                cleaned_response = self.clean_json_text_DeepSeek(raw_text)
                return cleaned_response
            
            return ""

        except Exception as e:
            raise ConnectionError(
                f"[DeepSeek-Cloud-671b Error] Could not connect to Ollama\n"
                f"Make sure Ollama is running"
            )
    
    @staticmethod
    #==============================
    #probably can be removed
    #=============================
    def clean_json_text_DeepSeek(text: str) -> str:
        # 1. Remove <think>...</think> blocks if present
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 2. Existing logic to find the first '{' and last '}'
        start = clean_text.find('{')
        end = clean_text.rfind('}')
        
        if start != -1 and end != -1:
            return clean_text[start:end+1]
        return clean_text