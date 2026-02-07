import os
import sys
import ollama

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from src.methods.llm_guided.llm_clients.base_client import BaseLLMClient
from src.methods.llm_guided.ScalarApproach.scalar_prompts import DOOR_KEY_SYSTEM_PROMPT, EMPTY_SYSTEM_PROMPT

class GPT_OSS_Client(BaseLLMClient):     
    def __init__(self, model_name="gpt-oss:120b-cloud", system_prompt=DOOR_KEY_SYSTEM_PROMPT, reasoning = True, temperature=0.3):     
        super().__init__(system_prompt=system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning = reasoning        

        # Verify connection on init
        try:
            ollama.show(self.model_name)
        except Exception:
            print(f"Warning: Model {self.model_name} not found locally (Run 'ollama pull {self.model_name}')")

    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        try:
            
            llm_options = {
                           'temperature': self.temperature,
                           "top_p": 0.9                     
                          }
            
            messages = [
                {'role': 'system', 'content': self.system_prompt}, 
                {'role': 'user', 'content': f"Observation: {prompt}"} 
            ]

            if self.reasoning:
                response = ollama.chat(
                    model=self.model_name,
                    options=llm_options,
                    messages=messages, 
                    think="high" # Enable CoT through string not boolean
                )
            else:
                response = ollama.chat(
                    model=self.model_name,
                    options=llm_options,
                    messages=messages
                )

            #print("THE RAW RESPONSE FROM GPT IS AS SUCH:")
            #print(response)

            if 'message' in response and 'content' in response['message']:
                raw_text = response['message']['content']
                
                # if self.reasoning == True and "thinking" in response["message"] and response["message"]["thinking"]:
                #     # [debug] print CoT
                #     print("\n" + "-"*40)
                #     print("CoT:")
                #     print(response['message']['thinking'])
                #     print("="*40 + "\n")

                return raw_text
            
            return ""

        except Exception as e:
            raise ConnectionError(
                f"[GPT Error] Could not connect to Ollama\n"
                f"Make sure Ollama is running"
            )
    