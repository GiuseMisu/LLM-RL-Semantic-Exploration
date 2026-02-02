import os
import sys
import ollama

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from src.methods.llm_guided.llm_clients.base_client import BaseLLMClient
from src.methods.llm_guided.ScalarApproach.scalar_prompts import DOOR_KEY_SYSTEM_PROMPT, EMPTY_SYSTEM_PROMPT

class HermesLLMClient(BaseLLMClient):
    def __init__(self, model_name="hermes3:8b", debug=False, system_prompt=DOOR_KEY_SYSTEM_PROMPT):     
        """
        Args:
            model_name (str): Name of the Ollama model tag (default: 'deepseek-r1')
            debug (bool): Enable verbose logging
        """
        super().__init__(debug=debug, system_prompt=system_prompt)
        self.model_name = model_name
        
        #no api key needed cause run locally

        # Check if SYSTEM_PROMPT is loaded
        if not system_prompt:
            raise ValueError("SYSTEM_PROMPT is not set properly in llm_shared_utils.")

    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        """
        Implementation of the abstract method
        """
        try:
            # 1. Configure Options
            llm_options = {'temperature': 0.1}
            
            # Stop optimization: Cut off generation at '}' if we don't need the essay
            if not generate_explanation:
                llm_options['stop'] = ['}']

            # 2. Call Ollama 
            response = ollama.chat(
                model=self.model_name,
                options=llm_options,
                messages=[
                    {'role': 'system', 'content': self.system_prompt}, 
                    {'role': 'user', 'content': f"Observation: {prompt}"} 
                ]
            )
            
            # 3. Extract Content
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            
            return ""

        except Exception as e:
            if self.debug:
                print(f"[Error] Could not connect to Ollama: {e}")
                print("Make sure Ollama is running ('ollama serve') and the model is pulled")
            return ""

# Simple check if run directly
if __name__ == "__main__":
    try:
        client = HermesLLMClient(debug=True)
        print(f"DeepSeek Client Initialized. Model: {client.model_name}")
    except Exception as e:
        print(e)