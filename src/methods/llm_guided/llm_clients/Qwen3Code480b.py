import ollama
from src.methods.llm_guided.llm_clients.base_client import BaseLLMClient

class Qwen3CoderClient(BaseLLMClient):
    def __init__(
        self,
        model_name="qwen3-coder:480b-cloud",
        system_prompt="",
        temperature=0.1,
        top_p=0.9
    ):
        super().__init__(system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

        try:
            ollama.show(self.model_name)
        except Exception:
            print(f"[WARN] Model {self.model_name} not found locally")

    def _get_raw_response(self, prompt: str, generate_explanation: bool) -> str:
        # Qwen needs strong constraints
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "think": True, 
        }

        messages = [
            {
                "role": "system",
                "content": self.system_prompt + "\nRespond ONLY with the Python function. No explanations, no markdown.",
            },
            {
                "role": "user",
                "content": f"Observation: {prompt}",
            },
        ]

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            think=True
        )

        print("THE RAW RESPONSE FROM QWEN-3-CODER IS AS SUCH:")
        print(response["message"]["content"])
        return response["message"]["content"]
