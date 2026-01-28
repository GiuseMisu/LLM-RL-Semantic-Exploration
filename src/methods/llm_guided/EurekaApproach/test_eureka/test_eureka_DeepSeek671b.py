import os
import warnings
# ---  SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../"))

from src.methods.llm_guided.DeepSeek671b import DeepSeekCloud671b_Client
from src.methods.llm_guided.EurekaApproach.eureka_search import EurekaSearch

if __name__ == "__main__":

    try:
        # Initialize the client
        llm = DeepSeekCloud671b_Client(
            reasoning=True, 
            temperature=0.3        
            )
        print(f"Model: {llm.model_name}")
    except Exception as e:
        print(f"Setup Failed: {e}")
        sys.exit(1)

    # Run Search
    eureka = EurekaSearch(
        env_id="MiniGrid-DoorKey-5x5-v0",
        llm_model=llm,
        reflection_iterations=3, # numero di volte che provi a miglioreare la reward heuristic returned by llm
        
        evaluation_epochs=30, 
        evaluation_max_steps=250, 
        pure_rl_baseline='PPO' 
    )
    eureka.run()