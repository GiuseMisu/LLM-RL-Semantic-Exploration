import os
import shutil


def move_experiment_files(model_name, environment, files_to_move=None):
    """
    Move experiment output files (GIF, CSV, model) to appropriate dir
    
    Args:
        model_name: Name of the model (e.g., "PPO", "RecurrentPPO_lstm", "RNDPPO")
        environment: Environment name (e.g., "EMPTY_5x5", "DOORKEY_5x5", "EMPTY_8x8", "DOORKEY_8x8")
        files_to_move: Optional dict with custom file mappings. If None, uses default patterns.
                       Format: {'gif': (src, dst), 'csv': (src, dst), 'model': (src, dst)}
    
    The function will:
    - Move GIF files to ../../results/visualizations/{env}/
    - Move CSV files to ../../logs/{model}_{env}/
    - Move model files to ../../results/models/{env}/
    """
    
    # Extract environment directory name (e.g., "empty_5x5" from "EMPTY_5x5")
    env_dir = environment.lower()
    
    if files_to_move is None:
        
        files_to_move = {
            'gif': (
                f"./{model_name}_{environment}_eval.gif",
                f"../../results/visualizations/{env_dir}/{model_name}_{environment}_eval.gif"
            ),
            'csv': (
                f"./{model_name}_{environment}_evaluation.csv",
                f"../../logs/{model_name}_{environment}/{model_name}_{environment}_evaluation.csv"
            ),
            'model': (
                f"./{model_name}_{environment}_best.pkl",
                f"../../results/models/{env_dir}/{model_name}_{environment}_best.pkl"
            )
        }
    
    # Move each file
    for file_type, (src, dst) in files_to_move.items():
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"{file_type.upper()} saved to: {dst}")
        else:
            print(f"Warning: {file_type.upper()} file not found at {src}")
