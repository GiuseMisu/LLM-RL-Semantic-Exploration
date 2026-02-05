import os
import shutil
import glob


def move_experiment_files(model_name, environment, source_pattern=None, include_reward_function=False):
    """
    file moving utility
    
    Args:
        model_name: Name of the model for destination files (e.g., "PPO", "Eureka_Qwen")
        environment: Environment name (e.g., "EMPTY_5x5", "DOORKEY_5x5", "EMPTY_8x8", "DOORKEY_8x8")
        source_pattern: Optional custom pattern for source files. If None, uses model_name.
                       For Eureka, use "PPO_FINAL__" to match Eureka's output files.
        include_reward_function: If True, also looks for and moves reward function .py files
    
    Automatically moves:
    - GIF files to ../../results/visualizations/{env}/
    - CSV files to ../../logs/{model}_{env}/
    - Model .pkl files to ../../results/models/{env}/
    - Reward functions (if enabled) to ../../results/reward_functions/{env}/
    """
    
    # Extract environment directory name (e.g., "empty_5x5" from "EMPTY_5x5")
    env_dir = environment.lower()
    
    # Determine source file pattern
    if source_pattern is None:
        src_pattern = model_name
    else:
        src_pattern = source_pattern
    
    # Define file mappings with smart pattern matching
    files_to_check = []
    
    # Standard experiment outputs
    files_to_check.extend([
        # GIF evaluation videos
        (f"./{src_pattern}*{environment}_eval.gif",
         f"../../results/visualizations/{env_dir}/{model_name}_{environment}_eval.gif",
         "GIF"),
        # CSV evaluation logs  
        (f"./{src_pattern}*{environment}_evaluation.csv",
         f"../../logs/{model_name}_{environment}/{model_name}_{environment}_evaluation.csv",
         "CSV"),
        # Model checkpoint files
        (f"./{src_pattern}*{environment}_best.pkl",
         f"../../results/models/{env_dir}/{model_name}_{environment}_best.pkl",
         "MODEL"),
    ])
    
    # Reward function files (for Eureka approach)
    if include_reward_function:
        # Extract environment name parts (e.g., "DoorKey8x8" from "DOORKEY_8x8")
        env_parts = environment.split('_')
        env_name = env_parts[0].capitalize() + env_parts[1]  # DoorKey8x8
        
        # Look for BestRwdFunc files
        reward_pattern = f"./BestRwdFunc_{env_name}_*.py"
        files_to_check.append(
            (reward_pattern,
             f"../../results/reward_functions/{env_dir}/{model_name}_{environment}_reward.py",
             "REWARD_FUNCTION")
        )
    
    # Move each file type
    moved_count = 0
    for src_glob, dst, file_type in files_to_check:
        # Find matching files
        matching_files = glob.glob(src_glob)
        
        if matching_files:
            # Use the first match (should only be one)
            src = matching_files[0]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"{file_type} saved to: {dst}")
            moved_count += 1
        else:
            # Only warn if it's a standard file (not reward function)
            if not include_reward_function or file_type != "REWARD_FUNCTION":
                print(f"Note: No {file_type} file found matching {src_glob}")
    
    if moved_count == 0:
        print(f"Warning: No experiment files found to move for {model_name}_{environment}")
    
    return moved_count
