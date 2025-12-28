# TEXTUALIZER WITH
# COMPACT GLOBAL DESCRIPTION INCLUDING:
# - DISTANCES 
# - RELATIVE DIRECTIONS


import os
import time
import sys
import warnings

# --- 1. SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Mappings
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
DIRECTION_TO_TEXT = {0: "East", 1: "South", 2: "West", 3: "North"}
ACTION_TO_TEXT = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}

def sanitize(value):
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, tuple):
        return tuple(sanitize(x) for x in value)
    return value

# --- NEW HELPER: CALCULATE DIRECTION ---
def get_relative_direction(agent_pos, agent_dir, target_pos):
    """
    Returns: 'Front', 'Left', 'Right', or 'Behind'
    """
    # 1. Get vector to target
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    # 2. Get Agent's forward vector (MiniGrid: 0=East, 1=South, 2=West, 3=North)
    # Be careful with Y! In MiniGrid, South is +Y.
    forward_map = {
        0: (1, 0),  # East
        1: (0, 1),  # South
        2: (-1, 0), # West
        3: (0, -1)  # North
    }
    fx, fy = forward_map[agent_dir]
    
    # 3. Dot product determines Front/Behind
    dot = dx*fx + dy*fy
    
    # 4. Cross product (2D) determines Left/Right
    # cross = fx*dy - fy*dx
    cross = fx*dy - fy*dx
    
    if dot > 0 and abs(cross) <= abs(dot): return "Front"
    if dot < 0 and abs(cross) <= abs(dot): return "Behind"
    if cross > 0: return "Right" # Check MiniGrid coordinate handedness!
    return "Left"

# --- UPDATED COMPACT TEXTUALIZER ---
def get_compact_global_description(env, include_distances=True):
    base_env = env.unwrapped
    agent_pos = sanitize(base_env.agent_pos)
    agent_dir = base_env.agent_dir
    
    # Get text direction
    if agent_dir in DIRECTION_TO_TEXT:
        facing = DIRECTION_TO_TEXT[agent_dir]
    else:
        facing = "Unknown"
        
    carrying = base_env.carrying
    
    key_info = "Not Found"
    door_info = "Not Found"
    goal_info = "Unknown"
    
    if carrying and carrying.type == 'key':
        key_info = "In Inventory (Carried)"

    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            cell = base_env.grid.get(x, y)
            
            if cell:
                target_pos = (int(x), int(y))
                dist = int(abs(x - agent_pos[0]) + abs(y - agent_pos[1]))
                
                # --- NEW: Compute Direction explicitly ---
                # We give the LLM the answer directly: "dir=South"
                # Pass the integer agent_dir found earlier in the function
                relative_dir = get_relative_direction(agent_pos, agent_dir, target_pos)
                
                # --- NEW: Check Reachability ---
                # "Reachable" means Distance is 1 AND Direction is Front
                is_reachable = (dist == 1 and relative_dir == "Front")
                reach_str = " <REACHABLE>" if is_reachable else ""

                # Build the info string
                if include_distances:
                    # Enrich: "loc=(1,2), dist=1, dir=South"
                    info_str = f"loc={target_pos}, dist={dist}, dir={relative_dir}{reach_str}"
                else:
                    # Even in ablation, knowing direction is fair game if using coordinates
                    info_str = f"loc={target_pos}, dir={relative_dir}{reach_str}"
                
                if cell.type == "key":
                    key_info = info_str
                elif cell.type == "door":
                    if cell.is_locked: state = "Locked"
                    elif cell.is_open: state = "Open"
                    else: state = "Closed"
                    door_info = f"{info_str}, state={state}"
                elif cell.type == "goal":
                    goal_info = info_str
                    
    if carrying:
        inventory = f"{carrying.color} {carrying.type}"
    else:
        inventory = "None"
    
    return (
        f"{{ 'Agent': {{ 'pos': {agent_pos}, 'facing': '{facing}', 'inventory': '{inventory}' }}, "
        f"'Key': '{key_info}', "
        f"'Door': '{door_info}', "
        f"'Goal': '{goal_info}' }}"
    )

# --- VISUALIZATION TEST BLOCK ---
if __name__ == "__main__":
    
    # go up two levels to import from src/common/env_setup.py
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
    from src.common.env_setup import make_minigrid_env
    
    env = make_minigrid_env(env_id="MiniGrid-DoorKey-5x5-v0", render_mode="human")()
    #make_minigrid_env already applies ImgObsWrapper => return the WRAPPED ENV
    env.reset()

    # Initial State
    print(f"\n[Start] Initial State:")
    print(get_compact_global_description(env))
    
    time.sleep(2.5)

    for i in range(1, 10):
        
        action = env.action_space.sample()
            
        action_name = ACTION_TO_TEXT.get(action, "unknown")
        
        # Execute
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Generate Compact Description
        desc = get_compact_global_description(env, include_distances=True)
        #descVerbose = get_enhanced_global_description(env)
        
        print("-" * 50)
        print(f"Step {i} | Action: {action} ({action_name.upper()})")
        print(f"LLM Input: {desc}")
        
        # FORCE PRINT TO TERMINAL => AVOID BUFFERING ISSUES
        sys.stdout.flush() 
        
        # Pause for visualization
        time.sleep(2.5) 
            
    env.close()