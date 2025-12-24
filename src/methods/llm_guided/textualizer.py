import os
import warnings
import time
import sys

# --- SILENCE WARNINGS & PYGAME SPAM ---
# This must run BEFORE importing minigrid/gymnasium
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Reverse mappings
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()} # given an index, get the object name
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}   # given an index, get the color name
DIRECTION_TO_TEXT = {0: "East", 1: "South", 2: "West", 3: "North"} # MiniGrid uses 0=right(E),1=down(S),2=left(W),3=up(N)
ACTION_TO_TEXT = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}


def sanitize(value):
    """
    convert NumPy int64 types to Python ints
     prevents np.int64(1) from appearing in LLM prompts
    """
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, tuple):
        return tuple(sanitize(x) for x in value)
    return value

def get_enhanced_global_description(env, include_distances=True):
    """
    global state VERBOSE description for LLM
    Args:
        env: MiniGrid environment WRAPPED WITH ImgObsWrapper
        include_distances (bool): Whether to include distances to objects in the description.
        neded for ablation studies ==> distances vs raw positions
    """
    #input is env WRAPPED WITH ImgObsWrapper => need to unwrap to access full state
    base_env = env.unwrapped
    
    # Agent state 
    agent_pos = sanitize(base_env.agent_pos) # position of the agent
    agent_dir = base_env.agent_dir # direction the agent is facing
    agent_dir_text = DIRECTION_TO_TEXT[agent_dir] # convert to text the direction
    carrying = base_env.carrying # object the agent is carrying (None if empty)
    
    # grid dim
    width, height = base_env.grid.width, base_env.grid.height
    
    # Object tracking
    objects = {
        'key': None,
        'door': None,
        'goal': None
    }
    
    # Scan grid
    for x in range(width):
        for y in range(height):
            cell = base_env.grid.get(x, y)
            if cell is not None:
                # Calculate Distance (Manhattan)
                # to help the llm reason also with numbers
                dist = int(abs(x - agent_pos[0]) + abs(y - agent_pos[1]))
                obj_pos = (int(x), int(y)) # Sanitize coordinates
                
                if cell.type == "key":
                    objects['key'] = {
                        'pos': obj_pos,
                        'color': cell.color,
                        'distance': dist
                    }
                elif cell.type == "door":
                    if cell.is_locked:
                        door_state = "Locked"
                    elif cell.is_open:
                        door_state = "Open"
                    else:
                        door_state = "Closed"
                        
                    objects['door'] = {
                        'pos': obj_pos,
                        'color': cell.color,
                        'state': door_state,
                        'distance': dist
                    }
                elif cell.type == "goal":
                    objects['goal'] = {
                        'pos': obj_pos,
                        'distance': dist
                    }
    
    # --- BUILD DESCRIPTION ---
    lines = []
    
    # 1. Self Awareness
    lines.append(f"## Agent Status")
    lines.append(f"- Location: {agent_pos}")
    lines.append(f"- Facing: {agent_dir_text}")
    if carrying:
        lines.append(f"- Inventory: Holding {carrying.color} {carrying.type}")
    else:
        lines.append(f"- Inventory: Empty")
    
    # 2. World Awareness
    lines.append(f"\n## Visible Objects")
    
    if objects['key']:
        k = objects['key']
        # Explicitly state distance to help the LLM generate progress rewards
        if include_distances:
            lines.append(f"- {k['color'].capitalize()} Key: {k['distance']} steps away at {k['pos']}")
        else:
            lines.append(f"- {k['color'].capitalize()} Key at {k['pos']}")
    else:
        # If key is gone and not in hand, we assume it's used or lost (rare in this env)
        if not carrying or carrying.type != 'key':
            lines.append(f"- Key: Not visible")

    if objects['door']:
        d = objects['door']
        if include_distances:
            lines.append(f"- {d['color'].capitalize()} Door: {d['distance']} steps away. Status: [{d['state']}]")
        else:
            lines.append(f"- {d['color'].capitalize()} Door. Status: [{d['state']}]")
        
    if objects['goal']:
        g = objects['goal']
        if include_distances:
            lines.append(f"- Goal: {g['distance']} steps away at {g['pos']}")
        else:
            lines.append(f"- Goal at {g['pos']}")

    return "\n".join(lines)


def get_compact_global_description(env, include_distances=True):
    """
    JSON-style format. COMPACT
    LLMs are trained on code/JSON. 
    They understand key-value pairs better than paragraphs TEXT
    Args:
        env: MiniGrid environment WRAPPED WITH ImgObsWrapper
        include_distances (bool): Whether to include distances to objects in the description.
        neded for ablation studies ==> distances vs raw positions
    """
    #input is env WRAPPED WITH ImgObsWrapper => need to unwrap to access full state
    base_env = env.unwrapped

    # Agent state
    agent_pos = sanitize(base_env.agent_pos)
    agent_dir = base_env.agent_dir
    facing = DIRECTION_TO_TEXT.get(agent_dir, "Unknown")
    carrying = base_env.carrying
    
    key_info = "Not Found"
    door_info = "Not Found"
    goal_dist = "Unknown"
    
    for x in range(base_env.grid.width):
            for y in range(base_env.grid.height):
                cell = base_env.grid.get(x, y)
                if cell:
                    dist = int(abs(x - agent_pos[0]) + abs(y - agent_pos[1]))
                    obj_pos = (int(x), int(y))
                    
                    if cell.type == "key":
                        if include_distances:
                            key_info = f"dist={dist}"
                        else:
                            key_info = f"pos={obj_pos}"
                    elif cell.type == "door":
                        state = "Locked" if cell.is_locked else ("Open" if cell.is_open else "Closed")
                        if include_distances:
                            door_info = f"dist={dist}, state={state}"
                        else:
                            door_info = f"pos={obj_pos}, state={state}"
                    elif cell.type == "goal":
                        if include_distances:
                            goal_dist = dist
                        else:
                            goal_dist = f"pos={obj_pos}"
                    
    inventory = f"{carrying.color} {carrying.type}" if carrying else "None"
    
    return (
        f"{{ 'Agent': {{ 'pos': {agent_pos}, 'facing': '{facing}', 'inventory': '{inventory}' }}, "
        f"'Key': '{key_info}', "
        f"'Door': '{door_info}', "
        f"'Goal_Dist': {goal_dist} }}"
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

    for i in range(1, 20):
        
        action = env.action_space.sample()
            
        action_name = ACTION_TO_TEXT.get(action, "unknown")
        
        # Execute
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Generate Compact Description
        desc = get_compact_global_description(env)
        descVerbose = get_enhanced_global_description(env)
        
        print("-" * 50)
        print(f"Step {i} | Action: {action} ({action_name.upper()})")
        print(f"LLM Input: {desc}")
        
        # FORCE PRINT TO TERMINAL => AVOID BUFFERING ISSUES
        sys.stdout.flush() 
        
        # Pause for visualization
        time.sleep(2.5) 
            
    env.close()
