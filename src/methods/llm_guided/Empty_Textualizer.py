import numpy as np
import os
import warnings
from minigrid.core.constants import OBJECT_TO_IDX

# --- 1. SILENCE WARNINGS ---
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


DIRECTION_TO_TEXT = {0: "East", 1: "South", 2: "West", 3: "North"}

def get_relative_direction(agent_pos, agent_dir, target_pos):
    """
    Returns: 'Front', 'Left', 'Right', 'Behind' or 'Here'
    """
    if agent_pos == target_pos: return "Here"

    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    # MiniGrid: 0=East, 1=South, 2=West, 3=North
    forward_map = { 0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1) }
    fx, fy = forward_map[agent_dir]
    
    dot = dx*fx + dy*fy
    cross = fx*dy - fy*dx
    
    if dot > 0 and abs(cross) <= abs(dot): return "Front"
    if dot < 0 and abs(cross) <= abs(dot): return "Behind"
    if cross > 0: return "Right"
    return "Left"

def get_EMPTY_description(env):
    """
    Textualizer specifically for MiniGrid-Empty.
    Focuses ONLY on the Goal.
    """
    # 1. Unpack Environment
    base_env = env.unwrapped
    agent_pos = tuple(int(x) for x in base_env.agent_pos)
    agent_dir = base_env.agent_dir
    facing = DIRECTION_TO_TEXT.get(agent_dir, "Unknown")
    
    goal_str = "Unknown"

    # 2. Scan for Goal
    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            cell = base_env.grid.get(x, y)
            if cell and cell.type == "goal":
                target_pos = (x, y)
                dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                relative_dir = get_relative_direction(agent_pos, agent_dir, target_pos)
                
                # Check Reachability (Simple for Empty env)
                is_reachable = (dist == 1 and relative_dir == "Front")
                reach_str = " <REACHABLE>" if is_reachable else ""
                
                goal_str = f"loc={target_pos}, dist={dist}, dir={relative_dir}{reach_str}"

    # 3. Construct Description
    # Note: No Inventory, No Keys, No Doors
    return (
        f"{{ 'Agent': {{ 'pos': {agent_pos}, 'facing': '{facing}' }}, "
        f"'Goal': '{goal_str}' }}"
    )