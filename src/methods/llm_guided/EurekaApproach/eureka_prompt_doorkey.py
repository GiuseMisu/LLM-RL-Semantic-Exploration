MINIGRID_API_CONTEXT_DOORKEY = """
You are an expert reward engineer for Reinforcement Learning agents in the MiniGrid environment.
Your goal is to design in Python code a DENSE reward function that provides continuous feedback to guide the agent toward solving the task.

ENVIRONMENT CONFIGURATION:
- Task: MiniGrid DoorKey
- Grid Size: {width}x{height}

ENVIRONMENT API DETAILS:
The `env` object is the RAW MiniGrid environment with these attributes:

1. `env.agent_pos`: is a TUPLE (x, y), NOT a NumPy array! DO NOT compare it directly with numpy arrays.
    x, y = env.agent_pos here x=column, y=row. Origin (0,0) is top-left corner.

2. `env.agent_dir`: int representing direction agent is facing
   - 0 = East (right)
   - 1 = South (down)
   - 2 = West (left)
   - 3 = North (up)

3. `env.carrying`: The object currently held by agent (None if empty hands)
   - Example check: `env.carrying is not None and env.carrying.type == 'key'`
   - Possible types: 'key', 'ball', 'box', etc.

4. `env.grid`: Grid object containing all environment objects
   - `env.grid.get(x, y)`: Returns object at position (x,y) or None if empty
   - `env.grid.width`: Grid width (integer)
   - `env.grid.height`: Grid height (integer)
   
5. Object attributes (when env.grid.get(x,y) returns an object):
   Always check `if obj is not None` before accessing attributes!
   - .type: One of ['wall', 'door', 'key', 'goal'] 
   - .is_open: bool (Only valid if type == 'door')
   - .is_locked: bool (Only valid if type == 'door')
   - .color: string ('red', 'green', 'blue', 'yellow', 'purple', 'grey')

DOORKEY TASK BREAKDOWN:
The agent must complete these stages sequentially:
1. **Locate Key**: Navigate to find the key object in the grid
2. **Pick Up Key**: Stand next to key and execute toggle/pickup action
3. **Locate Door**: Navigate to find the locked door
4. **Unlock Door**: Stand in front of door (while holding key) and execute toggle action
5. **Reach Goal**: Navigate through open door to reach the goal tile

INTERACTION RULES:
To PICKUP an object or TOGGLE/OPEN a door, you must be in the adjacent cell facing it (dir=Front). 
You cannot interact with objects to your Left, Right, or Behind.

REWARD DESIGN PRINCIPLES:
- Use DENSE rewards: provide continuous feedback at every step based on progress
- Scale rewards appropriately: keep values roughly in [-0.1, 1] range for training stability
- Encourage progress through stages: higher reward for completing later stages
- Use distance-based shaping: reward decreasing distance to current objective
- Avoid reward conflicts: don't simultaneously reward contradictory behaviors
- Handle edge cases: check for None before accessing object attributes
- To prevent the agent from just standing still near the target to accumulate rewards, consider penalizing each time step slightly or using potential-based rewards (difference in distance between steps).

MANDATORY ANTI-EXPLOITATION RULES:
1. You MUST include a small negative reward per step (e.g., -0.01) to prevent the agent from standing still.
WITHOUT this penalty, the agent WILL exploit distance-based rewards by standing near targets.
2. Distance-based rewards should be SMALL, not large. 
3. One-time bonuses for milestones (key pickup and door open) should be larger than distance rewards
4. Final reward MUST be clamped: `return max(-0.1, min(1.0, reward))`

DISTANCE CALCULATION HELPERS:
- Manhattan distance: `abs(x1 - x2) + abs(y1 - y2)`
- Normalize by grid size: `distance / (env.grid.width + env.grid.height)`

CRITICAL CONSTRAINTS and REQUIREMENTS:
- Output a SINGLE Python function named `compute_reward(env)`.
- The reward should be DENSE (continuous feedback based on distance) to guide exploration.
- SCALING: The output reward at each step MUST strictly be between -0.1 and +1.0. Do not return values outside this range.
- The Python function MUST return a single float value
- Function signature MUST be exactly `def compute_reward(env):`
- Available imports: Only `np` (numpy) and `math` are available
- Safety: Always check if `env.grid.get(x, y)` returns None before accessing attributes
- Do not use `env.step()` inside the reward function.
- Always check `if obj is not None` before accessing `.type`.
- It is forbidden to generate a function that returns values outside the range [-0.1, 1.0]
- CRITICAL SAFETY: `env.grid.get(x, y)` returns `None` for empty tiles. YOU MUST check `if obj is not None:` before accessing `obj.type`.
- Do not use `env.get_cell(...)` or other helper methods. Use only `env.grid.get(x,y)`.
"""

EUREKA_INITIAL_PROMPT_DOORKEY = f"""
{MINIGRID_API_CONTEXT_DOORKEY}

TASK:
Write a Python function named `compute_reward(env)` that implements a dense reward function for the MiniGrid DoorKey task.

OUTPUT REQUIREMENTS:
- Return ONLY the Python function code
- DO NOT use any methods like get_direction(), get_obj_dir(), or cur_pos - they don't exist
- ONLY use the attributes listed above: agent_pos, agent_dir, carrying, grid.get(), grid.width, grid.height
- NO explanations, NO comments, NO markdown formatting
- NO text before or after the function
- Function must be named exactly: `compute_reward(env)`
- Function must return a float value between -0.1 and +1.0 at each step
- It is forbidden to generate a function that returns values outside the range [-0.1, 1.0]
- Do not include any code comments or docstrings

Start your response with: def compute_reward(env):
"""

EUREKA_FEEDBACK_PROMPT_TEMPLATE_DOORKEY = """
I tested your precious reward function. 

PREVIOUS CODE SUBMISSION:
{previous_code}

Here is the detailed performance breakdown:
NUMERICAL METRICS:
- Global Success Rate: {success_rate:.2f}% (Goal Reached)
- Key Pickup Rate: {key_pickup_rate:.2f}% (Sub-goal 1)
- Door Open Rate: {door_open_rate:.2f}% (Sub-goal 2)
- Mean Episodic Reward: {mean_reward:.2f}
- Mean Steps to Success: {mean_steps}

ERROR LOGS:
{error_log}

QUALITATIVE ANALYSIS:
{feedback_text}

TASK:
Analyze the correlation between the metrics above and the reward logic.
1. If Key Pickup is low, the agent isn't exploring enough or the reward for reaching the key is too weak.
2. If Key Pickup is high but Door Open is low, the agent doesn't know what to do with the key. Check the "carrying" logic.
3. If Success is low but Door Open is high, the agent isn't navigating to the goal after opening the door.

OUTPUT REQUIREMENTS:
- Return ONLY the improved Python function code
- NO explanations, NO comments, NO markdown formatting
- NO text before or after the function
- Function must be named exactly: `compute_reward(env)`
- Function must return a float value
- Do not include any code comments or docstrings

Start your response with: def compute_reward(env):
"""