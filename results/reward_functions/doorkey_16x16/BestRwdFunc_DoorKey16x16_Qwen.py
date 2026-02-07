def compute_reward(env):
    import numpy as np
    
    agent_x, agent_y = env.agent_pos
    carrying_key = env.carrying is not None and env.carrying.type == 'key'
    
    min_dist_to_key = float('inf')
    min_dist_to_door = float('inf')
    min_dist_to_goal = float('inf')
    door_pos = None
    key_pos = None
    goal_pos = None
    
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is not None:
                if obj.type == 'key' and not carrying_key:
                    d = abs(agent_x - x) + abs(agent_y - y)
                    if d < min_dist_to_key:
                        min_dist_to_key = d
                        key_pos = (x, y)
                elif obj.type == 'door' and obj.is_locked:
                    d = abs(agent_x - x) + abs(agent_y - y)
                    if d < min_dist_to_door:
                        min_dist_to_door = d
                        door_pos = (x, y)
                elif obj.type == 'goal':
                    d = abs(agent_x - x) + abs(agent_y - y)
                    if d < min_dist_to_goal:
                        min_dist_to_goal = d
                        goal_pos = (x, y)
    
    max_dist = env.grid.width + env.grid.height
    reward = -0.01
    
    if not carrying_key and key_pos is not None:
        norm_dist = min_dist_to_key / max_dist
        reward += (1.0 - norm_dist) * 0.1
    elif carrying_key and door_pos is not None:
        norm_dist = min_dist_to_door / max_dist
        reward += (1.0 - norm_dist) * 0.2
        if min_dist_to_door < 1.5:
            reward += 0.1
    elif carrying_key and door_pos is None:
        if goal_pos is not None:
            norm_dist = min_dist_to_goal / max_dist
            reward += (1.0 - norm_dist) * 0.05
            
    return max(-0.1, min(1.0, reward))