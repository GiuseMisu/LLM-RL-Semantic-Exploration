def compute_reward(env):
    import math
    agent_x, agent_y = env.agent_pos
    agent_dir = env.agent_dir
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
                if obj.type == 'key':
                    key_pos = (x, y)
                    dist = abs(agent_x - x) + abs(agent_y - y)
                    if dist < min_dist_to_key:
                        min_dist_to_key = dist
                elif obj.type == 'door' and obj.is_locked:
                    door_pos = (x, y)
                    dist = abs(agent_x - x) + abs(agent_y - y)
                    if dist < min_dist_to_key:
                        min_dist_to_door = dist
                elif obj.type == 'goal':
                    goal_pos = (x, y)
                    dist = abs(agent_x - x) + abs(agent_y - y)
                    if dist < min_dist_to_goal:
                        min_dist_to_goal = dist
    
    max_dist = env.grid.width + env.grid.height
    norm_min_dist_to_key = min_dist_to_key / max_dist
    norm_min_dist_to_door = min_dist_to_door / max_dist if min_dist_to_door != float('inf') else 1.0
    norm_min_dist_to_goal = min_dist_to_goal / max_dist if min_dist_to_goal != float('inf') else 1.0
    
    reward = -0.01
    
    if carrying_key:
        reward += 0.3 * (1 - norm_min_dist_to_door)
        if door_pos:
            dx, dy = door_pos
            front_x, front_y = agent_x, agent_y
            if agent_dir == 0:
                front_x += 1
            elif agent_dir == 1:
                front_y += 1
            elif agent_dir == 2:
                front_x -= 1
            elif agent_dir == 3:
                front_y -= 1
            
            if (front_x, front_y) == door_pos:
                door_obj = env.grid.get(front_x, front_y)
                if door_obj is not None and door_obj.type == 'door' and not door_obj.is_locked:
                    reward += 0.5
    else:
        reward += 0.2 * (1 - norm_min_dist_to_key)
        if key_pos:
            dx, dy = key_pos
            front_x, front_y = agent_x, agent_y
            if agent_dir == 0:
                front_x += 1
            elif agent_dir == 1:
                front_y += 1
            elif agent_dir == 2:
                front_x -= 1
            elif agent_dir == 3:
                front_y -= 1
            
            if (front_x, front_y) == key_pos:
                key_obj = env.grid.get(front_x, front_y)
                if key_obj is not None and key_obj.type == 'key':
                    reward += 0.3
    
    if not carrying_key and norm_min_dist_to_key < 0.1:
        reward += 0.1
        
    if carrying_key and norm_min_dist_to_door < 0.1:
        reward += 0.1
        
    reward += 0.5 * (1 - norm_min_dist_to_goal)
    
    return max(-0.1, min(1.0, reward))