def compute_reward(env):
    step_penalty = -0.02
    agent_x, agent_y = env.agent_pos
    
    cell = env.grid.get(agent_x, agent_y)
    if cell is not None and cell.type == 'goal':
        return 1.0
        
    key_pos = None
    door_pos = None
    door_is_locked = False
    door_is_open = False
    goal_pos = None
    
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            if obj.type == 'key':
                key_pos = (x, y)
            elif obj.type == 'door':
                door_pos = (x, y)
                door_is_locked = obj.is_locked
                door_is_open = obj.is_open
            elif obj.type == 'goal':
                goal_pos = (x, y)
                
    has_key = env.carrying is not None and env.carrying.type == 'key'
    
    if door_is_open:
        stage = 2
        target = goal_pos
    elif has_key:
        stage = 1
        target = door_pos
    else:
        stage = 0
        target = key_pos
        
    if target is None:
        return max(-0.1, min(1.0, step_penalty))
        
    dist = abs(agent_x - target[0]) + abs(agent_y - target[1])
    max_dist = env.grid.width + env.grid.height
    
    base_rewards = [0.1, 0.4, 0.7]
    base_reward = base_rewards[stage]
    
    dist_reward = (1 - dist / max_dist) * 0.5
    
    interaction_bonus = 0.0
    front_x, front_y = agent_x, agent_y
    if env.agent_dir == 0:
        front_x += 1
    elif env.agent_dir == 1:
        front_y += 1
    elif env.agent_dir == 2:
        front_x -= 1
    elif env.agent_dir == 3:
        front_y -= 1
        
    if 0 <= front_x < env.grid.width and 0 <= front_y < env.grid.height:
        front_obj = env.grid.get(front_x, front_y)
        if front_obj is not None:
            if stage == 0 and front_obj.type == 'key' and not has_key:
                interaction_bonus = 0.1
            elif stage == 1 and front_obj.type == 'door' and front_obj.is_locked and has_key:
                interaction_bonus = 0.1
    
    reward = base_reward + dist_reward + interaction_bonus + step_penalty
    
    return max(-0.1, min(1.0, reward))