def compute_reward(env):
    reward = -0.01
    agent_x, agent_y = env.agent_pos
    carrying_key = env.carrying is not None and env.carrying.type == 'key'
    key_pos = None
    door_pos = None
    goal_pos = None
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is not None:
                if obj.type == 'key':
                    key_pos = (x, y)
                elif obj.type == 'door':
                    door_pos = (x, y)
                elif obj.type == 'goal':
                    goal_pos = (x, y)
    if not carrying_key and key_pos is not None:
        dist_to_key = abs(agent_x - key_pos[0]) + abs(agent_y - key_pos[1])
        max_dist = env.grid.width + env.grid.height
        reward += (1.0 - (dist_to_key / max_dist)) * 0.1
    elif carrying_key and not (door_pos is not None and env.grid.get(door_pos[0], door_pos[1]).is_open):
        if door_pos is not None:
            dist_to_door = abs(agent_x - door_pos[0]) + abs(agent_y - door_pos[1])
            max_dist = env.grid.width + env.grid.height
            reward += (1.0 - (dist_to_door / max_dist)) * 0.2
    elif carrying_key and door_pos is not None and env.grid.get(door_pos[0], door_pos[1]).is_open:
        if goal_pos is not None:
            dist_to_goal = abs(agent_x - goal_pos[0]) + abs(agent_y - goal_pos[1])
            max_dist = env.grid.width + env.grid.height
            reward += (1.0 - (dist_to_goal / max_dist)) * 0.3
    if carrying_key and not hasattr(env, '_key_picked_up'):
        reward += 0.5
        env._key_picked_up = True
    door_obj = None
    if door_pos is not None:
        door_obj = env.grid.get(door_pos[0], door_pos[1])
    if door_obj is not None and door_obj.is_open and not hasattr(env, '_door_opened'):
        reward += 0.7
        env._door_opened = True
    if door_obj is not None and door_obj.is_open and goal_pos is not None:
        if agent_x == goal_pos[0] and agent_y == goal_pos[1]:
            reward += 1.0
    return max(-0.1, min(1.0, reward))