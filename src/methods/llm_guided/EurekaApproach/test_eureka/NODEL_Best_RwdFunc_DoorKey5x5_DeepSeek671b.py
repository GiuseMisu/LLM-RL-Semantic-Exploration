def compute_reward(env):
    reward = -0.01
    key_pos = None
    door_pos = None
    door_locked = True
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
                door_locked = obj.is_locked
            elif obj.type == 'goal':
                goal_pos = (x, y)

    target = None
    if env.carrying is None:
        if key_pos is not None:
            target = key_pos
        else:
            target = goal_pos
    else:
        if door_pos is not None and door_locked:
            target = door_pos
        else:
            target = goal_pos

    if target is not None:
        agent_x, agent_y = env.agent_pos
        target_x, target_y = target
        distance = abs(agent_x - target_x) + abs(agent_y - target_y)
        max_distance = env.grid.width + env.grid.height - 2
        if max_distance > 0:
            reward += (max_distance - distance) / max_distance
        else:
            reward += 1.0

    reward = max(-0.1, min(1.0, reward))
    return reward