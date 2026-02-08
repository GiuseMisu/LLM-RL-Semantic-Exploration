def compute_reward(env):
    ax, ay = env.agent_pos
    reward = -0.01
    key_pos = None
    door_pos = None
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
            elif obj.type == 'goal':
                goal_pos = (x, y)
    has_key = env.carrying is not None and getattr(env.carrying, 'type', None) == 'key'
    max_dist = env.grid.width + env.grid.height - 2
    target = None
    stage_weight = 0.0
    if not has_key:
        if key_pos is not None:
            target = key_pos
            stage_weight = 0.2
    else:
        if door_pos is not None:
            door_obj = env.grid.get(door_pos[0], door_pos[1])
            if door_obj is not None and door_obj.type == 'door' and getattr(door_obj, 'is_open', False):
                if goal_pos is not None:
                    target = goal_pos
                    stage_weight = 0.6
            else:
                target = door_pos
                stage_weight = 0.4
        else:
            if goal_pos is not None:
                target = goal_pos
                stage_weight = 0.6
    if target is not None and max_dist > 0:
        tx, ty = target
        dist = abs(ax - tx) + abs(ay - ty)
        norm = dist / max_dist
        reward += stage_weight * (1 - norm)
    def dir_to(ax, ay, tx, ty):
        dx = tx - ax
        dy = ty - ay
        if dx == 1 and dy == 0:
            return 0
        if dx == -1 and dy == 0:
            return 2
        if dy == 1 and dx == 0:
            return 1
        if dy == -1 and dx == 0:
            return 3
        return None
    if not has_key and key_pos is not None:
        dist_key = abs(ax - key_pos[0]) + abs(ay - key_pos[1])
        if dist_key == 1:
            needed = dir_to(ax, ay, key_pos[0], key_pos[1])
            if needed is not None and env.agent_dir == needed:
                reward += 0.05
    if has_key and door_pos is not None:
        dist_door = abs(ax - door_pos[0]) + abs(ay - door_pos[1])
        if dist_door == 1:
            needed = dir_to(ax, ay, door_pos[0], door_pos[1])
            if needed is not None and env.agent_dir == needed:
                reward += 0.1
    if has_key:
        reward += 0.02
    if door_pos is not None:
        door_obj = env.grid.get(door_pos[0], door_pos[1])
        if door_obj is not None and door_obj.type == 'door' and getattr(door_obj, 'is_open', False):
            reward += 0.15
    obj_here = env.grid.get(ax, ay)
    if obj_here is not None and obj_here.type == 'goal':
        reward = 1.0
    if reward > 1.0:
        reward = 1.0
    if reward < -0.1:
        reward = -0.1
    return float(reward)