"""Bundled examples for plumbing tests, NOT model-generated improvements."""

BASELINE = '''def choose_action(state):
    x, y = state["snake"][0]
    fx, fy = state["food"]
    if fx > x:
        return "right"
    if fx < x:
        return "left"
    return "down" if fy > y else "up"
'''

SAFE = '''def choose_action(state):
    x, y = state["snake"][0]
    fx, fy = state["food"]
    moves = [("up", 0, -1), ("right", 1, 0), ("down", 0, 1), ("left", -1, 0)]
    options = []
    for action, dx, dy in moves:
        point = [x + dx, y + dy]
        body = state["snake"] if point == state["food"] else state["snake"][:-1]
        if point in body or point in state["obstacles"]:
            continue
        if not (0 <= point[0] < state["width"] and 0 <= point[1] < state["height"]):
            continue
        distance = abs(point[0] - fx) + abs(point[1] - fy)
        options.append((distance, action))
    return min(options)[1] if options else "up"
'''

SPACE = '''def choose_action(state):
    x, y = state["snake"][0]
    fx, fy = state["food"]
    width, height = state["width"], state["height"]
    moves = [("up", 0, -1), ("right", 1, 0), ("down", 0, 1), ("left", -1, 0)]
    options = []
    for action, dx, dy in moves:
        point = (x + dx, y + dy)
        eating = list(point) == state["food"]
        body = state["snake"] if eating else state["snake"][:-1]
        occupied = set(tuple(p) for p in body + state["obstacles"])
        if point in occupied or not (0 <= point[0] < width and 0 <= point[1] < height):
            continue
        visited = {point}
        queue = [point]
        for px, py in queue:
            for _, sx, sy in moves:
                neighbor = (px + sx, py + sy)
                if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height
                        and neighbor not in occupied and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
        distance = abs(point[0] - fx) + abs(point[1] - fy)
        trap_penalty = max(0, len(state["snake"]) + 3 - len(visited)) * 20
        options.append((distance + trap_penalty, -len(visited), action))
    return min(options)[2] if options else "up"
'''
