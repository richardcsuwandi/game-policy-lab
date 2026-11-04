"""Fixed evaluator. Candidate policies receive observations, never this module's globals."""

import random

DIRECTIONS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
TRAIN_SEEDS = tuple(range(8))
TEST_SEEDS = tuple(range(1000, 1016))


def obstacles(enabled):
    return [[7, y] for y in range(2, 9)] if enabled else []


def play(policy, seed, blocked=False, max_steps=400, trace=False):
    rng = random.Random(seed)
    width = height = 12
    snake = [[4, 6], [3, 6], [2, 6]]
    walls = obstacles(blocked)
    direction = "right"
    score = 0
    hunger = 0
    frames = []

    def spawn_food():
        free = [[x, y] for y in range(height) for x in range(width)
                if [x, y] not in snake and [x, y] not in walls]
        return rng.choice(free) if free else None

    food = spawn_food()
    reason = "step_limit"
    for step in range(max_steps):
        state = {"width": width, "height": height, "snake": [p[:] for p in snake],
                 "food": food[:], "obstacles": [p[:] for p in walls],
                 "direction": direction, "score": score}
        if trace:
            frames.append(state)
        action = policy(state)
        if action not in DIRECTIONS:
            raise ValueError("choose_action must return up/right/down/left")
        dx, dy = DIRECTIONS[action]
        old_dx, old_dy = DIRECTIONS[direction]
        if (dx, dy) == (-old_dx, -old_dy):
            reason = "reverse"
            break
        head = [snake[0][0] + dx, snake[0][1] + dy]
        eating = head == food
        body = snake if eating else snake[:-1]
        if not (0 <= head[0] < width and 0 <= head[1] < height):
            reason = "boundary"
            break
        if head in walls or head in body:
            reason = "obstacle" if head in walls else "body"
            break
        snake.insert(0, head)
        direction = action
        if eating:
            score += 1
            hunger = 0
            food = spawn_food()
            if food is None:
                reason = "filled_board"
                break
        else:
            snake.pop()
            hunger += 1
        if hunger >= 120:
            reason = "starvation"
            break
    result = {"seed": seed, "food": score, "steps": step + 1, "reason": reason}
    if trace:
        result["frames"] = frames
    return result


def evaluate(policy, seeds, blocked=False):
    episodes = [play(policy, seed, blocked) for seed in seeds]
    return {"mean_food": sum(e["food"] for e in episodes) / len(episodes),
            "episodes": episodes}
