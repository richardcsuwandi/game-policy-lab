"""Small deterministic brick-breaker. One paddle, one ball, one life per serve."""
import random

WIDTH, PADDLE_Y, HALF, STEP_LIMIT = 48, 32, 2, 600
SELECTION = tuple(range(16))
FINAL = tuple(range(1000, 1016))

FOLLOW = '''def choose_action(state):
    x = state["x"]
    p = state["paddle"]
    return 1 if x > p else -1 if x < p else 0
'''
PREDICT = '''def choose_action(state):
    target = state["x"]
    if state["vy"] > 0:
        ticks = state["paddle_y"] - state["y"]
        edge = state["width"] - 1
        phase = (target + ticks * state["vx"]) % (2 * edge)
        target = min(phase, 2 * edge - phase)
    p = state["paddle"]
    return 1 if target > p else -1 if target < p else 0
'''


def brick(x, y):
    if 2 <= y < 5:
        return 1 << ((y - 2) * 8 + x // 6)
    return 0


def play(policy, seed, trace=False):
    rng = random.Random(seed)
    x, y = rng.randrange(5, WIDTH - 5), rng.randrange(14, 21)
    vx, vy = rng.choice((-4, -3, -2, 2, 3, 4)), -1
    paddle, mask, hits = WIDTH // 2, (1 << 24) - 1, 0
    reason, frames = 'step_limit', []

    def frame():
        return [x, y, paddle, mask]

    if trace:
        frames.append(frame())
    for step in range(1, STEP_LIMIT + 1):
        action = policy(dict(x=x, y=y, vx=vx, vy=vy, paddle=paddle, width=WIDTH,
                             paddle_y=PADDLE_Y, half=HALF, bricks=mask))
        if type(action) is not int or action not in (-1, 0, 1):
            raise ValueError('Expected integer action -1, 0 or 1')
        paddle = min(WIDTH-1-HALF, max(HALF, paddle + 2 * action))
        for _ in range(abs(vx)):
            dx = 1 if vx > 0 else -1
            if not 0 <= x + dx < WIDTH:
                vx, dx = -vx, -dx
            obstacle = mask & brick(x + dx, y)
            if obstacle:
                mask ^= obstacle
                vx = -vx
                break
            x += dx
        if y + vy < 0:
            vy = 1
        obstacle = mask & brick(x, y + vy)
        if obstacle:
            mask ^= obstacle
            vy = -vy
        else:
            y += vy
        if y == PADDLE_Y:
            if abs(x - paddle) > HALF:
                reason = 'miss'
            else:
                hits += 1
                vy = -1
                if x != paddle:
                    vx = abs(vx) if x > paddle else -abs(vx)
        if not mask:
            reason = 'cleared'
        if trace:
            frames.append(frame())
        if reason != 'step_limit':
            break
    result = dict(seed=seed, bricks=24-mask.bit_count(), steps=step, returns=hits,
                  end=reason, end_gap=abs(x-paddle))
    if trace:
        result['frames'] = frames
    return result


def evaluate(policy, seeds, replay):
    episodes = [play(policy, seed) for seed in seeds]
    result = dict(mean_bricks=sum(e['bricks'] for e in episodes)/len(episodes), episodes=episodes)
    if replay:
        result['replay'] = play(policy, seeds[0], True)
    return result
