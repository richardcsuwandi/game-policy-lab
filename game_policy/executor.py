import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from game_policy.breakout import FOLLOW, PREDICT
from game_policy.policies import BASELINE, SAFE, SPACE
from game_policy.policy_contract import validate

ROOT = Path(__file__).resolve().parent.parent


def evaluate_code(code, seeds, blocked, trusted_mock=False, replay=False, *, game="snake"):
    validate(code)
    if game not in ("snake", "breakout"):
        raise ValueError("Unknown evaluator")
    fixtures = (BASELINE, SAFE, SPACE) if game == "snake" else (FOLLOW, PREDICT)
    if trusted_mock and code not in fixtures:
        raise ValueError("Local execution only permits exact bundled mock policies")
    request = json.dumps({"code": code, "seeds": list(seeds), "blocked": blocked,
                          "replay": replay, "game": game})
    name = "game-policy-" + uuid.uuid4().hex
    if trusted_mock:
        command = [sys.executable, "-m", "game_policy.worker"]
    else:
        command = ["docker", "run", "--rm", "--name", name, "-i", "--network=none",
                   "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--pids-limit=32", "--memory=256m", "--cpus=1", "--user=65534:65534",
                   "--ulimit", "fsize=1048576:1048576",
                   "--mount", f"type=bind,src={ROOT / 'game_policy'},dst=/app/game_policy,readonly",
                   "--workdir=/app", "--env=PYTHONDONTWRITEBYTECODE=1",
                   "game-policy-worker:local", "python", "-m", "game_policy.worker"]
    env = {key: value for key, value in os.environ.items()
           if key in ("PATH", "SYSTEMROOT", "DOCKER_HOST", "DOCKER_CONTEXT", "XDG_RUNTIME_DIR")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # File output avoids unbounded pipe allocation if a candidate prints excessively.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(command, input=request.encode(), stdout=stdout,
                                     stderr=stderr, cwd=ROOT, env=env, timeout=20)
        except subprocess.TimeoutExpired:
            if not trusted_mock:
                try:
                    removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                                             env=env, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    raise RuntimeError(f"Could not confirm timeout cleanup. Inspect container {name}.") from None
                if removed.returncode:
                    raise RuntimeError(f"Could not confirm timeout cleanup. Inspect container {name}.")
            raise ValueError("Evaluation exceeded 20-second deadline") from None
        if process.returncode:
            raise ValueError("Worker failed; candidate error or unavailable Docker worker")
        stdout.seek(0)
        raw = stdout.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Worker output exceeded limit")
        result = json.loads(raw)
    metric, mean, cap = ("food", "mean_food", 400) if game == "snake" else ("bricks", "mean_bricks", 24)
    score = result.get(mean)
    episodes = result.get("episodes", [])
    if (not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= cap or len(episodes) != len(seeds)
            or [e.get("seed") for e in episodes] != list(seeds)
            or any(type(e.get(metric)) is not int or not 0 <= e[metric] <= cap
                   for e in episodes)
            or abs(score - sum(e[metric] for e in episodes) / len(seeds)) > 1e-9):
        raise ValueError("Invalid worker result")
    return result
