"""Worker protocol. Untrusted policies must run inside the Docker executor."""

import json
import sys

from game_policy.game import evaluate, play
from game_policy.policy_contract import load


def main():
    request = json.load(sys.stdin)
    policy = load(request["code"])
    if request.get("game") == "breakout":
        from game_policy import breakout
        json.dump(breakout.evaluate(policy, request["seeds"], request["replay"]), sys.stdout)
        return
    result = evaluate(policy, request["seeds"], request["blocked"])
    if request.get("replay"):
        result["replay"] = play(policy, request["seeds"][0], request["blocked"], trace=True)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
