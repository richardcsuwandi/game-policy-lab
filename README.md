# Game Policy Lab

A small Python environment for testing model-written policies in Snake and Breakout. A policy is one function, `choose_action(state)`. Between games, a model can propose a replacement. The evaluator keeps it only if its average score improves on the same set of games. Separate games measure the final score.

## Run offline

From the repository root, use uv and Python 3.12. The offline examples use handwritten policies and require no credentials or Docker.

```sh
uv sync --locked --python 3.12
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m game_policy --provider mock --output runs/snake-1
.venv/bin/python -m game_policy.breakout_run offline runs/breakout-1
```

Use a new output directory for each run. For Snake, read `final.json` to compare food collected by the starter and retained policy. For Breakout, `gate.json` compares two bundled paddles on the same serves.

## Run with a model

Install and start Docker. Build and verify the worker from the repository root.

```sh
docker build -t game-policy-worker:local .
GAME_POLICY_TEST_DOCKER=1 .venv/bin/python -m unittest discover -s tests -p test_docker.py -v
```

The model adapter runs on the host. Generated game code runs inside a container without networking or API credentials. The worker enforces time and memory limits.

### Connect your own model

Create a trusted Python module such as `my_adapter.py` in the repository root. It can call a local model or your provider’s SDK. Expose a callable with this signature:

```python
def propose(*, system, current, feedback, generation, max_tokens):
    # Send the game's system prompt, current policy and feedback to your model.
    # Apply max_tokens to the request. Parse the answer into these two strings.
    return {"summary": summary, "code": source}
```

`system` describes the game, the allowed Python syntax and the required return value. `current` is the retained policy's source. `feedback` is a dictionary of scores or errors, and `generation` counts attempts starting at one. Return the full function as a string. Keep credentials and private prompts specific to your provider in your local configuration.

```sh
.venv/bin/python -m game_policy --provider my_adapter:propose \
  --generations 2 --max-tokens 4096 --output runs/snake-live-1

# Use the directory from the offline Breakout comparison above.
.venv/bin/python -m game_policy.breakout_run live runs/breakout-1 --provider my_adapter:propose
```

Snake introduces a wall at generation four by default. Use `--obstacles-at 2` to introduce it before the second edit or `--obstacles-at 0` for an open board throughout. When the wall appears, the evaluator first scores the retained policy on the changed board. Breakout requests up to two edits if its offline comparison passes.

### Optional Naive adapter

The original live integration was tested with `naive-n0.5-flash`. To use this adapter, copy `.env.example` to `.env` in a fresh clone and set `NAIVE_API_KEY`. Shell variables override the file. The adapter uses streaming Messages, records the requested tier and keeps thinking separate from answer text.

```sh
.venv/bin/python -m game_policy --provider naive --connection-smoke --output runs/connection-1
.venv/bin/python -m game_policy --provider naive --generations 2 --max-tokens 4096 --output runs/snake-naive-1
```

These two commands make live API requests. To use the same adapter with Breakout, pass `--provider naive` to its live command.
