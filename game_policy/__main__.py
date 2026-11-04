import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from game_policy.executor import evaluate_code
from game_policy.game import TRAIN_SEEDS, TEST_SEEDS
from game_policy.messages import MessageError
from game_policy.policies import BASELINE
from game_policy.providers import MockProvider, live_provider


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(args):
    if args.generations < 1 or args.obstacles_at < 0:
        raise ValueError('Use positive generations and nonnegative obstacles-at (0 disables)')
    trusted = args.provider == 'mock'
    smoke = getattr(args, 'connection_smoke', False)
    if smoke and args.provider != 'naive':
        raise ValueError('Use --provider naive for a connection check.')
    provider = MockProvider() if trusted else live_provider(
        args.provider, tier=args.tier, max_tokens=args.max_tokens)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(Path(__file__).parent.glob('*.py'))}
    save(output / 'config.json', {'provider': provider.name, 'mock': trusted,
         'model': getattr(provider, 'model', None), 'generations': 0 if smoke else args.generations,
         'kind': 'connection_smoke' if smoke else 'experiment',
         'requested_tier': getattr(provider, 'tier', None),
         'served_tier': 'unverified' if args.provider == 'naive' else None,
         'max_tokens_per_request': getattr(provider, 'max_tokens', None),
         'obstacles_at': args.obstacles_at, 'train_seeds': TRAIN_SEEDS,
         'test_seeds': TEST_SEEDS, 'source_sha256': hashes,
         'warning': 'Mock proposals are handwritten fixtures, not AI results' if trusted else
                    ('Requested tier alone does not confirm the served tier.' if args.provider == 'naive' else None)})
    start = time.monotonic()

    def event(value):
        value['elapsed_seconds'] = time.monotonic() - start
        with (output / 'events.jsonl').open('a') as file:
            file.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    if smoke:
        evaluate_code(BASELINE, [TRAIN_SEEDS[0]], False)
        try:
            metrics = provider.smoke()
        except MessageError as error:
            event({'type': 'failed', 'error': str(error), **error.metrics})
            raise
        event({'type': 'connection_ok', **metrics})
        return

    stopped = None
    current = BASELINE
    blocked = False
    best = evaluate_code(current, TRAIN_SEEDS, blocked, trusted, replay=True)
    (output / 'candidate-000.py').write_text(current)
    save(output / 'baseline.json', best)
    feedback = {'current': {k: v for k, v in best.items() if k != 'replay'}}

    event({'type': 'baseline', 'blocked': blocked, 'score': best['mean_food']})
    for generation in range(1, args.generations + 1):
        if generation == args.obstacles_at:
            blocked = True
            (output / 'pre-change-champion.py').write_text(current)
            best = evaluate_code(current, TRAIN_SEEDS, blocked, trusted, replay=True)
            save(output / 'change-baseline.json', best)
            feedback = {'environment_changed': True, 'obstacles': True,
                        'current': {k: v for k, v in best.items() if k != 'replay'}}
            event({'type': 'environment_change', 'generation': generation,
                   'blocked': blocked, 'score': best['mean_food']})
        proposal = None
        evaluation_start = None
        try:
            proposal = provider.propose(current, feedback, generation)
            (output / f'candidate-{generation:03}.py').write_text(proposal.code)
            evaluation_start = time.monotonic()
            result = evaluate_code(proposal.code, TRAIN_SEEDS, blocked, trusted, replay=True)
            evaluation_seconds = time.monotonic() - evaluation_start
        except (ValueError, SyntaxError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            message = str(error) if isinstance(error, MessageError) else type(error).__name__
            metrics = error.metrics if isinstance(error, MessageError) else {}
            feedback = {'current_score': best['mean_food'], 'last_failure': message}
            event({'type': 'failed', 'generation': generation, 'blocked': blocked,
                   'error': message, 'best_score': best['mean_food'],
                   'request_seconds': proposal.request_seconds if proposal else None,
                   'usage': proposal.usage if proposal else {},
                   'evaluation_seconds': None if evaluation_start is None else time.monotonic() - evaluation_start,
                   **(proposal.metrics if proposal else metrics)})
            if isinstance(error, MessageError) and error.fatal:
                stopped = error
                break
            continue
        accepted = result['mean_food'] > best['mean_food']
        if accepted:
            current, best = proposal.code, result
        feedback = {'current_score': best['mean_food'],
                    'last_attempt': {k: v for k, v in result.items() if k != 'replay'},
                    'last_attempt_code': proposal.code, 'accepted': accepted, 'obstacles': blocked}
        save(output / f'evaluation-{generation:03}.json', result)
        event({'type': 'candidate', 'generation': generation, 'blocked': blocked,
               'accepted': accepted, 'score': result['mean_food'],
               'best_score': best['mean_food'], 'summary': proposal.summary,
               'request_seconds': proposal.request_seconds,
               'evaluation_seconds': evaluation_seconds, 'usage': proposal.usage, **proposal.metrics})
    (output / 'champion.py').write_text(current)
    # Final test cases never enter a subsequent proposal prompt.
    final = {'mock': trusted, 'blocked': blocked,
             'baseline': evaluate_code(BASELINE, TEST_SEEDS, blocked, trusted, replay=True),
             'champion': evaluate_code(current, TEST_SEEDS, blocked, trusted, replay=True)}
    frozen = output / 'pre-change-champion.py'
    if frozen.exists():
        final['pre_change_champion'] = evaluate_code(
            frozen.read_text(), TEST_SEEDS, blocked, trusted, replay=True)
    save(output / 'final.json', final)
    event({'type': 'stopped' if stopped else 'complete', 'baseline_test_score': final['baseline']['mean_food'],
           'champion_test_score': final['champion']['mean_food']})
    if stopped:
        raise stopped


def main():
    parser = argparse.ArgumentParser(description='Snake policy improvement experiment')
    parser.add_argument('--provider', default='mock', help='mock, naive, or a trusted module:function')
    parser.add_argument('--generations', type=int, default=6)
    parser.add_argument('--obstacles-at', type=int, default=4)
    parser.add_argument('--output', help='New directory for this run')
    parser.add_argument('--tier', choices=('standard', 'ultrafast'))
    parser.add_argument('--connection-smoke', action='store_true', help='One streaming OK check')
    parser.add_argument('--max-tokens', type=int, help='Output budget per request, including thinking')
    args = parser.parse_args()
    if args.output is None:
        kind = 'connection' if args.connection_smoke else 'demo'
        label = args.provider.replace(':', '-')
        args.output = f'runs/{label}-{kind}'
    if args.max_tokens is None:
        args.max_tokens = 256 if args.connection_smoke else 2048
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
