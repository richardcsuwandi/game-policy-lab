import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from game_policy.__main__ import run
from game_policy.executor import evaluate_code
from game_policy.game import TRAIN_SEEDS, TEST_SEEDS, evaluate, play
from game_policy.policies import BASELINE, SAFE, SPACE
from game_policy.policy_contract import load, validate
from game_policy.providers import NaiveProvider, Proposal


class GameTests(unittest.TestCase):
    def test_deterministic_replay(self):
        for blocked in (False, True):
            self.assertEqual(play(load(SAFE), 42, blocked, trace=True),
                             play(load(SAFE), 42, blocked, trace=True))

    def test_obstacles_do_not_overlap_start_or_food(self):
        for seed in TRAIN_SEEDS:
            state = play(load(SAFE), seed, True, trace=True)['frames'][0]
            self.assertNotIn(state['food'], state['obstacles'] + state['snake'])
            self.assertFalse(any(p in state['snake'] for p in state['obstacles']))

    def test_reverse_ends_episode(self):
        result = play(lambda state: 'left', 0)
        self.assertEqual(result['reason'], 'reverse')
        self.assertEqual(result['steps'], 1)

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            play(lambda state: 'teleport', 0)

    def test_observation_mutation_does_not_modify_world(self):
        def policy(state):
            state['snake'][0][0] = 9999
            state['obstacles'].append([5, 6])
            return 'right'
        self.assertEqual(play(policy, 0)['reason'], 'boundary')
        self.assertEqual(play(policy, 0)['steps'], 8)

    def test_fixtures_are_valid_and_selection_has_signal(self):
        scores = [evaluate(load(code), TRAIN_SEEDS)['mean_food']
                  for code in (BASELINE, SAFE, SPACE)]
        self.assertGreater(scores[1], scores[0])
        self.assertTrue(all(score >= 0 for score in scores))

    def test_test_seeds_are_disjoint(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(TEST_SEEDS))


class ContractTests(unittest.TestCase):
    def test_rejects_imports_introspection_and_top_level_code(self):
        for code in [
            'import os\ndef choose_action(state): return "up"',
            'def choose_action(state): return state.__class__',
            'def choose_action(state): return globals()',
            'def choose_action(state): return open("/etc/passwd")',
            '@print\ndef choose_action(state): return "up"',
        ]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate(code)

    def test_local_executor_cannot_run_new_code(self):
        with self.assertRaisesRegex(ValueError, 'exact bundled'):
            evaluate_code('def choose_action(state): return "up"', [0], False, True)


class ProviderTests(unittest.TestCase):
    def test_missing_key_fails_without_network(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            NaiveProvider(root=Path('/nonexistent-fixture-root'))

class ExperimentTests(unittest.TestCase):
    def test_selection_change_final_test_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            args = argparse.Namespace(provider='mock', generations=3, obstacles_at=2,
                                      output=str(output))
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['type'], 'complete')
            self.assertTrue(any(e['type'] == 'environment_change' for e in events))
            self.assertTrue(any(e.get('accepted') is True for e in events))
            self.assertFalse(events[-2]['accepted'])
            final = json.loads((output / 'final.json').read_text())
            self.assertTrue(final['mock'])
            self.assertEqual([e['seed'] for e in final['champion']['episodes']], list(TEST_SEEDS))
            self.assertIn('pre_change_champion', final)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_failed_candidate_does_not_replace_champion(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            args = argparse.Namespace(provider='mock', generations=1, obstacles_at=0,
                                      output=str(output))
            with patch('game_policy.__main__.MockProvider.propose', return_value=Proposal(
                    'import os', 'Invalid fixture', 0.0, {})), contextlib.redirect_stdout(io.StringIO()):
                run(args)
            self.assertEqual((output / 'champion.py').read_text(), BASELINE)
            events = (output / 'events.jsonl').read_text()
            self.assertIn('"type": "failed"', events)


if __name__ == '__main__':
    unittest.main()
