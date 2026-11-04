"""Opt-in integration tests against the actual Docker daemon and worker image."""
import os
import subprocess
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from game_policy.executor import evaluate_code
from game_policy.policies import SAFE


@unittest.skipUnless(os.environ.get('GAME_POLICY_TEST_DOCKER') == '1', 'opt-in Docker integration')
class DockerTests(unittest.TestCase):
    def test_fixture_matches_local_result(self):
        self.assertEqual(evaluate_code(SAFE, [3, 7], True),
                         evaluate_code(SAFE, [3, 7], True, trusted_mock=True))

    def test_timeout_removes_container_and_no_credentials_enter_worker(self):
        token = uuid.uuid4()
        name = 'game-policy-' + token.hex
        code = 'def choose_action(state):\n    while True:\n        pass'
        with patch('game_policy.executor.uuid.uuid4', return_value=token), patch.dict(
                os.environ, {'NAIVE_API_KEY': 'docker-secret-sentinel'}), ThreadPoolExecutor() as pool:
            future = pool.submit(evaluate_code, code, [0], False)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                probe = subprocess.run(['docker', 'exec', name, 'python', '-c',
                    "import os; assert 'NAIVE_API_KEY' not in os.environ; "
                    "assert 'docker-secret-sentinel' not in os.environ.values(); "
                    "assert not os.path.exists('/app/.env')"], capture_output=True)
                if probe.returncode == 0:
                    break
                time.sleep(.15)
            else:
                self.fail('Could not verify the environment inside the running policy worker')
            with self.assertRaisesRegex(ValueError, 'deadline'):
                future.result(timeout=30)
        self.assertEqual(subprocess.check_output([
            'docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$']).strip(), b'')
