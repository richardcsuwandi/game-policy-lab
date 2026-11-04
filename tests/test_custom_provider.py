import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from game_policy.messages import MessageError
from game_policy.providers import SYSTEM, live_provider


class CustomProviderTests(unittest.TestCase):
    def setUp(self):
        self.module = ModuleType('local_model')
        self.module.generate = Mock(return_value={'summary': 'edit', 'code': 'def choose_action(state): return "up"'})
        modules = patch.dict('sys.modules', {'local_model': self.module})
        modules.start()
        self.addCleanup(modules.stop)
        self.provider = live_provider('local_model:generate', max_tokens=128)

    def test_game_prompt_and_feedback_pass_through(self):
        feedback = {'current_score': 2}
        proposal = self.provider.propose('old code', feedback, 2)
        self.module.generate.assert_called_with(system=SYSTEM, current='old code', feedback=feedback,
                                                generation=2, max_tokens=128)
        self.assertEqual(proposal.code, self.module.generate.return_value['code'])
        self.provider.propose('paddle code', feedback, 1, system='Breakout rules')
        self.assertEqual(self.module.generate.call_args.kwargs['system'], 'Breakout rules')

    def test_invalid_result_and_private_error(self):
        self.module.generate.return_value = {}
        with self.assertRaises(MessageError) as invalid:
            self.provider.propose('', {}, 1)
        self.assertFalse(invalid.exception.fatal)
        self.module.generate.side_effect = RuntimeError('private-token')
        with self.assertRaises(MessageError) as failed:
            self.provider.propose('', {}, 2)
        self.assertTrue(failed.exception.fatal)
        self.assertNotIn('private-token', str(failed.exception))
