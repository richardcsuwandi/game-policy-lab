import unittest
from game_policy.breakout import FOLLOW, PREDICT, play
from game_policy.policy_contract import load


class BreakoutTests(unittest.TestCase):
    def test_seed_pair_and_replay(self):
        for seed in (0, 12, 1011):
            a, b = (play(load(code), seed, True) for code in (FOLLOW, PREDICT))
            self.assertEqual(a['frames'][0], b['frames'][0])
            self.assertEqual(b, play(load(PREDICT), seed, True))
            for r in (a, b):
                masks = [frame[-1] for frame in r['frames']]
                self.assertEqual(r['bricks'], 24-masks[-1].bit_count())
                self.assertTrue(all((m & n) == n for m,n in zip(masks,masks[1:])))
                self.assertEqual(len(masks), r['steps'] + 1)
                self.assertTrue(all(0 <= x < 48 and 0 <= y <= 32 for x,y,*_ in r['frames']))

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            play(lambda state: True, 0)
