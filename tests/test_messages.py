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
from game_policy.messages import MessageError
from game_policy.policies import BASELINE
from game_policy.providers import NaiveProvider


KEY = 'not-a-real-key-test'
ANSWER = json.dumps({'summary': 'Démo', 'code': BASELINE}, ensure_ascii=False)


class Response(io.BytesIO):
    def __init__(self, body, mime='application/json', status=200, fragment=11):
        super().__init__(body)
        self.mime, self.status, self.fragment = mime, status, fragment

    def getheader(self, name, default):
        return self.mime

    def read1(self, n):
        return super().read1(min(n, self.fragment))


def message(text=ANSWER, reason='end_turn'):
    return Response(json.dumps({'content': [{'type': 'thinking', 'thinking': KEY},
        {'type': 'text', 'text': text[:5]}, {'type': 'text', 'text': text[5:]}],
        'stop_reason': reason, 'usage': {'output_tokens': 7}}).encode())


def sequence(reason='end_turn'):
    return [
        dict(type='ping'),
        dict(type='message_start', message={'usage': {'input_tokens': 3, 'output_tokens': 1,
            'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 4,
            'cache_creation': {'ephemeral_5m_input_tokens': 4}, 'other': KEY}}),
        dict(type='content_block_start', index=0, content_block={'type': 'thinking'}),
        dict(type='content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': KEY}),
        dict(type='content_block_stop', index=0),
        dict(type='content_block_start', index=1, content_block={'type': 'text', 'text': ''}),
        dict(type='content_block_delta', index=1, delta={'type': 'text_delta', 'text': ANSWER[:17]}),
        dict(type='content_block_delta', index=1, delta={'type': 'text_delta', 'text': ANSWER[17:]}),
        dict(type='content_block_stop', index=1),
        dict(type='message_delta', delta={'stop_reason': reason},
             usage={'output_tokens': 17, 'service_tier': 'standard'}),
        dict(type='message_stop')]


def stream(events):
    # Split UTF-8, CRLF, line prefixes and JSON at arbitrary byte boundaries.
    raw = ': keepalive\r\n\r\n'
    for event in events:
        body = json.dumps(event, ensure_ascii=False)
        raw += f"event: {event['type']}\r\ndata: {body[:-1]}\r\ndata: }}\r\n\r\n"
    return Response(raw.encode(), 'text/event-stream; charset=utf-8', fragment=1)


class MessagesTests(unittest.TestCase):
    def setUp(self):
        self.root_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.root_tmp.cleanup)
        self.root = Path(self.root_tmp.name)
        env = patch.dict(os.environ, {'NAIVE_API_KEY': KEY}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        transport = patch('game_policy.providers.HTTPSConnection')
        self.transport = transport.start()
        self.addCleanup(transport.stop)
        self.conn = self.transport.return_value
        self.provider = NaiveProvider(root=self.root, max_tokens=384)

    def propose(self, response):
        self.conn.getresponse.return_value = response
        return self.provider.propose(BASELINE, {}, 1)

    def test_messages_body_and_fragmented_stream(self):
        proposal = self.propose(stream(sequence()))
        self.transport.assert_called_once_with('api.naive.ai', None, timeout=30)
        call = self.conn.request.call_args
        self.assertEqual(call.args, ('POST', '/v1/messages'))
        self.assertEqual(call.kwargs['headers']['Authorization'], 'Bearer ' + KEY)
        self.assertEqual(call.kwargs['headers']['anthropic-version'], '2023-06-01')
        body = json.loads(call.kwargs['body'])
        self.assertEqual(body['model'], 'naive-n0.5-flash')
        self.assertEqual(body['max_tokens'], 384)
        self.assertEqual(body['service_tier'], 'ultrafast')
        self.assertTrue(body['stream'])
        self.assertIn('system', body)
        self.assertEqual([m['role'] for m in body['messages']], ['user'])
        self.assertEqual((proposal.code, proposal.summary), (BASELINE, 'Démo'))
        m = proposal.metrics
        times = [m[k] for k in ('first_event_seconds','first_thinking_seconds',
            'first_answer_delta_seconds','completion_seconds','request_seconds')]
        self.assertEqual(times, sorted(times))
        self.assertEqual(m['served_tier'], 'unverified')
        self.assertEqual(m['reported_tier_usage'], 'standard')
        self.assertEqual(m['usage'], {'input_tokens': 3, 'output_tokens': 17,
            'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 4,
            'cache_creation': {'ephemeral_5m_input_tokens': 4}})
        self.assertNotIn(KEY, repr(proposal))

    def test_explicit_budget_above_smoke_cap(self):
        self.provider = NaiveProvider(root=self.root, max_tokens=16384)
        self.propose(message())
        self.assertEqual(json.loads(self.conn.request.call_args.kwargs['body'])['max_tokens'], 16384)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            NaiveProvider(root=self.root, max_tokens=0)

    def test_custom_stream_limits_reach_parser(self):
        self.provider = NaiveProvider(root=self.root, stream_timeout=600, stream_byte_limit=4096)
        # The response arrives after the usual 120-second ceiling.
        with patch('game_policy.providers.time.monotonic', side_effect=[0] + [150]*200):
            self.assertEqual(self.propose(message()).code, BASELINE)
        with patch('game_policy.providers.time.monotonic', side_effect=[0, 601, 602]):
            with self.assertRaisesRegex(MessageError, 'time budget'):
                self.propose(message())
        self.provider = NaiveProvider(root=self.root, stream_byte_limit=5)
        with self.assertRaisesRegex(MessageError, 'size limit'):
            self.propose(message())

    def test_complete_json_concatenates_only_text(self):
        proposal = self.propose(message())
        self.assertEqual(proposal.code, BASELINE)
        self.assertIsNone(proposal.metrics['first_event_seconds'])
        self.assertIsNone(proposal.metrics['first_answer_delta_seconds'])

    def test_missing_stop_and_truncation(self):
        for response, pattern in [(stream(sequence()[:-1]), 'message_stop'),
                (stream(sequence('max_tokens')), 'max_tokens'), (message(reason='max_tokens'), 'max_tokens')]:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(MessageError, pattern) as exc:
                self.propose(response)
            self.assertGreater(exc.exception.metrics['usage']['output_tokens'], 0)

    def test_stream_error_discards_partial_answer(self):
        response = stream(sequence()[:-2] + [dict(type='error',
            error={'type': 'rate_limit_error', 'message': KEY})])
        with self.assertRaisesRegex(MessageError, '429') as exc:
            self.propose(response)
        self.assertTrue(exc.exception.fatal)
        self.assertNotIn(KEY, str(exc.exception) + repr(exc.exception.metrics))

    def test_bad_payloads_and_echoed_secret(self):
        for answer in ('', '[]', '{bad', '{"code":"x"}', json.dumps({'code': KEY, 'summary': 'x'})):
            with self.subTest(), self.assertRaises(MessageError) as exc:
                self.propose(message(answer))
            self.assertNotIn(KEY, str(exc.exception))

    def test_malformed_event(self):
        with self.assertRaisesRegex(MessageError, 'Malformed'):
            self.propose(Response(b'event: message_start\ndata: {bad}\n\n', 'text/event-stream'))

    def test_fatal_status_codes_and_timeout_no_retry(self):
        for status in (302, 401, 402, 403, 429, 500):
            with self.subTest(status=status), self.assertRaisesRegex(MessageError, str(status)) as exc:
                self.propose(Response(KEY.encode(), status=status))
            self.assertTrue(exc.exception.fatal)
            self.assertNotIn(KEY, str(exc.exception))
        self.conn.getresponse.side_effect = TimeoutError(KEY)
        with self.assertRaisesRegex(MessageError, 'timed out'):
            self.provider.smoke()
        self.assertEqual(self.conn.request.call_count, 7)

    def test_local_config_precedence_and_legacy_rejection(self):
        (self.root / '.env').write_text('NAIVE_API_KEY=file-only\nNAIVE_SERVICE_TIER=standard\n'
                                       'NAIVE_API_BASE=https://example.invalid\n')
        config = NaiveProvider(root=self.root)
        self.assertEqual(config.key, KEY)
        self.assertEqual(config.host, 'example.invalid')
        self.assertEqual(config.tier, 'standard')
        self.assertEqual(NaiveProvider(root=self.root, tier='ultrafast').tier, 'ultrafast')
        del os.environ['NAIVE_API_KEY']
        self.assertEqual(NaiveProvider(root=self.root).key, 'file-only')
        self.assertNotIn('NAIVE_API_KEY', os.environ)
        os.environ['NAIVE_CHAT_URL'] = KEY
        with self.assertRaisesRegex(ValueError, 'Remove NAIVE_CHAT_URL'):
            NaiveProvider(root=self.root)

    def test_fatal_run_saves_incumbent_and_never_reissues(self):
        output = self.root / 'run'
        args = argparse.Namespace(output=output, provider='naive', generations=2, obstacles_at=0,
                                  tier='ultrafast', max_tokens=384)
        self.conn.getresponse.return_value = Response(KEY.encode(), status=429)
        log = io.StringIO()
        with patch('game_policy.__main__.live_provider', return_value=self.provider), patch(
                'game_policy.__main__.evaluate_code', return_value={'mean_food': 0}), \
                contextlib.redirect_stdout(log), self.assertRaisesRegex(MessageError, '429'):
            run(args)
        self.assertEqual(self.conn.request.call_count, 1)
        self.assertEqual((output / 'champion.py').read_text(), BASELINE)
        self.assertNotIn(KEY, log.getvalue() + ''.join(p.read_text() for p in output.iterdir()))
        self.assertFalse(json.loads((output / 'final.json').read_text())['mock'])

    def test_smoke_preflight_and_one_request(self):
        args = argparse.Namespace(output=self.root / 'run', provider='naive', connection_smoke=True,
                                  generations=1, obstacles_at=0, tier='ultrafast', max_tokens=256)
        self.conn.getresponse.return_value = message('OK')
        with patch('game_policy.__main__.live_provider', return_value=self.provider), patch(
                'game_policy.__main__.evaluate_code', side_effect=OSError('Docker unavailable')), self.assertRaises(OSError):
            run(args)
        self.conn.request.assert_not_called()
        args.output = self.root / 'ready'
        with patch('game_policy.__main__.live_provider', return_value=self.provider), patch(
                'game_policy.__main__.evaluate_code'), contextlib.redirect_stdout(io.StringIO()):
            run(args)
        self.conn.request.assert_called_once()
        self.assertEqual(json.loads((args.output / 'events.jsonl').read_text())['type'], 'connection_ok')
