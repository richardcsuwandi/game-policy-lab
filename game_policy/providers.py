"""Handwritten fixtures and the streaming Naive Messages provider."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPException, HTTPSConnection
from importlib import import_module
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from game_policy.config import settings
from game_policy.messages import LIMIT, MessageError, read_message

from game_policy.policies import BASELINE, SAFE, SPACE

SYSTEM = '''Improve a Snake policy using measured evaluation feedback.
Return only JSON with two string fields: summary and code.
code must define exactly one choose_action(state) function returning up/right/down/left.
state contains width, height, snake (head first, [x,y] cells), food, obstacles, direction,
and score. Moving into the vacating tail is legal unless eating. Reversing is illegal.
The objective is mean food across games. Games end after 400 steps or 120 without food.
Use only basic Python, loops, comprehensions, and these builtins:
abs, all, any, bool, dict, enumerate, float, int, len, list, max, min, range, reversed,
set, sorted, sum, tuple, zip. The only allowed methods are .append() and .add().
No imports, nested functions, decorators, annotations, private names or introspection.
Keep code under 20,000 characters. Explain the proposed change in one short sentence.
'''


@dataclass
class Proposal:
    code: str
    summary: str
    request_seconds: float
    usage: dict
    metrics: dict = field(default_factory=dict)


class MockProvider:
    name = 'mock'

    def propose(self, current, feedback, generation):
        samples = [(SAFE, 'Bundled collision avoidance example'),
                   (SPACE, 'Bundled reachable-space heuristic example'),
                   (BASELINE, 'Bundled regression to exercise rejection')]
        code, summary = samples[(generation - 1) % len(samples)]
        return Proposal(code, summary, 0.0, {})


class NaiveProvider:
    name = 'naive'

    def __init__(self, tier=None, max_tokens=2048, root=Path(__file__).resolve().parent.parent,
                 *, stream_timeout=120, stream_byte_limit=LIMIT):
        config = settings(root)
        if config.get('NAIVE_CHAT_URL') or config.get('NAIVE_EXTRA_BODY'):
            raise ValueError('Remove NAIVE_CHAT_URL and NAIVE_EXTRA_BODY. Use NAIVE_API_BASE and NAIVE_SERVICE_TIER.')
        self.key = config.get('NAIVE_API_KEY', '')
        if not self.key or not self.key.isascii() or not self.key.isprintable() or any(
                c.isspace() for c in self.key):
            raise ValueError('Set a valid NAIVE_API_KEY in the project .env or shell.')
        self.model = config.get('NAIVE_MODEL', 'naive-n0.5-flash')
        self.tier = tier or config.get('NAIVE_SERVICE_TIER', 'ultrafast')
        if not re.fullmatch(r'[a-zA-Z0-9._-]{1,128}', self.model) or self.key in self.model:
            raise ValueError('Invalid NAIVE_MODEL.')
        if self.tier not in ('standard', 'ultrafast'):
            raise ValueError('NAIVE_SERVICE_TIER must be standard or ultrafast.')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('Set max-tokens to a positive integer.')
        if any(type(v) is not int or v < 1 for v in (stream_timeout, stream_byte_limit)):
            raise ValueError('Stream limits must be positive integers.')
        self.stream_timeout = stream_timeout
        self.stream_byte_limit = stream_byte_limit
        self.max_tokens = max_tokens
        base = urlsplit(config.get('NAIVE_API_BASE', 'https://api.naive.ai'))
        if (base.scheme != 'https' or not base.hostname or base.username is not None
                or base.path.strip('/') or base.query or base.fragment):
            raise ValueError('NAIVE_API_BASE must be an HTTPS origin without credentials or a path.')
        self.host, self.port = base.hostname, base.port

    def request(self, system, prompt):
        body = {'model': self.model, 'system': system, 'stream': True,
                'max_tokens': self.max_tokens, 'service_tier': self.tier,
                'messages': [{'role': 'user', 'content': prompt}]}
        metrics = {'request_start_utc': datetime.now(timezone.utc).isoformat(),
                   'first_event_seconds': None, 'first_thinking_seconds': None,
                   'first_answer_delta_seconds': None, 'completion_seconds': None,
                   'requested_tier': self.tier, 'served_tier': 'unverified', 'usage': {}}
        start = time.monotonic()
        def elapsed():
            return time.monotonic() - start

        # Direct HTTPS, no redirects, SDK retries or proxy credential forwarding.
        conn = HTTPSConnection(self.host, self.port, timeout=30)
        try:
            conn.request('POST', '/v1/messages', body=json.dumps(body).encode(), headers={
                'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json',
                'Accept': 'text/event-stream', 'anthropic-version': '2023-06-01'})
            with conn.getresponse() as response:
                request_id = response.getheader('request-id', '') or response.getheader('x-request-id', '')
                if re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', request_id) and self.key not in request_id:
                    metrics['request_id'] = request_id
                if response.status != 200:
                    raise MessageError.for_status(response.status)
                mime = response.getheader('Content-Type', '').split(';')[0].strip().lower()
                answer = read_message(response, mime, metrics, elapsed,
                                      timeout_seconds=self.stream_timeout, byte_limit=self.stream_byte_limit)
                if self.key in answer:
                    raise MessageError('Response contained credential material. Output discarded.')
        except MessageError as error:
            error.metrics = metrics
            raise
        except (OSError, HTTPException) as error:
            reason = 'timed out' if isinstance(error, TimeoutError) else 'failed'
            raise MessageError(f'Messages transport {reason}. Check connectivity before retrying.',
                               metrics=metrics) from None
        finally:
            if self.key in json.dumps(metrics['usage']):
                metrics['usage'].clear()
            metrics['request_seconds'] = elapsed()
            conn.close()
        return answer, metrics

    def smoke(self):
        answer, metrics = self.request('Reply only with OK.', 'Connection check. Reply OK.')
        if answer.strip() != 'OK':
            raise MessageError('Unexpected connection check answer.', metrics=metrics)
        return metrics

    def propose(self, current, feedback, generation, *, system=None):
        answer, metrics = self.request(SYSTEM if system is None else system, json.dumps({
            'current_policy': current, 'feedback': feedback, 'generation': generation}))
        if answer.startswith('```') and answer.endswith('```'):
            answer = '\n'.join(answer.splitlines()[1:-1])
        try:
            payload = json.loads(answer)
        except (ValueError, RecursionError):
            payload = None
        if not isinstance(payload, dict) or not all(isinstance(payload.get(k), str)
                and payload[k].strip() for k in ('code', 'summary')):
            raise MessageError('Expected JSON with nonempty summary and code strings.',
                               fatal=False, metrics=metrics)
        if any(self.key in payload[k] for k in ('code', 'summary')):
            raise MessageError('Response contained credential material. Output discarded.', metrics=metrics)
        return Proposal(payload['code'], payload['summary'][:500], metrics['request_seconds'],
                        metrics['usage'], metrics)


class FunctionProvider:
    """Connect a trusted host-side model callable to either game."""

    def __init__(self, name, max_tokens):
        module, separator, function = name.partition(':')
        if not separator or not all(part.isidentifier() for part in module.split('.')) or not function.isidentifier():
            raise ValueError('Use a provider of the form module:function.')
        try:
            self.generate = getattr(import_module(module), function)
        except (ImportError, AttributeError):
            raise ValueError('Could not import the provider callable.') from None
        if not callable(self.generate) or max_tokens < 1:
            raise ValueError('Set a callable provider and a positive output budget.')
        self.name = name
        self.max_tokens = max_tokens

    def propose(self, current, feedback, generation, *, system=SYSTEM):
        start = time.monotonic()
        try:
            result = self.generate(system=system, current=current, feedback=feedback,
                                   generation=generation, max_tokens=self.max_tokens)
        except Exception:
            raise MessageError('Custom provider failed. Check its host-side configuration.') from None
        if not isinstance(result, dict) or not all(isinstance(result.get(k), str) and result[k].strip()
                                                   for k in ('summary', 'code')):
            raise MessageError('Custom provider must return summary and code strings.', fatal=False)
        return Proposal(result['code'], result['summary'][:500], time.monotonic() - start, {})


def live_provider(name, *, max_tokens, tier=None):
    if name == 'naive':
        return NaiveProvider(tier=tier, max_tokens=max_tokens)
    if tier is not None:
        raise ValueError('--tier applies to the Naive adapter.')
    return FunctionProvider(name, max_tokens)
