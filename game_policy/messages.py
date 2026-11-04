"""Parse Anthropic Messages and retain only answer text and safe telemetry."""
import json
import re

LIMIT = 2 * 1024 * 1024


class MessageError(ValueError):
    def __init__(self, message, *, fatal=True, metrics=None):
        super().__init__(message)
        self.fatal = fatal
        self.metrics = metrics or {}

    @classmethod
    def for_status(cls, status):
        hint = {
            401: 'Check NAIVE_API_KEY and whether the key is active.',
            402: 'Check balance and quota before another request.',
            403: 'Check account status and tier entitlement.',
            429: 'Check quota, balance and concurrency. Wait before retrying a rate limit.',
        }.get(status, 'Check the endpoint, model and service status before retrying.')
        return cls(f'Messages HTTP {status}. {hint}')


def object_data(raw):
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        value = None
    if not isinstance(value, dict):
        raise MessageError('Malformed Messages response.')
    return value


def report(data, metrics):
    usage = data.get('usage') or {}
    if not isinstance(usage, dict):
        raise MessageError('Malformed Messages usage.')
    # Only numeric token counters, including nested cache-creation breakdowns.
    for key, value in usage.items():
        if re.fullmatch(r'[a-z_]{1,60}_tokens', key) and type(value) is int and value >= 0:
            metrics['usage'][key] = value
        if key == 'cache_creation' and isinstance(value, dict):
            metrics['usage'].setdefault(key, {}).update({
                k: v for k, v in value.items() if re.fullmatch(r'[a-z0-9_]{1,60}_tokens', k)
                and type(v) is int and v >= 0})
    for source, value in [('message', data.get('service_tier')), ('usage', usage.get('service_tier'))]:
        if value in ('standard', 'ultrafast'):
            metrics['reported_tier_' + source] = value


def chunks(response, elapsed, timeout_seconds, byte_limit):
    size = 0
    while True:
        if elapsed() >= timeout_seconds:
            raise MessageError('Messages request exceeded its time budget.')
        chunk = response.read1(16384)
        if not chunk:
            break
        size += len(chunk)
        if size > byte_limit:
            raise MessageError('Messages response exceeded its size limit.')
        yield chunk


def events(chunks):
    # LF, CRLF and CR are valid SSE line endings, including split CRLF packets.
    line = bytearray()
    skip_lf = False
    data = []
    event = ''
    for chunk in chunks:
        for byte in chunk:
            if skip_lf and byte == 10:
                skip_lf = False
                continue
            skip_lf = byte == 13
            if byte not in (10, 13):
                line.append(byte)
                continue
            if line:
                name, _, value = line.partition(b':')
                if value.startswith(b' '):
                    value = value[1:]
                if name == b'data':
                    data.append(bytes(value))
                elif name == b'event':
                    event = value.decode('ascii', errors='replace')
                line.clear()
            else:
                if data:
                    yield event, object_data(b'\n'.join(data))
                data, event = [], ''


def completed(reason, parts):
    if reason == 'max_tokens':
        raise MessageError('Messages output reached max_tokens. Candidate discarded.', fatal=False)
    if reason != 'end_turn':
        raise MessageError('Messages output lacks a normal end_turn.')
    answer = ''.join(parts).strip()
    if not answer:
        raise MessageError('Messages returned no answer text.', fatal=False)
    return answer


def check_error(data):
    if data.get('type') == 'error':
        detail = data.get('error')
        code = detail.get('type') if isinstance(detail, dict) else None
        status = {'authentication_error': 401, 'permission_error': 403, 'rate_limit_error': 429,
                  'billing_error': 402, 'insufficient_quota': 402}.get(code) if isinstance(code, str) else None
        if status:
            raise MessageError.for_status(status)
        raise MessageError('Messages stream returned a provider error. Check account and service status.')


def read_message(response, mime, metrics, elapsed, *, timeout_seconds=120, byte_limit=LIMIT):
    packets = chunks(response, elapsed, timeout_seconds, byte_limit)
    parts = []
    if mime == 'application/json':
        data = object_data(b''.join(packets))
        metrics['completion_seconds'] = elapsed()
        check_error(data)
        report(data, metrics)
        blocks = data.get('content')
        if not isinstance(blocks, list):
            raise MessageError('Messages content must be an array.')
        for block in blocks:
            if not isinstance(block, dict):
                raise MessageError('Malformed Messages block.')
            if block.get('type') == 'text':
                if not isinstance(block.get('text'), str):
                    raise MessageError('Malformed Messages text.')
                parts.append(block['text'])
        return completed(data.get('stop_reason'), parts)
    if mime != 'text/event-stream':
        raise MessageError('Expected an SSE stream or Messages JSON.')

    started = False
    active = {}
    reason = None
    for label, data in events(packets):
        now = elapsed()
        if metrics['first_event_seconds'] is None:
            metrics['first_event_seconds'] = now
        kind = data.get('type')
        if not isinstance(kind, str) or (label and label != kind):
            raise MessageError('Malformed SSE event type.')
        check_error(data)
        if kind == 'ping':
            continue
        if kind == 'message_start':
            if started or not isinstance(data.get('message'), dict):
                raise MessageError('Malformed message_start.')
            started = True
            report(data['message'], metrics)
        elif not started:
            raise MessageError('Missing message_start.')
        elif kind in ('content_block_start', 'content_block_delta', 'content_block_stop'):
            index = data.get('index')
            if type(index) is not int or index < 0:
                raise MessageError('Malformed content block index.')
            if kind == 'content_block_start':
                block = data.get('content_block')
                if index in active or not isinstance(block, dict):
                    raise MessageError('Malformed content block start.')
                active[index] = block.get('type')
                if active[index] == 'text':
                    initial = block.get('text', '')
                    if not isinstance(initial, str):
                        raise MessageError('Malformed initial text.')
                    parts.append(initial)
            elif index not in active:
                raise MessageError('Content block was never opened.')
            elif kind == 'content_block_stop':
                del active[index]
            else:
                delta = data.get('delta')
                if not isinstance(delta, dict):
                    raise MessageError('Malformed content block delta.')
                if delta.get('type') == 'text_delta':
                    text = delta.get('text')
                    if active[index] != 'text' or not isinstance(text, str):
                        raise MessageError('Malformed answer delta.')
                    if text and metrics['first_answer_delta_seconds'] is None:
                        metrics['first_answer_delta_seconds'] = now
                    parts.append(text)
                elif active[index] == 'thinking' and delta.get('type') == 'thinking_delta':
                    if delta.get('thinking') and metrics['first_thinking_seconds'] is None:
                        metrics['first_thinking_seconds'] = now
        elif kind == 'message_delta':
            if not isinstance(data.get('delta'), dict):
                raise MessageError('Malformed message_delta.')
            reason = data['delta'].get('stop_reason', reason)
            report(data, metrics)
        elif kind == 'message_stop':
            if active:
                raise MessageError('Unclosed content block at message_stop.')
            metrics['completion_seconds'] = now
            return completed(reason, parts)
    raise MessageError('Stream ended before message_stop.')
