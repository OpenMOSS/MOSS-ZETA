"""One explicit response boundary for online, stateless and teacher-forced APIs."""
import json
from . import COMPACT, EOS, prefill, public_profile, reconstruct


def complete_body(body, profile, enabled):
    body['lumine_rendering'] = public_profile(profile)
    body['lumine_outputs'] = []
    raw_outputs = body.pop('lumine_engine_outputs', None)
    for i, choice in enumerate(body.get('choices', [])):
        raw = raw_outputs[i] if raw_outputs else dict(text=choice['message'].get('content') or '', token_ids=choice.get('token_ids') or [])
        record = reconstruct(raw['text'], raw['token_ids'], choice.get('finish_reason'), choice.get('stop_reason'), profile, enabled)
        body['lumine_outputs'].append(record)
        choice['message']['content'] = record['full_content']
    return body


async def complete_stream(source, profile, enabled):
    # A short tail is retained only to handle a terminal EOS split over SSE deltas.
    raw = ''; ids = []; emitted = 0; prefixed = False
    try:
        async for chunk in source:
            if not chunk.startswith('data: ') or chunk.startswith('data: [DONE]'):
                yield chunk
                continue
            body = json.loads(chunk[6:])
            body['lumine_rendering'] = public_profile(profile)
            for c in body.get('choices', []):
                if c.get('index', 0) != 0:
                    raise ValueError('Compact protocol supports n=1')
                delta = c.setdefault('delta', {})
                raw += delta.get('content') or ''
                ids.extend(c.get('token_ids') or [])
                prefix = '' if prefixed else prefill(profile, enabled)
                prefixed = True
                if c.get('finish_reason') is not None:
                    record = reconstruct(raw, ids, c['finish_reason'], c.get('stop_reason'), profile, enabled)
                    suffix = record['full_content'][len(prefill(profile, enabled)):]
                    delta['content'] = prefix + suffix[emitted:]
                    emitted = len(suffix)
                    body['lumine_outputs'] = [record]
                else:
                    safe = max(emitted, len(raw) - len(EOS))
                    delta['content'] = prefix + raw[emitted:safe]
                    emitted = safe
            yield 'data: ' + json.dumps(body, ensure_ascii=False) + '\n\n'
    finally:
        if hasattr(source, 'aclose'):
            await source.aclose()
