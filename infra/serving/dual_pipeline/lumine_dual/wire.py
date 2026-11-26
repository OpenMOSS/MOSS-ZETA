"""The checkpoint-bound think-memory-minimap-compact-v1 wire contract."""
import math

from lumine_protocol import load_profile, source_identity


def generation_options(body):
    """Validate and forward sampling; never silently force greedy decoding."""
    allowed = {'model', 'messages', 'stream', 'temperature', 'top_p', 'top_k',
               'min_p', 'max_tokens', 'max_completion_tokens', 'seed', 'stop',
               'presence_penalty', 'frequency_penalty', 'repetition_penalty',
               'chat_template_kwargs', 'chat_template', 'n', 'lumine_state',
               'lumine_session_id', 'lumine_request_id', 'lumine_format_version', 'lumine_window',
               'lumine_execution_mode', 'lumine_execution_receipts', 'lumine_history_mode', 'lumine_stream_version',
               'await_next_prepare', 'update_version'}
    if type(body.get('await_next_prepare', False)) is not bool:
        raise ValueError('await_next_prepare must be boolean')
    if 'update_version' in body and (type(body['update_version']) not in (str, int) or isinstance(body['update_version'], bool) or str(body['update_version']) == ''):
        raise ValueError('update_version must be a nonempty string or integer')
    unknown = set(body) - allowed
    if unknown:
        raise ValueError('Unsupported request fields: ' + ', '.join(sorted(unknown)))
    if type(body.get('stream', False)) is not bool or body.get('n', 1) != 1:
        raise ValueError('stream must be boolean and n=1')
    if body.get('stream') and (body.get('lumine_stream_version') != 'policy-state-stream-v1' or body.get('lumine_history_mode') != 'generated_dialogue_v1' or body.get('lumine_execution_mode')):
        raise ValueError('Streaming requires generated_dialogue_v1 and policy-state-stream-v1')
    if body.get('lumine_history_mode') not in (None, 'generated_dialogue_v1'):
        raise ValueError('Unknown history mode')
    kw = body.get('chat_template_kwargs', {})
    if not isinstance(kw, dict) or set(kw) - {'enable_thinking'}:
        raise ValueError('Only enable_thinking is supported in chat_template_kwargs')
    enabled = kw.get('enable_thinking', True)
    if type(enabled) is not bool:
        raise ValueError('enable_thinking must be boolean')
    if not enabled:
        raise ValueError('This checkpoint requires enable_thinking=true')
    count = body.get('max_completion_tokens', body.get('max_tokens', 128))
    if count is not None and (type(count) is not int or not 1 <= count <= 4096):
        raise ValueError('max_tokens must be null (remaining model context) or 1..4096')
    if 'max_tokens' in body and 'max_completion_tokens' in body and body['max_tokens'] != count:
        raise ValueError('Conflicting token limits')
    result = dict(enable_thinking=enabled, max_tokens=count)
    specs = {'temperature': (0., 0, 2), 'top_p': (1., 0, 1),
             'min_p': (0., 0, 1), 'presence_penalty': (0., -2, 2),
             'frequency_penalty': (0., -2, 2), 'repetition_penalty': (1., 0, 2)}
    for key, (default, low, high) in specs.items():
        value = body.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid ' + key)
        if key in ('top_p', 'repetition_penalty') and value == 0:
            raise ValueError(key + ' must be positive')
        result[key] = value
    k = body.get('top_k', -1)
    if type(k) is not int or (k != -1 and k < 1):
        raise ValueError('top_k must be -1 or positive')
    result['top_k'] = k
    seed = body.get('seed')
    if seed is not None and type(seed) is not int:
        raise ValueError('seed must be an integer')
    result['seed'] = seed
    stop = body.get('stop', ['</a>'])
    if stop is None:
        stop = ['</a>']
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 16 or any(not isinstance(x, str) or not x for x in stop):
        raise ValueError('Invalid stop')
    result['stop'] = stop
    return result


def rendering_profile(template_path, tokenizer_identity, image_preprocessing):
    profile = load_profile('think-memory-minimap-compact-v1', template_path,
                           tokenizer_identity, image_preprocessing)
    profile['source_identity'] = source_identity()
    return profile
