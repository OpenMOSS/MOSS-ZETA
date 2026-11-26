"""Versioned, shared deployment/evaluation rendering and lossless output contract."""
import hashlib
import json
import re
from pathlib import Path

COMPACT = 'think-memory-minimap-compact-v1'
LEGACY = 'training-turnwise-20261009'
ROLE_HEAD = '<|im_start|>assistant\n'
EOS = '<|im_end|>'
PREFILLS = {COMPACT: {True: '<think>', False: '<think></think>'},
            LEGACY: {True: '<think>\n', False: '<think>\n\n</think>\n\n'}}


def prefill(profile, enable_thinking=False):
    if type(enable_thinking) is not bool:
        raise ValueError('enable_thinking must be boolean')
    return profile['generation_prefixes']['enabled' if enable_thinking else 'disabled']


def load_profile(version=COMPACT, template_path=None, tokenizer_identity='unverified', image_preprocessing=None):
    if version not in PREFILLS:
        raise ValueError('Unknown rendering version: ' + version)
    path = Path(__file__).with_name(version + '.jinja')
    template = path.read_text()
    if template_path is not None and Path(template_path).read_bytes() != path.read_bytes():
        raise ValueError('Template bytes differ from selected version asset')
    from jinja2 import Environment
    renderer = Environment().from_string(template)
    for enabled, prefix in PREFILLS[version].items():
        kw = dict(messages=[dict(role='user', content='')], enable_thinking=enabled)
        base = renderer.render(**kw, add_generation_prompt=False)
        if renderer.render(**kw, add_generation_prompt=True) != base + ROLE_HEAD + prefix:
            raise ValueError('Template/prefill mismatch')
    return dict(version=version, template=template, template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                tokenizer_identity=tokenizer_identity, image_preprocessing=image_preprocessing,
                generation_prefixes=dict(enabled=PREFILLS[version][True], disabled=PREFILLS[version][False]),
                generation_prefix=PREFILLS[version][True], response_format='full' if version == COMPACT else 'suffix',
                assistant_rule='training-preserve-think-turnwise' if version == COMPACT else 'legacy-reformatted')


def public_profile(profile):
    return {k: v for k, v in profile.items() if k != 'template'}


def protocol_errors(content, version=COMPACT, enable_thinking=False):
    if version == LEGACY:
        return []  # Legacy scoring remains in actions.py.
    if version != COMPACT:
        raise ValueError('Unknown protocol version')
    from lumine_dual.protocol import parse_action
    errors = []
    try: parse_action(content, 'think_memory_minimap_compact_v1')
    except (ValueError,TypeError): errors.append('invalid_assistant_envelope')
    if not enable_thinking: errors.append('thinking_required_by_checkpoint')
    if EOS in content or '<|im_start|>' in content:
        errors.append('unexpected_message_boundary')
    return errors


def reconstruct(raw_suffix, token_ids, finish_reason, stop_reason, profile, enable_thinking=False):
    """Input is always engine suffix. Never infer the format from generated text."""
    if not isinstance(raw_suffix, str):
        raise TypeError('raw suffix must be a string')
    body = raw_suffix
    terminal_eos = finish_reason == 'stop' and body.endswith(EOS)
    if terminal_eos:
        body = body[:-len(EOS)]
    full = prefill(profile, enable_thinking) + body
    errors = protocol_errors(full, profile['version'], enable_thinking)
    truncated = finish_reason == 'length'
    if truncated:
        errors.append('truncated')
    if finish_reason not in ('stop', 'length'):
        errors.append('unexpected_finish_reason')
    return dict(raw_suffix=raw_suffix, raw_token_ids=list(token_ids), full_content=full,
                prefilled_content=prefill(profile, enable_thinking), enable_thinking=enable_thinking,
                finish_reason=finish_reason, stop_reason=stop_reason, truncated=truncated,
                terminal_eos_removed=terminal_eos, protocol_version=profile['version'],
                protocol_errors=errors, protocol_valid=not errors)


def media_identity(prompt):
    result = []
    for mod, slots in sorted(prompt.get('mm_placeholders', {}).items()):
        hashes = prompt['mm_hashes'][mod]
        if len(slots) != len(hashes):
            raise ValueError('media hash/position mismatch')
        result.extend((mod, p.offset, p.length, h) for p, h in zip(slots, hashes))
    return sorted(result, key=lambda x: (x[1], x[0]))


def can_continue(tokens, media, previous_tokens, previous_media, contiguous=True):
    if not contiguous or not previous_tokens or len(tokens) <= len(previous_tokens):
        return False
    if tokens[:len(previous_tokens)] != previous_tokens:
        return False
    prefix_media = [m for m in media if m[1] < len(previous_tokens)]
    return prefix_media == previous_media and all(m[1] + m[2] <= len(previous_tokens) for m in prefix_media)


def source_identity():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.suffix in ('.py', '.jinja')}
