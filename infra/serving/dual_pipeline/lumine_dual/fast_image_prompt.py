"""Exact fast paths for single-token image replacements and placeholder scans.

Unrecognized multimodal layouts use the original vLLM implementation. All paths
retain vLLM's count checks, content comparisons, and final prefix/media checks.
"""

def replace_image_tokens(prompt, contents, target):
    out = []
    cursor = 0
    for content in contents:
        try:
            start = prompt.index(target, cursor)
        except ValueError:
            return None
        out.extend(prompt[cursor:start])
        out.extend(content)
        cursor = start + 1
    out.extend(prompt[cursor:])
    return out


def locate_image_tokens(prompt, contents):
    positions = []
    cursor = 0
    for content in contents:
        if not content:
            return None
        while True:
            try:
                start = prompt.index(content[0], cursor)
            except ValueError:
                return positions
            end = start + len(content)
            if prompt[start:end] == content:
                positions.append(start)
                cursor = end
                break
            cursor = start + 1
    return positions


def install(processor, *, verify=False, verify_first=0):
    from vllm.multimodal.processing.processor import (
        _seq2tokens, PlaceholderFeaturesInfo, UpdateMode,
    )
    original_replace = processor._apply_token_matches
    original_find = processor._find_mm_placeholders
    tokenizer = processor.info.get_tokenizer()
    counters = dict(replacement_fast=0, placeholder_fast=0, verified=0, fallback=0)

    def contents_for(updates):
        if set(updates) != {'image'} or not updates['image']:
            return None
        if any(len(items) != 1 or items[0].content.is_embed is not None for items in updates['image']):
            return None
        return [_seq2tokens(tokenizer, items[0].content.full) for items in updates['image']]

    def replace(prompt, updates):
        contents = contents_for(updates)
        targets = []
        if contents is not None:
            for items in updates['image']:
                update = items[0]
                if update.mode != UpdateMode.REPLACE or not isinstance(update.target, (str, list)):
                    contents = None
                    break
                target = _seq2tokens(tokenizer, update.target)
                if len(target) != 1:
                    contents = None
                    break
                targets.append(target[0])
        if contents is None or len(set(targets)) != 1:
            counters['fallback'] += 1
            return original_replace(prompt, updates)
        expanded = replace_image_tokens(prompt, contents, targets[0])
        if expanded is None:
            counters['fallback'] += 1
            return original_replace(prompt, updates)
        result = expanded, {'image': [0] * len(contents)}
        counters['replacement_fast'] += 1
        if verify or counters['verified'] < verify_first:
            assert result == original_replace(prompt, updates), 'Image replacement differs from frozen renderer'
            counters['verified'] += 1
        return result

    def find(prompt, updates):
        contents = contents_for(updates)
        if contents is None or any(not content for content in contents):
            counters['fallback'] += 1
            return original_find(prompt, updates)
        positions = locate_image_tokens(prompt, contents)
        result = {'image': [PlaceholderFeaturesInfo('image', i, start, contents[i], None) for i, start in enumerate(positions)]} if positions else {}
        counters['placeholder_fast'] += 1
        if verify or counters['verified'] < verify_first:
            assert result == original_find(prompt, updates), 'Image placeholder differs from frozen renderer'
            counters['verified'] += 1
        return result

    processor._apply_token_matches = replace
    processor._find_mm_placeholders = find
    return counters
