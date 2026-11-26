"""Separate a session's APC namespace from exact preparation receipt identities."""
import re

def engine_cache_salt(cache_key, scope):
    if scope == 'request':
        return cache_key
    if scope != 'session':
        raise ValueError('Unknown prefix_cache_scope')
    # Logical receipts remain unique for each exact prepared prompt. vLLM also
    # hashes prompt tokens and multimodal identities before reusing any blocks.
    # A fallback UUID without this namespace keeps an isolated cold cache.
    match = re.fullmatch(r'([0-9a-f]{32}):[0-9a-f]{32}', cache_key)
    return match.group(1) if match else cache_key
