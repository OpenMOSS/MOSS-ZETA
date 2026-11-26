"""Immutable per-service rendering identity and generated/history token audit."""
import hashlib
import json
import os
from pathlib import Path


def profile():
    from lumine_protocol import load_profile, source_identity
    p = load_profile(os.environ.get("VLLM_LUMINE_TEMPLATE_VERSION", "training-turnwise-20261009"),
                     os.environ.get("VLLM_LUMINE_CHAT_TEMPLATE"),
                     os.environ.get("VLLM_LUMINE_TOKENIZER_IDENTITY", "unverified"),
                     os.environ.get("VLLM_LUMINE_IMAGE_PREPROCESSING", "raw"))
    p["source_identity"] = source_identity()
    return p


def public_profile(p):
    return {k: v for k, v in p.items() if k != "template"}


def flat_ids(value):
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one rendered sequence")
        value = value[0]
    return list(value)


def history_token_audit(tokenizer, template, assistant, generated, prefix_ids):
    # Render with a dummy user just as the training per-message helper does.
    kwargs = dict(chat_template=template, tokenize=True, add_generation_prompt=False, enable_thinking=True)
    user = {"role": "user", "content": ""}
    dummy = flat_ids(tokenizer.apply_chat_template([user], **kwargs))
    expected = flat_ids(tokenizer.apply_chat_template([user, {"role": "assistant", "content": assistant}], **kwargs))[len(dummy):]
    boundary = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    # The streaming engine appends the boundary on the next observation.
    generated = list(generated)
    stop = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if stop and generated[-len(stop):] == stop:
        generated = generated[:-len(stop)]
    actual = list(prefix_ids) + generated + boundary
    index = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
    return dict(equal=actual == expected, first_difference=None if actual == expected else index,
                generated_tokens=len(actual), history_tokens=len(expected))
