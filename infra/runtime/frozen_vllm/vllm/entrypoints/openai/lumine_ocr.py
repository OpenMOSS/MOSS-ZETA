# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain inline training OCR without moving images before their first prefill.

The current layout is ``user: <screen_ocr>...</screen_ocr> + image``;
legacy image-first messages are also supported without reordering their input.
After visual eviction, cached user delimiters and OCR spans survive as individual
messages. This preserves all S/D events and their order, but deliberately does
not rewrite them into the training packer's single archive envelope: rewriting
that prefix would require a rebuild, not a lossless reuse of its old KV.
"""

import re
from collections.abc import Mapping
from typing import Any

OCR_OPEN = "<screen_ocr>"
OCR_CLOSE = "</screen_ocr>"
OCR_ARCHIVE_HEADER = "此前已移除图片和动作的 OCR 历史（按帧序）："
OCR_ARCHIVE_HEADER_NO_FRAME = "此前已移除图片和动作的 OCR 历史（按出现顺序）："
OCR_ARCHIVE_HEADERS = (OCR_ARCHIVE_HEADER, OCR_ARCHIVE_HEADER_NO_FRAME)
EMPTY_OCR = "<screen_ocr>本轮无新增OCR</screen_ocr>"
_BLOCK = re.compile(r"<screen_ocr>.*?</screen_ocr>", re.DOTALL)


def inline_ocr_blocks(message: Mapping[str, Any]) -> list[str]:
    """Read retained user OCR events, omitting explicit no-update envelopes.

    The original image message is unchanged for prefill. Empty envelopes carry
    no state update and need not survive image eviction; empty S and field
    clears are still real events and must be retained.
    """
    if message.get("role") != "user":
        return []
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") in ("text", "input_text")
        )
    else:
        return []
    blocks = _BLOCK.findall(text)
    if text.count(OCR_OPEN) != len(blocks) or text.count(OCR_CLOSE) != len(blocks):
        raise ValueError("inline OCR must have balanced, non-nested screen_ocr markers")
    return [
        block
        for block in blocks
        if block != EMPTY_OCR and block[len(OCR_OPEN) : -len(OCR_CLOSE)].strip()
    ]


def inline_ocr_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """OCR-only cold-start history for an evicted image message."""
    blocks = inline_ocr_blocks(message)
    return {"role": "user", "content": "\n".join(blocks)} if blocks else None


def coalesce_ocr_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the training packer's single archive envelope at cold start only.

    There is no existing KV to invalidate at cold start. Concatenate event
    bodies verbatim, including an existing archive's events. Legacy free-form
    OCR is left in its original messages because it has no S/D event format.
    Do not call this on a live session's cached messages.
    """
    bodies = []
    styles = set()
    opening, closing = OCR_OPEN + "\n", "\n" + OCR_CLOSE
    for message in messages:
        blocks = inline_ocr_blocks(message)
        if not blocks:
            return messages
        for block in blocks:
            if not block.startswith(opening) or not block.endswith(closing):
                return messages
            body = block[len(opening) : -len(closing)]
            first_line = body.split("\n", 1)[0]
            if re.fullmatch(r"(?:[1-9][0-9]* )?[SD]", first_line) is None:
                return messages
            styles.add("numbered" if " " in first_line else "plain")
            bodies.append(body)
    if not bodies or len(styles) != 1:
        return messages
    header = OCR_ARCHIVE_HEADER if "numbered" in styles else OCR_ARCHIVE_HEADER_NO_FRAME
    return [
        {
            "role": "user",
            "content": header + "\n" + opening + "\n".join(bodies) + closing,
        }
    ]


def _find(tokens: list[int], needle: list[int], start: int, end: int) -> int:
    if not needle:
        raise ValueError("OCR retention requires non-empty chat/marker token sequences")
    for index in range(start, end - len(needle) + 1):
        if tokens[index : index + len(needle)] == needle:
            return index
    raise ValueError(
        "OCR retention could not locate a marker in the rendered user message"
    )


def inline_ocr_token_ranges(
    tokens: list[int],
    user_start: int,
    limit: int,
    tokenizer: Any,
    blocks: list[str],
) -> list[tuple[int, int]]:
    """Select chat delimiters plus OCR, excluding image and assistant tokens.

    All offsets refer to the original rendered prompt. No text is re-rendered,
    no synthetic token is inserted, and image placeholders keep their offsets.
    The engine additionally rejects ranges that intersect multimodal tokens.
    """
    if not blocks:
        return []
    user = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    ending = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    opening = tokenizer.encode(OCR_OPEN, add_special_tokens=False)
    closing = tokenizer.encode(OCR_CLOSE, add_special_tokens=False)
    if tokens[user_start : user_start + len(user)] != user:
        raise ValueError("OCR retention requires the Qwen user-message boundary")
    user_end = _find(tokens, ending, user_start + len(user), limit)
    result = [(user_start, user_start + len(user))]
    cursor = user_start + len(user)
    for block in blocks:
        lo = _find(tokens, opening, cursor, user_end)
        hi = _find(tokens, closing, lo + len(opening), user_end) + len(closing)
        # Match the full block rather than accepting a marker in image data or
        # a different user message. Markers are special tokens in our tokenizer.
        expected = tokenizer.encode(block, add_special_tokens=False)
        if tokens[lo:hi] != expected:
            raise ValueError("rendered OCR tokens differ from the requested OCR block")
        result.append((lo, hi))
        cursor = hi
    # The image may follow the OCR, so do not retain the span between the last
    # OCR block and the user delimiter. Keep only the delimiter and its newline.
    tail_end = user_end + len(ending)
    ending_line = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    if tokens[user_end : user_end + len(ending_line)] == ending_line:
        tail_end = user_end + len(ending_line)
    result.append((user_end, tail_end))
    return [(start, end) for start, end in result if start < end]
