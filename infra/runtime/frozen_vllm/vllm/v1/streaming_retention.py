# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation for explicitly retained ranges in streaming input tokens."""

from collections.abc import Iterable


def validate_retained_ranges(
    ranges: Iterable[tuple[int, int]],
    start: int,
    end: int,
    forbidden: Iterable[tuple[int, int]] = (),
) -> tuple[tuple[int, int], ...]:
    """Require ordered, disjoint text ranges inside one input/turn.

    Retained text must not keep even part of an evicted multimodal placeholder.
    Adjacent ranges are allowed; compaction coalesces them before copying KV.
    """
    result = tuple(tuple(pair) for pair in ranges)
    forbidden = tuple(forbidden)
    cursor = start
    for lo, hi in result:
        if not cursor <= lo < hi <= end:
            raise ValueError(
                "retained token ranges must be ordered, disjoint and within "
                "their streaming input/turn"
            )
        if any(
            lo < other_end and other_start < hi for other_start, other_end in forbidden
        ):
            raise ValueError("retained text ranges cannot overlap multimodal tokens")
        cursor = hi
    return result
